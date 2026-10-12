"""Central Logging: runs the four stages on every file in Input_PLC.

Each input file gets a row in Logs.csv. Every stage writes "Passed (seconds)"
or "Failed: reason" for that file. A file that passes all stages is marked
DONE and its input and intermediate files are removed. A file that fails is
moved to Error_Files; stages that already passed are skipped when it is put
back and run again. Files in Input_Multi_PLC go through the multi-PLC module.
"""

import argparse
import logging
import shutil
import sys
import time
from datetime import datetime, timedelta

import pandas as pd

from . import combine, embeddings, multi_plc, text_clusters, time_clusters
from .config import LEDGER_COLUMNS, LOG_RETENTION_DAYS, STAGES, Paths

log = logging.getLogger("central_logging")


def setup_logging(paths):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    file_handler = logging.FileHandler(
        paths.run_logs / f"central_logging_{datetime.now():%Y%m%d_%H%M%S}.log")
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logger.addHandler(file_handler)
    logger.addHandler(console)


def load_ledger(paths, now):
    if not paths.ledger.exists():
        return pd.DataFrame(columns=LEDGER_COLUMNS)
    ledger = pd.read_csv(paths.ledger, dtype=str)
    stamps = pd.to_datetime(ledger["TIMESTAMP"], errors="coerce")
    keep = stamps >= now - timedelta(days=LOG_RETENTION_DAYS)
    if (~keep).any():
        log.info("Removed %d ledger rows older than %d days", (~keep).sum(), LOG_RETENTION_DAYS)
    return ledger[keep].reset_index(drop=True)


def intermediate(paths, stem, stage):
    return {
        "PROCESS TIME": paths.time_data / f"{stem}.csv",
        "GENERATE EMBEDDINGS": paths.embeddings / f"{stem}.npy",
        "PROCESS EMBEDDINGS": paths.text_clusters / f"{stem}.csv",
    }.get(stage)


def run_multi_plc(paths):
    for src in sorted(paths.multi_input.glob("*.csv")):
        try:
            out = multi_plc.run_file(src, paths.multi_output)
            src.unlink()
            log.info("Multi-PLC: processed %s -> %s", src.name, out.name)
        except Exception as exc:  # keep going with the other files
            log.error("Multi-PLC: failed on %s: %s", src.name, exc)


def run(root, embedder=None, stop_after=None, now=None, min_samples=None):
    """One run of the application. Returns the ledger rows of this run."""
    paths = Paths(root).make()
    now = now or datetime.now()
    if not logging.getLogger().handlers:
        setup_logging(paths)
    log.info("Starting central logging pipeline at %s", now)

    run_multi_plc(paths)

    files = sorted(paths.input.glob("*.csv"))
    if not files:
        log.warning("No CSV files found in Input_PLC.")
        return pd.DataFrame(columns=LEDGER_COLUMNS)
    backup = paths.input / "backup" / f"{now:%Y%m%d_%H%M%S}"
    backup.mkdir(parents=True, exist_ok=True)
    for f in files:
        shutil.copy2(f, backup / f.name)
    log.info("Found %d files, backup in %s", len(files), backup.relative_to(paths.root))

    ledger = load_ledger(paths, now)
    rows = {}
    for f in files:
        open_rows = ledger.index[(ledger["TO DO"] == f.stem) & (ledger["DONE"] != "Passed")]
        if len(open_rows):
            rows[f.stem] = open_rows[-1]
        else:
            ledger.loc[len(ledger), ["TO DO", "TIMESTAMP"]] = [f.stem, f"{now:%Y-%m-%d %H:%M:%S}"]
            rows[f.stem] = ledger.index[-1]
    ledger.to_csv(paths.ledger, index=False)

    ok = {f.stem: True for f in files}
    tools = {}
    for stage in STAGES:
        for f in files:
            stem, i = f.stem, rows[f.stem]
            if not ok[stem]:
                continue
            done_before = str(ledger.at[i, stage]).startswith("Passed")
            out = intermediate(paths, stem, stage)
            if done_before and (out is None or out.exists()):
                continue
            start = time.perf_counter()
            try:
                if stage == "PROCESS TIME":
                    time_clusters.run_file(f, out, **({} if min_samples is None else {"min_samples": min_samples}))
                elif stage == "GENERATE EMBEDDINGS":
                    if "embedder" not in tools:
                        tools["embedder"] = embedder or embeddings.TextEmbedder(cache_dir=paths.hf_cache)
                    embeddings.run_file(intermediate(paths, stem, "PROCESS TIME"), out, tools["embedder"])
                elif stage == "PROCESS EMBEDDINGS":
                    if "text" not in tools:
                        tools["text"] = text_clusters.TextClusterer(paths.text_model)
                    text_clusters.run_file(intermediate(paths, stem, "GENERATE EMBEDDINGS"), out, tools["text"])
                else:
                    combine.run_file(intermediate(paths, stem, "PROCESS EMBEDDINGS"),
                                     intermediate(paths, stem, "PROCESS TIME"), paths.output, paths.kmeans_model)
                ledger.at[i, stage] = f"Passed ({time.perf_counter() - start:.2f}s)"
            except Exception as exc:
                reason = str(exc).splitlines()[0] if str(exc) else type(exc).__name__
                ok[stem] = False
                ledger.at[i, stage] = f"Failed: {reason}"
                log.error("%s failed on %s: %s", stage, f.name, reason)
        ledger.to_csv(paths.ledger, index=False)
        log.info("%s finished for %d of %d files", stage, sum(ok.values()), len(files))
        if stage == stop_after:
            log.warning("Run stopped after %s", stage)
            return ledger.loc[list(rows.values())]

    for f in files:
        stem, i = f.stem, rows[f.stem]
        if ok[stem]:
            ledger.at[i, "DONE"] = "Passed"
            f.unlink()
            for stage in STAGES[:3]:
                intermediate(paths, stem, stage).unlink(missing_ok=True)
        else:
            ledger.at[i, "DONE"] = "Failed"
            shutil.move(str(f), paths.errors / f.name)
            log.warning("Moved %s to Error_Files", f.name)
    ledger.to_csv(paths.ledger, index=False)
    log.info("Run complete: %d passed, %d failed", sum(ok.values()), len(ok) - sum(ok.values()))
    return ledger.loc[list(rows.values())]


def cli():
    parser = argparse.ArgumentParser(description="Central Logging: error cascade recognition")
    parser.add_argument("root", nargs="?", default=".", help="application folder (contains Input_PLC)")
    args = parser.parse_args()
    run(args.root)


if __name__ == "__main__":
    cli()
