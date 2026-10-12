"""Full run on generated data: one week of history, then one week of daily runs.

    python -m central_logging.benchmark --work work --results results

1. Generate 14 days of alerts for 75 PLCs (plus hidden labels).
2. Week 1 is history: BERT embeddings per PLC, scaler, autoencoder trained PLC by
   PLC, HDBSCAN fitted on the latent space. The application then runs once on
   the week-1 files per PLC, which fits the Model 3 scaler, PCA and KMeans.
3. Week 2 is live: every day, that day's export (one file per PLC, plus one
   plant-wide file for the multi-PLC module) goes into the input folders and
   the application runs once.
4. Week-2 outputs are scored against the hidden labels.
"""

import argparse
import json
import shutil
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from . import evaluate
from .config import TIME_MIN_SAMPLES, Paths
from .embeddings import TextEmbedder
from .generate import alert_day, export_daily_files, write_dataset
from .main import run
from .text_clusters import train_text_model


def run_benchmark(work, results, history_days=7, epochs=None, embedder=None, min_samples=None,
                  text_model=None, **generate_kwargs):
    work, results = Path(work), Path(results)
    if work.exists():
        shutil.rmtree(work)
    results.mkdir(parents=True, exist_ok=True)
    alerts, truth, plant = write_dataset(work / "generated", **generate_kwargs)
    days = sorted(alert_day(alerts).unique())
    history, live = days[:history_days], days[history_days:]
    app = Paths(work / "app").make()
    embedder = embedder or TextEmbedder()

    # Project 1: train the text model on the history, PLC by PLC
    hist = alerts[alert_day(alerts).isin(history)]
    if text_model is None:
        sets = [(plc, embedder.embed(g["alertText"].tolist())) for plc, g in sorted(hist.groupby("alertPlc"))]
        kwargs = {} if epochs is None else {"epochs": epochs}
        training, _ = train_text_model(sets, app.text_model, **kwargs)
        training.to_csv(results / "autoencoder_training.csv", index=False)
    else:
        shutil.copytree(text_model, app.text_model, dirs_exist_ok=True)

    # The application on the history files fits Model 3
    for plc, g in hist.groupby("alertPlc"):
        g.to_csv(app.input / f"{plc}_history.csv", index=False)
    run(app.root, embedder=embedder, min_samples=min_samples,
        now=datetime.fromisoformat(history[-1]) + timedelta(days=1, hours=5))
    shutil.move(str(app.output), str(work / "history_output"))
    app.output.mkdir()

    # Week 2: one run per day
    ledgers = []
    for day in live:
        export_daily_files(alerts, day, single_dir=app.input, multi_dir=app.multi_input)
        ledgers.append(run(app.root, embedder=embedder, min_samples=min_samples,
                           now=datetime.fromisoformat(day) + timedelta(days=1, hours=5)))
    tag = f"min_samples_{min_samples or TIME_MIN_SAMPLES}"
    pd.concat(ledgers).to_csv(results / f"ledger_week2_{tag}.csv", index=False)

    out = evaluate.load_outputs(app.output)
    multi = pd.concat([pd.read_csv(p).assign(source=p.stem) for p in sorted(app.multi_output.glob("*.csv"))],
                      ignore_index=True)
    scores = {
        "model1": evaluate.score_model1(out, truth),
        "model2": evaluate.score_model2(out, truth),
        "model3": evaluate.score_model3(out, truth, app.kmeans_model),
    }
    scores["multi_plc"], mixed = evaluate.score_multi(multi, truth, plant)
    (results / f"scores_{tag}.json").write_text(json.dumps(scores, indent=2))
    evaluate.model3_profile(out, truth).to_csv(results / f"model3_profile_{tag}.csv", index=False)
    mixed.to_csv(results / "multi_plc_mixed_clusters.csv", index=False)
    return scores


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", default="work")
    parser.add_argument("--results", default="results")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--min-samples", type=int, default=None)
    parser.add_argument("--text-model", default=None, help="reuse a trained text model folder")
    args = parser.parse_args()
    scores = run_benchmark(args.work, args.results, epochs=args.epochs, min_samples=args.min_samples,
                           text_model=args.text_model)
    print(json.dumps(scores, indent=2))
