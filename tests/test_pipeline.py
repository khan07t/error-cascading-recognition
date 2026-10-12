"""The test cases from Project II: preprocessing, temporal clustering, text clustering,
fusion, multi-PLC and the run ledger. BERT is replaced by a small stub so the tests
run in seconds without the model download."""

import hashlib

import numpy as np
import pandas as pd
import pytest

from central_logging import combine, generate, main, multi_plc, time_clusters
from central_logging.config import Paths
from central_logging.text_clusters import TextClusterer, make_autoencoder, train_text_model


class StubEmbedder:
    """Deterministic 16-dim vectors per text, standing in for BERT."""

    def embed(self, texts):
        out = []
        for t in texts:
            seed = int(hashlib.md5(str(t).encode()).hexdigest()[:8], 16)
            out.append(np.random.default_rng(seed).normal(size=16))
        return np.vstack(out)


def export(rows, plc="A1ROB01"):
    base = pd.Timestamp("2024-04-09 10:00:00")
    df = pd.DataFrame([{
        "alertId": 1000 + i,
        "alertBegin": (base + pd.Timedelta(seconds=s)).strftime("%Y-%m-%d %H:%M:%S.%f") + " +02:00",
        "alertEnd": (base + pd.Timedelta(seconds=s + 2)).strftime("%Y-%m-%d %H:%M:%S.%f") + " +02:00",
        "alertTotalDuration": "00:00:02", "alertIsBandStop": 0, "alertIsTpm": 0,
        "alertCategory": cat, "alertText": text, "alertPlc": p or plc,
    } for i, (s, text, cat, p) in enumerate(rows)])
    return df


def test_preprocess_features():
    df = time_clusters.preprocess(export([(0, "Ventil 1 ohne Rueckmeldung", "organizational", None)]))
    assert "alertIsTpm" not in df
    assert df.loc[0, "alertTotalDuration"] == 2_000_000_000
    assert df.loc[0, "alertCategory"] == 1
    assert df.loc[0, "alertBegin_hour"] == 8          # 10:00 at +02:00 is 08:00 UTC


def test_close_alerts_form_one_group():
    rows = [(i * 0.05, f"text {i}", "technical", None) for i in range(5)] + [(30, "alone", "technical", None)]
    df = time_clusters.cluster_by_hour(time_clusters.preprocess(export(rows)), eps_seconds=0.2, min_samples=3)
    grouped = df[df["alertText"] != "alone"]
    assert grouped["Cluster"].nunique() == 1 and (grouped["Cluster"] >= 0).all()
    assert df.loc[df["alertText"] == "alone", "Cluster"].item() == -1
    assert df["alert_combined"].str.startswith("8_").all()


def test_autoencoder_latent_size():
    import torch

    model = make_autoencoder(16)
    assert model.encoder(torch.zeros(4, 16)).shape == (4, 10)


@pytest.fixture
def app(tmp_path):
    paths = Paths(tmp_path / "app").make()
    rng = np.random.default_rng(0)
    centers = rng.normal(size=(4, 16)) * 5
    sets = [(f"plc{i}", np.vstack([c + rng.normal(scale=0.1, size=(40, 16)) for c in centers]))
            for i in range(2)]
    train_text_model(sets, paths.text_model, epochs=3)
    return paths


def test_text_model_labels_every_alert(app):
    clusterer = TextClusterer(app.text_model)
    labels = clusterer.predict(StubEmbedder().embed([f"t{i}" for i in range(25)]))
    assert len(labels) == 25


def test_fusion_keeps_every_alert(app):
    rows = [(i * 0.05, f"text {i % 3}", "technical", None) for i in range(30)]
    time_df = time_clusters.cluster_by_hour(time_clusters.preprocess(export(rows)), min_samples=3)
    time_df.to_csv(app.time_data / "x.csv", index=False)
    labels = TextClusterer(app.text_model).predict(StubEmbedder().embed(time_df["alertText"]))
    pd.DataFrame({"alertId": time_df["alertId"], "Text_Cluster": labels}).to_csv(app.text_clusters / "x.csv", index=False)
    written = combine.run_file(app.text_clusters / "x.csv", app.time_data / "x.csv", app.output, app.kmeans_model)
    out = pd.read_csv(written[0])
    assert len(out) == 30 and out["Final_Cluster"].notna().all()
    assert (app.kmeans_model / "incremental_kmeans_model.pkl").exists()


def test_multi_plc_flags_alerts_from_two_plcs():
    rows = [(0.00, "a", "technical", "A1ROB01"), (0.03, "b", "technical", "A1SWE01"),
            (10.0, "c", "technical", "A1ROB01")]
    out = multi_plc.cluster(export(rows))
    assert len(out) == 2                            # the alert 10 s later has no neighbour
    assert out["mixedAlertPlc"].all()


def test_ledger_failure_and_resume(app):
    good = export([(i * 0.05, f"text {i % 3}", "technical", None) for i in range(20)])
    bad = good.copy()
    bad["alertBegin"] = "not a time"
    good.to_csv(app.input / "A1ROB01_20240409.csv", index=False)
    bad.to_csv(app.input / "A1ROB02_20240409.csv", index=False)

    ledger = main.run(app.root, embedder=StubEmbedder(), min_samples=3).set_index("TO DO")
    assert ledger.loc["A1ROB01_20240409", "DONE"] == "Passed"
    assert ledger.loc["A1ROB02_20240409", "DONE"] == "Failed"
    assert ledger.loc["A1ROB02_20240409", "PROCESS TIME"].startswith("Failed")
    assert (app.errors / "A1ROB02_20240409.csv").exists()
    assert not (app.input / "A1ROB01_20240409.csv").exists()

    # an interrupted run resumes without repeating finished stages
    good.to_csv(app.input / "A1ROB03_20240409.csv", index=False)
    main.run(app.root, embedder=StubEmbedder(), min_samples=3, stop_after="GENERATE EMBEDDINGS")
    first = pd.read_csv(app.ledger).set_index("TO DO").loc["A1ROB03_20240409", "PROCESS TIME"]
    ledger = main.run(app.root, embedder=StubEmbedder(), min_samples=3).set_index("TO DO")
    assert ledger.loc["A1ROB03_20240409", "PROCESS TIME"] == first
    assert ledger.loc["A1ROB03_20240409", "DONE"] == "Passed"


def test_generator_labels_line_up():
    alerts, truth, plant = generate.generate(days=2, background_per_day=500, cascades_per_day=20,
                                             bursts_per_day=5)
    assert alerts["alertId"].is_monotonic_increasing and alerts["alertId"].is_unique
    assert (alerts["alertId"].values == truth["alertId"].values).all()
    assert set(alerts["alertPlc"]) <= set(plant["alertPlc"])
    assert list(alerts.columns) == generate.EXPORT_COLUMNS
