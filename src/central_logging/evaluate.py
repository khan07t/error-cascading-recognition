"""Scoring against the generator's hidden labels. The pipeline itself never reads them."""

import re
from itertools import combinations
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (adjusted_rand_score, completeness_score, davies_bouldin_score,
                             homogeneity_score, silhouette_score)
from sklearn.metrics.cluster import pair_confusion_matrix

from .config import MODEL3_FEATURES

OUTPUT_NAME = re.compile(r"clustered_output_(?P<stem>.+)_\d{8}_\d{6}\.csv$")


def load_outputs(output_dir):
    parts = []
    for path in sorted(Path(output_dir).glob("clustered_output_*.csv")):
        df = pd.read_csv(path)
        df["source"] = OUTPUT_NAME.match(path.name).group("stem")
        parts.append(df)
    return pd.concat(parts, ignore_index=True)


def _singletons(labels, mask):
    """Rows outside mask become their own one-element group."""
    labels = pd.Series(labels).astype(str).to_numpy(dtype=object)
    idx = np.flatnonzero(~mask)
    labels[idx] = [f"single_{i}" for i in idx]
    return labels


def pair_scores(truth, pred):
    (tn, fp), (fn, tp) = pair_confusion_matrix(truth, pred) // 2
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"pair_precision": precision, "pair_recall": recall, "pair_f1": f1,
            "ari": adjusted_rand_score(truth, pred)}


def score_model1(out, truth):
    """Model 1 against the cascades, seen one PLC file at a time."""
    df = out.merge(truth[["alertId", "cascade_id", "cascade_role"]], on="alertId")
    fragment = df["cascade_id"].astype(str) + "|" + df["source"]
    in_chain = (df["cascade_id"] >= 0) & (fragment.map(fragment.value_counts()) >= 2)
    clustered = df["Cluster"] >= 0
    truth_labels = _singletons(fragment, in_chain.to_numpy())
    pred_labels = _singletons(df["source"] + "|" + df["alert_combined"].astype(str), clustered.to_numpy())
    scores = pair_scores(truth_labels, pred_labels)
    scores.update({
        "alerts": len(df),
        "chain_alerts": int(in_chain.sum()),
        "clustered_alerts": int(clustered.sum()),
        "chain_alerts_found": float((clustered & in_chain).sum() / in_chain.sum()),
        "clustered_that_are_chain": float((clustered & in_chain).sum() / max(clustered.sum(), 1)),
        "clustered_that_are_repeats": float((clustered & (df["cascade_role"] == "repeat")).sum()
                                            / max(clustered.sum(), 1)),
    })
    return scores


def score_model2(out, truth):
    """Model 2 against the fault family that produced each message."""
    df = out.merge(truth[["alertId", "family"]], on="alertId")
    labelled = df["Text_Cluster"] >= 0
    return {
        "alerts": len(df),
        "text_clusters": int(df.loc[labelled, "Text_Cluster"].nunique()),
        "noise_share": float(1 - labelled.mean()),
        "distinct_texts": int(df["alertText"].nunique()),
        "ari": adjusted_rand_score(df["family"], df["Text_Cluster"]),
        "homogeneity": homogeneity_score(df["family"], df["Text_Cluster"]),
        "completeness": completeness_score(df["family"], df["Text_Cluster"]),
    }


def model3_features(out, model_dir):
    df = out.copy()
    df["alertBegin"] = pd.to_datetime(df["alertBegin"], utc=True).dt.as_unit("ns").astype("int64")
    scaler = joblib.load(Path(model_dir) / "scaler.pkl")
    pca = joblib.load(Path(model_dir) / "pca_model.pkl")
    return pca.transform(scaler.transform(df[MODEL3_FEATURES]))


def score_model3(out, truth, model_dir, sample=20_000, seed=0):
    df = out.merge(truth[["alertId", "family"]], on="alertId")
    x = model3_features(df, model_dir)
    labels = df["Final_Cluster"].to_numpy()
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(df), size=min(sample, len(df)), replace=False)
    return {
        "alerts": len(df),
        "final_clusters": int(df["Final_Cluster"].nunique()),
        "silhouette": float(silhouette_score(x[pick], labels[pick])),
        "davies_bouldin": float(davies_bouldin_score(x, labels)),
        "ari_vs_family": adjusted_rand_score(df["family"], labels),
    }


def model3_profile(out, truth):
    df = out.merge(truth[["alertId", "family"]], on="alertId")
    rows = []
    for k, g in df.groupby("Final_Cluster"):
        fam = g["family"].value_counts(normalize=True)
        rows.append({"Final_Cluster": k, "alerts": len(g), "share": len(g) / len(df),
                     "top_family": fam.index[0], "top_family_share": fam.iloc[0],
                     "band_stop_share": g["alertIsBandStop"].mean(),
                     "median_duration_s": g["alertTotalDuration"].median() / 1e9,
                     "in_temporal_cluster": (g["Cluster"] >= 0).mean(),
                     "median_hour_utc": g["alertBegin_hour"].median(),
                     "median_text_cluster": g["Text_Cluster"].median(),
                     "plcs": g["alertPlc"].nunique()})
    return pd.DataFrame(rows)


def neighbours(plant, max_gap=2):
    pos = plant.set_index("alertPlc")[["line", "position"]]
    def near(a, b):
        return pos.at[a, "line"] == pos.at[b, "line"] and abs(pos.at[a, "position"] - pos.at[b, "position"]) <= max_gap
    return near


def score_multi(out, truth, plant):
    """Multi-PLC clusters: real cross-PLC cascade links or coincidences?"""
    df = out.merge(truth[["alertId", "cascade_id"]], on="alertId")
    df["key"] = df["source"] + "|" + df["cluster"].astype(str)
    near = neighbours(plant)
    rows = []
    for key, g in df[df["mixedAlertPlc"]].groupby("key"):
        plcs = sorted(g["alertPlc"].unique())
        linked = g[g["cascade_id"] >= 0].groupby("cascade_id")["alertPlc"].nunique()
        rows.append({"key": key, "alerts": len(g), "plcs": len(plcs),
                     "true_link": bool((linked >= 2).any()),
                     "all_neighbours": all(near(a, b) for a, b in combinations(plcs, 2))})
    mixed = pd.DataFrame(rows)
    clusters = df.groupby("key").size()
    coincidences = mixed[~mixed["true_link"]]
    return {
        "alerts_kept": len(df),
        "clusters": int(len(clusters)),
        "mixed_plc_clusters": int(len(mixed)),
        "mixed_true_cascade": float(mixed["true_link"].mean()),
        "mixed_coincidence": float(1 - mixed["true_link"].mean()),
        "coincidences_far_apart": float((~coincidences["all_neighbours"]).mean()) if len(coincidences) else 0.0,
        "true_links_far_apart": float((~mixed.loc[mixed["true_link"], "all_neighbours"]).mean()),
    }, mixed


def sweep_model1(alerts, truth, days, settings):
    """Model 1 alone on the daily per-PLC files, for several (eps_seconds, min_samples) pairs."""
    from .generate import alert_day
    from .time_clusters import cluster_by_hour, preprocess

    day = alert_day(alerts)
    files = []
    for (plc, d), g in alerts[day.isin(days)].groupby(["alertPlc", day]):
        files.append(preprocess(g).assign(source=f"{plc}_{d.replace('-', '')}"))
    rows = []
    for eps, min_samples in settings:
        out = pd.concat([cluster_by_hour(f, eps, min_samples) for f in files], ignore_index=True)
        rows.append({"eps_seconds": eps, "min_samples": min_samples, **score_model1(out, truth)})
    return pd.DataFrame(rows)
