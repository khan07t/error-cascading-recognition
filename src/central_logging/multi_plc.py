"""Multi-PLC mode: alerts from all PLCs that start within ±0.05 s of each other."""

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

from .config import MULTI_EPS_SECONDS, MULTI_MIN_SAMPLES, MULTI_TOLERANCE_SECONDS


def within_tolerance(df, column="alertBegin", tolerance=MULTI_TOLERANCE_SECONDS):
    """Keep only alerts that have a neighbour (any PLC) within the tolerance."""
    df = df.sort_values(by=column).reset_index(drop=True)
    close = df[column].diff().abs() <= pd.Timedelta(seconds=tolerance)
    close.iloc[0] = False
    keep = close | close.shift(-1, fill_value=False)
    return df.loc[df.index[keep]]


def cluster(df):
    df = df.copy()
    df["alertBegin"] = pd.to_datetime(df["alertBegin"], format="mixed", errors="coerce")
    df["alertEnd"] = pd.to_datetime(df["alertEnd"], format="mixed", errors="coerce")
    df = df.dropna(subset=["alertBegin", "alertEnd"])

    out = within_tolerance(df)
    ns = out["alertBegin"].dt.as_unit("ns").astype(np.int64)
    seconds = (ns - ns.min()) / 1e9
    out = out.assign(cluster=DBSCAN(eps=MULTI_EPS_SECONDS, min_samples=MULTI_MIN_SAMPLES)
                     .fit_predict(seconds.to_numpy().reshape(-1, 1)))
    out["mixedAlertPlc"] = out.groupby("cluster")["alertPlc"].transform(lambda x: x.nunique() > 1)
    out["cluster_hour"] = out["alertBegin"].dt.hour
    out["cluster_hour_minute_5min"] = out["alertBegin"].apply(lambda x: f"{x.hour}_{(x.minute // 5) * 5:02d}")
    return out


def run_file(src, output_dir):
    out = cluster(pd.read_csv(src))
    path = output_dir / f"{src.stem}_clustered.csv"
    out.to_csv(path, index=False)
    return path
