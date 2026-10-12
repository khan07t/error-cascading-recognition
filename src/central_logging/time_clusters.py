"""Model 1: temporal clustering of alerts with DBSCAN on alertBegin."""

import pandas as pd
from sklearn.cluster import DBSCAN

from .config import CATEGORY_CODES, TIME_EPS_SECONDS, TIME_MIN_SAMPLES


def to_ns(values):
    return pd.to_datetime(values, utc=True).dt.as_unit("ns").astype("int64")


def preprocess(df):
    """Clean one alert export and add the time features."""
    df = df.drop(columns=["alertIsTpm", "alertTotalDuration"], errors="ignore").copy()
    df["alertBegin"] = to_ns(df["alertBegin"])
    df["alertEnd"] = to_ns(df["alertEnd"])
    df["alertTotalDuration"] = df["alertEnd"] - df["alertBegin"]
    df["alertCategory"] = df["alertCategory"].map(CATEGORY_CODES)
    df["alertBegin_hour"] = pd.to_datetime(df["alertBegin"], unit="ns").dt.hour
    return df


def cluster_by_hour(df, eps_seconds=TIME_EPS_SECONDS, min_samples=TIME_MIN_SAMPLES):
    """DBSCAN on start time, run separately inside each hour of the day.

    Alerts that start within eps_seconds of each other end up in one cluster.
    DBSCAN marks isolated alerts as -1. alert_combined joins hour and cluster,
    because cluster 3 at 14:00 and cluster 3 at 02:00 are different events.
    """
    parts = []
    for _, hour_df in df.groupby("alertBegin_hour", sort=False):
        hour_df = hour_df.copy()
        seconds = (hour_df["alertBegin"] - hour_df["alertBegin"].min()) / 1e9
        dbscan = DBSCAN(eps=eps_seconds, min_samples=min_samples)
        hour_df["Cluster"] = dbscan.fit_predict(seconds.to_numpy().reshape(-1, 1))
        parts.append(hour_df)
    df = pd.concat(parts, ignore_index=True)
    df["alert_combined"] = df["alertBegin_hour"].astype(str) + "_" + df["Cluster"].astype(str)
    return df


def run_file(src, dst, eps_seconds=TIME_EPS_SECONDS, min_samples=TIME_MIN_SAMPLES):
    df = cluster_by_hour(preprocess(pd.read_csv(src)), eps_seconds, min_samples)
    df.to_csv(dst, index=False)
    return df
