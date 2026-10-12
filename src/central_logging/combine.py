"""Model 3: temporal and text clusters combined, PCA, incremental MiniBatchKMeans."""

from datetime import datetime

import joblib
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, StandardScaler

from .config import KMEANS_BATCH_SIZE, KMEANS_CLUSTERS, MODEL3_FEATURES, PCA_COMPONENTS


def _to_text(ns):
    return pd.to_datetime(ns, unit="ns").dt.strftime("%Y-%m-%d %H:%M:%S.%f") + " +00:00"


def run_file(text_csv, time_csv, output_dir, model_dir, seed=42):
    text = pd.read_csv(text_csv).rename(columns={"alertID": "alertId"})
    time = pd.read_csv(time_csv)
    data = pd.merge(text, time, on="alertId")

    encoder = LabelEncoder()
    data["alert_combined"] = encoder.fit_transform(data["alert_combined"])
    data["alertPlc_encoded"] = encoder.fit_transform(data["alertPlc"])

    kmeans_file = model_dir / "incremental_kmeans_model.pkl"
    scaler_file = model_dir / "scaler.pkl"
    pca_file = model_dir / "pca_model.pkl"
    kmeans = joblib.load(kmeans_file) if kmeans_file.exists() else MiniBatchKMeans(
        n_clusters=KMEANS_CLUSTERS, batch_size=KMEANS_BATCH_SIZE, n_init="auto", random_state=seed)
    scaler = joblib.load(scaler_file) if scaler_file.exists() else StandardScaler()
    pca = joblib.load(pca_file) if pca_file.exists() else PCA(n_components=PCA_COMPONENTS)
    fitted = scaler_file.exists()

    written = []
    for plc in data["alertPlc"].unique():
        plc_data = data[data["alertPlc"] == plc].copy()
        features = plc_data[MODEL3_FEATURES]
        scaled = scaler.transform(features) if fitted else scaler.fit_transform(features)
        reduced = pca.transform(scaled) if fitted else pca.fit_transform(scaled)
        kmeans.partial_fit(reduced)
        plc_data["Final_Cluster"] = kmeans.predict(reduced)
        plc_data["alertBegin"] = _to_text(plc_data["alertBegin"])
        plc_data["alertEnd"] = _to_text(plc_data["alertEnd"])
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = text_csv.stem if text_csv.stem.startswith(str(plc)) else f"{plc}_{text_csv.stem}"
        path = output_dir / f"clustered_output_{name}_{stamp}.csv"
        plc_data.to_csv(path, index=False)
        written.append(path)

    joblib.dump(kmeans, kmeans_file)
    joblib.dump(scaler, scaler_file)
    joblib.dump(pca, pca_file)
    return written
