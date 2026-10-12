"""Folder layout and model settings of the Central Logging application."""

import os
from dataclasses import dataclass
from pathlib import Path

# Model 1: DBSCAN on alertBegin, inside each hour
TIME_EPS_SECONDS = 0.2          # 200 ms between consecutive alerts of one group
TIME_MIN_SAMPLES = 7
CATEGORY_CODES = {"technical": 0, "organizational": 1, "other": 2}

# Model 2: BERT, autoencoder, HDBSCAN
BERT_MODEL = os.environ.get("ECR_BERT_MODEL", "bert-base-uncased")
BERT_BATCH_SIZE = 256
BERT_MAX_LENGTH = 512
AE_HIDDEN = (500, 200)
AE_LATENT = 10
AE_EPOCHS = 50
AE_BATCH_SIZE = 32
AE_LR = 1e-3
HDBSCAN_MIN_CLUSTER_SIZE = 10
HDBSCAN_MIN_SAMPLES = 10
TEXT_BATCH_SIZE = 100_000

# Model 3: StandardScaler, PCA, MiniBatchKMeans
KMEANS_CLUSTERS = 10
KMEANS_BATCH_SIZE = 100_000
PCA_COMPONENTS = 6
MODEL3_FEATURES = ["Text_Cluster", "alertTotalDuration", "alertIsBandStop", "alertCategory",
                   "alertId", "Cluster", "alertBegin_hour"]

# Multi-PLC mode
MULTI_TOLERANCE_SECONDS = 0.05
MULTI_EPS_SECONDS = 0.05
MULTI_MIN_SAMPLES = 1

LOG_RETENTION_DAYS = 7
LEDGER_COLUMNS = ["TO DO", "PROCESS TIME", "GENERATE EMBEDDINGS", "PROCESS EMBEDDINGS",
                  "COMBINE CLUSTER", "DONE", "TIMESTAMP"]
STAGES = LEDGER_COLUMNS[1:5]


@dataclass
class Paths:
    """Same folders as the deployed application."""

    root: Path

    def __post_init__(self):
        self.root = Path(self.root)
        lib = self.root / "scripts" / "Lib"
        work = lib / "Processing_Files"
        self.input = self.root / "Input_PLC"
        self.output = self.root / "Output_PLC"
        self.multi_input = self.root / "Input_Multi_PLC"
        self.multi_output = self.root / "Output_Multi_PLC"
        self.errors = self.root / "Error_Files"
        self.ledger = lib / "Logs" / "Logs.csv"
        self.run_logs = self.root / "logs"
        self.time_data = work / "PreProcessedTimeData"
        self.embeddings = work / "AlertIDEmbedding"
        self.text_clusters = work / "AlertIDEmbedding" / "ProcessedTextClusters"
        self.text_model = work / "TextEmbeddingModel"
        self.kmeans_model = work / "KMeansModel"
        self.hf_cache = self.root / "scripts" / "hf_cache"

    def make(self):
        for p in [self.input, self.output, self.multi_input, self.multi_output, self.errors,
                  self.ledger.parent, self.run_logs, self.time_data, self.embeddings,
                  self.text_clusters, self.text_model, self.kmeans_model]:
            p.mkdir(parents=True, exist_ok=True)
        return self
