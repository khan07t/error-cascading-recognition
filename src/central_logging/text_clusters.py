"""Model 2: autoencoder on the BERT embeddings, then HDBSCAN (DEC step + clustering)."""

import logging

import joblib
import numpy as np
import pandas as pd

from .config import (AE_BATCH_SIZE, AE_EPOCHS, AE_HIDDEN, AE_LATENT, AE_LR,
                     HDBSCAN_MIN_CLUSTER_SIZE, HDBSCAN_MIN_SAMPLES, TEXT_BATCH_SIZE)

log = logging.getLogger(__name__)


def make_autoencoder(input_dim, latent_dim=AE_LATENT):
    from torch import nn

    class Autoencoder(nn.Module):
        def __init__(self):
            super().__init__()
            h1, h2 = AE_HIDDEN
            self.encoder = nn.Sequential(nn.Linear(input_dim, h1), nn.ReLU(), nn.Linear(h1, h2),
                                         nn.ReLU(), nn.Linear(h2, latent_dim))
            self.decoder = nn.Sequential(nn.Linear(latent_dim, h2), nn.ReLU(), nn.Linear(h2, h1),
                                         nn.ReLU(), nn.Linear(h1, input_dim))

        def forward(self, x):
            return self.decoder(self.encoder(x))

    return Autoencoder()


def train_autoencoder(model, x, epochs=AE_EPOCHS, batch_size=AE_BATCH_SIZE, lr=AE_LR):
    """MSE reconstruction, Adam, mini-batches in file order. Returns the last loss per epoch."""
    import torch

    data = torch.tensor(x, dtype=torch.float32)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    losses = []
    for _ in range(epochs):
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            optimizer.zero_grad()
            loss = criterion(model(batch), batch)
            loss.backward()
            optimizer.step()
        losses.append(float(loss.item()))
    model.eval()
    return losses


def encode(model, x):
    import torch

    with torch.no_grad():
        return model.encoder(torch.tensor(x, dtype=torch.float32)).numpy()


def train_text_model(embedding_sets, model_dir, epochs=AE_EPOCHS, hdbscan_rows=TEXT_BATCH_SIZE, seed=42):
    """Project 1 training on history, one PLC after another.

    The scaler is fitted on the first PLC. The autoencoder is trained on the
    first PLC and then fine-tuned on each further PLC instead of being
    retrained. HDBSCAN is fitted on the latent space of the history.
    """
    import hdbscan
    import torch
    from sklearn.preprocessing import StandardScaler

    torch.manual_seed(seed)
    model_dir.mkdir(parents=True, exist_ok=True)
    scaler, model, history = None, None, []
    for name, emb in embedding_sets:
        if scaler is None:
            scaler = StandardScaler().fit(emb)
            model = make_autoencoder(emb.shape[1])
        losses = train_autoencoder(model, scaler.transform(emb), epochs=epochs)
        history.append({"plc_file": name, "rows": len(emb), "first_epoch_loss": losses[0],
                        "last_epoch_loss": losses[-1]})
        log.info("autoencoder updated on %s (%d rows), loss %.4f", name, len(emb), losses[-1])
    latent = np.vstack([encode(model, scaler.transform(emb)) for _, emb in embedding_sets])[:hdbscan_rows]
    clusterer = hdbscan.HDBSCAN(min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
                                min_samples=HDBSCAN_MIN_SAMPLES, prediction_data=True)
    clusterer.fit(latent)
    joblib.dump(scaler, model_dir / "scaler.pkl")
    torch.save(model.state_dict(), model_dir / "autoencoder.pth")
    joblib.dump(clusterer, model_dir / "hdbscan_model.pkl")
    return pd.DataFrame(history), clusterer


class TextClusterer:
    """Loads the saved scaler, autoencoder and HDBSCAN model and labels new alerts."""

    def __init__(self, model_dir):
        import torch

        self.model_dir = model_dir
        self.scaler = joblib.load(model_dir / "scaler.pkl")
        self.model = make_autoencoder(self.scaler.n_features_in_)
        self.model.load_state_dict(torch.load(model_dir / "autoencoder.pth"))
        self.model.eval()
        path = model_dir / "hdbscan_model.pkl"
        self.clusterer = joblib.load(path) if path.exists() else None

    def predict(self, embeddings, batch_size=TEXT_BATCH_SIZE):
        import hdbscan
        from hdbscan import approximate_predict

        labels = []
        for i in range(0, len(embeddings), batch_size):
            latent = encode(self.model, self.scaler.transform(embeddings[i:i + batch_size]))
            if self.clusterer is None:
                self.clusterer = hdbscan.HDBSCAN(min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
                                                 min_samples=HDBSCAN_MIN_SAMPLES, prediction_data=True)
                labels.append(self.clusterer.fit_predict(latent))
                joblib.dump(self.clusterer, self.model_dir / "hdbscan_model.pkl")
            else:
                labels.append(approximate_predict(self.clusterer, latent)[0])
        return np.concatenate(labels) if labels else np.array([], dtype=int)


def run_file(src_npy, dst_csv, clusterer):
    data = np.load(src_npy)
    labels = clusterer.predict(data[:, 1:])
    out = pd.DataFrame({"alertId": data[:, 0].astype(np.int64), "Text_Cluster": labels.astype(int)})
    out.to_csv(dst_csv, index=False)
    return out
