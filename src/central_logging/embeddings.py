"""BERT embeddings of alertText (input to Model 2)."""

import os
from pathlib import Path

import numpy as np

from .config import BERT_BATCH_SIZE, BERT_MAX_LENGTH, BERT_MODEL


class TextEmbedder:
    """bert-base-uncased, CLS token, batches of 256, offline from a local cache.

    Alert texts repeat a lot, so each distinct text is encoded once per run and
    the vector is reused for every row with that text.
    """

    def __init__(self, model_name=BERT_MODEL, cache_dir=None, device=None):
        import torch
        from transformers import BertModel, BertTokenizer

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if Path(model_name).exists():
            kwargs = {}
        else:
            kwargs = {"cache_dir": str(cache_dir) if cache_dir else None}
            if os.environ.get("ECR_OFFLINE", "1") == "1":
                kwargs["local_files_only"] = True
        try:
            self.tokenizer = BertTokenizer.from_pretrained(model_name, **kwargs)
            self.model = BertModel.from_pretrained(model_name, **kwargs)
        except OSError:
            if not kwargs.get("local_files_only"):
                raise
            kwargs.pop("local_files_only")       # not in the local cache yet: download once
            self.tokenizer = BertTokenizer.from_pretrained(model_name, **kwargs)
            self.model = BertModel.from_pretrained(model_name, **kwargs)
        self.model.to(self.device).eval()
        self.cache = {}

    def _encode(self, texts):
        out = []
        for i in range(0, len(texts), BERT_BATCH_SIZE):
            batch = texts[i:i + BERT_BATCH_SIZE]
            inputs = self.tokenizer(batch, return_tensors="pt", truncation=True, padding=True,
                                    max_length=BERT_MAX_LENGTH)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with self.torch.no_grad():
                out.append(self.model(**inputs).last_hidden_state[:, 0, :].cpu().numpy())
        return np.vstack(out)

    def embed(self, texts):
        texts = [str(t) for t in texts]
        new = sorted({t for t in texts if t not in self.cache})
        if new:
            for t, v in zip(new, self._encode(new)):
                self.cache[t] = v
        return np.vstack([self.cache[t] for t in texts])


def run_file(src_csv, dst_npy, embedder):
    import pandas as pd

    df = pd.read_csv(src_csv)
    vectors = embedder.embed(df["alertText"].astype(str).tolist())
    np.save(dst_npy, np.hstack([df["alertId"].to_numpy().reshape(-1, 1), vectors]))
    return vectors
