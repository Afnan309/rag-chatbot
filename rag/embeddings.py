"""Embedding models (the dense half of hybrid search).

* OpenAIEmbedder  - text-embedding-3-small through the OpenAI API (used when
                    OPENAI_API_KEY is set).
* LocalEmbedder   - offline semantic vectors (TF-IDF + truncated SVD / LSA).
                    Runs with no API key, which keeps the demo free to run.
"""
import os

import numpy as np
import requests
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer

from .text import split_sentences, tokenize


class LocalEmbedder:
    name = "local-lsa"

    def __init__(self, dims: int = 96):
        self.dims = dims
        self.model = None

    def fit(self, texts: list[str]) -> "LocalEmbedder":
        dims = min(self.dims, len(texts) - 1)
        self.model = make_pipeline(
            TfidfVectorizer(tokenizer=tokenize, token_pattern=None, ngram_range=(1, 2),
                            sublinear_tf=True, min_df=1),
            TruncatedSVD(n_components=dims, random_state=42),
            Normalizer(copy=False),
        )
        # Fit on passages AND their sentences: more co-occurrence signal for LSA.
        sentences = [s for t in texts for s in split_sentences(t)]
        self.model.fit(texts + sentences)
        return self

    def embed(self, texts: list[str]) -> np.ndarray:
        return self.model.transform(texts).astype(np.float32)


class OpenAIEmbedder:
    name = "openai/text-embedding-3-small"

    def __init__(self, model: str = "text-embedding-3-small"):
        self.model = model
        self.key = os.environ["OPENAI_API_KEY"]

    def fit(self, texts: list[str]) -> "OpenAIEmbedder":
        return self

    def embed(self, texts: list[str]) -> np.ndarray:
        resp = requests.post(
            "https://api.openai.com/v1/embeddings",
            headers={"Authorization": f"Bearer {self.key}"},
            json={"model": self.model, "input": texts},
            timeout=60,
        )
        resp.raise_for_status()
        vecs = np.array([d["embedding"] for d in resp.json()["data"]], dtype=np.float32)
        return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


def get_embedder():
    return OpenAIEmbedder() if os.getenv("OPENAI_API_KEY") else LocalEmbedder()
