"""In-memory cosine-similarity vector index.

The interface (add / search) matches what a hosted vector database exposes,
so this class can be swapped for Pinecone, Qdrant or pgvector in production.
"""
import numpy as np


class VectorStore:
    def __init__(self):
        self.vectors: np.ndarray | None = None

    def add(self, vectors: np.ndarray) -> None:
        self.vectors = vectors if self.vectors is None else np.vstack([self.vectors, vectors])

    def search(self, query_vec: np.ndarray, top_k: int = 5) -> list[tuple[int, float]]:
        sims = self.vectors @ query_vec
        idx = np.argsort(-sims)[:top_k]
        return [(int(i), float(sims[i])) for i in idx]
