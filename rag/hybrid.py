"""Hybrid retriever: BM25 + vector search merged with Reciprocal Rank Fusion."""
from dataclasses import dataclass

from .bm25 import BM25Index
from .chunker import Chunk
from .embeddings import get_embedder
from .reranker import LexicalReranker
from .vector_store import VectorStore


@dataclass
class Hit:
    chunk: Chunk
    score: float
    bm25: float
    vector: float


class HybridRetriever:
    def __init__(self, chunks: list[Chunk], rrf_k: int = 60, embedder=None):
        self.chunks = chunks
        self.rrf_k = rrf_k
        texts = [c.searchable_text for c in chunks]
        self.bm25 = BM25Index(texts)
        self.embedder = (embedder or get_embedder()).fit(texts)
        self.store = VectorStore()
        self.store.add(self.embedder.embed(texts))
        self.reranker = LexicalReranker(self.bm25.idf)

    def keyword_search(self, query: str, top_k: int = 5):
        return self.bm25.search(query, top_k)

    def vector_search(self, query: str, top_k: int = 5):
        q = self.embedder.embed([query])[0]
        return self.store.search(q, top_k)

    def search(self, query: str, top_k: int = 4, pool: int = 12, mode: str = "hybrid",
               rerank: bool = True) -> list[Hit]:
        kw = self.keyword_search(query, pool)
        vec = self.vector_search(query, pool)
        kw_map = dict(kw)
        vec_map = dict(vec)

        if mode == "keyword":
            ranked = [(i, s) for i, s in kw]
        elif mode == "vector":
            ranked = [(i, s) for i, s in vec]
        else:
            fused: dict[int, float] = {}
            for rank, (i, _) in enumerate(kw):
                fused[i] = fused.get(i, 0) + 1 / (self.rrf_k + rank + 1)
            for rank, (i, _) in enumerate(vec):
                fused[i] = fused.get(i, 0) + 1 / (self.rrf_k + rank + 1)
            ranked = sorted(fused.items(), key=lambda x: x[1], reverse=True)

        hits = [Hit(self.chunks[i], s, kw_map.get(i, 0.0), vec_map.get(i, 0.0))
                for i, s in ranked[:pool]]
        if mode == "hybrid" and rerank:
            hits = self.reranker.rerank(query, hits)
        return hits[:top_k]
