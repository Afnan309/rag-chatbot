"""End-to-end RAG pipeline: guardrails -> hybrid retrieval -> grounded answer."""
import time
from dataclasses import dataclass, field
from pathlib import Path

from .chunker import load_corpus
from .generator import get_generator
from .guardrails import FALLBACK, INJECTION_REPLY, is_grounded, is_prompt_injection
from .hybrid import HybridRetriever


@dataclass
class ChatResult:
    answer: str
    sources: list = field(default_factory=list)
    status: str = "answered"          # answered | fallback | blocked
    latency_ms: float = 0.0


class RAGPipeline:
    def __init__(self, docs_dir: Path, top_k: int = 4,
                 grounding_threshold: float = 1.05):
        self.chunks = load_corpus(docs_dir)
        self.retriever = HybridRetriever(self.chunks)
        self.generator = get_generator()
        self.top_k = top_k
        self.grounding_threshold = grounding_threshold

    @property
    def stats(self) -> dict:
        return {
            "documents": len({c.doc_id for c in self.chunks}),
            "chunks": len(self.chunks),
            "embedder": self.retriever.embedder.name,
            "generator": self.generator.name,
        }

    def ask(self, question: str, audience: str = "customer") -> ChatResult:
        start = time.perf_counter()

        if is_prompt_injection(question):
            return ChatResult(INJECTION_REPLY, status="blocked",
                              latency_ms=(time.perf_counter() - start) * 1000)

        hits = self.retriever.search(question, top_k=self.top_k + 3)
        if audience == "customer":   # role-based access: hide internal documents
            hits = [h for h in hits if "(Internal)" not in h.chunk.doc_title]
        hits = hits[: self.top_k]

        if not is_grounded(question, hits, self.retriever.bm25.idf, self.grounding_threshold):
            return ChatResult(FALLBACK, status="fallback",
                              latency_ms=(time.perf_counter() - start) * 1000)

        answer = self.generator.answer(question, hits, self.retriever.bm25.idf)
        sources = [{
            "n": n,
            "doc": h.chunk.doc_title,
            "section": h.chunk.section,
            "file": h.chunk.metadata["source"],
            "snippet": h.chunk.text[:220] + ("..." if len(h.chunk.text) > 220 else ""),
            "score": round(h.score, 4),
        } for n, h in enumerate(hits, 1)]
        return ChatResult(answer, sources, "answered",
                          latency_ms=(time.perf_counter() - start) * 1000)
