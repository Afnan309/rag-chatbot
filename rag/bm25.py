"""Okapi BM25 keyword index (the sparse half of hybrid search)."""
import math
from collections import Counter

from .text import tokenize


class BM25Index:
    def __init__(self, documents: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = [tokenize(d) for d in documents]
        self.doc_len = [len(d) for d in self.docs]
        self.avgdl = sum(self.doc_len) / max(len(self.docs), 1)
        self.tf = [Counter(d) for d in self.docs]
        df = Counter(term for d in self.docs for term in set(d))
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: str) -> list[float]:
        q_terms = tokenize(query)
        out = []
        for tf, dl in zip(self.tf, self.doc_len):
            s = 0.0
            for t in q_terms:
                if t not in tf:
                    continue
                f = tf[t]
                s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
            out.append(s)
        return out

    def search(self, query: str, top_k: int = 5) -> list[tuple[int, float]]:
        s = self.scores(query)
        ranked = sorted(range(len(s)), key=lambda i: s[i], reverse=True)
        return [(i, s[i]) for i in ranked[:top_k] if s[i] > 0]
