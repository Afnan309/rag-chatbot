"""Second-stage reranker for the fused candidate pool.

Scores each candidate passage against the full question using:
  * IDF-weighted query coverage of the passage body,
  * a heading match bonus (FAQ-style headings often restate the question),
  * the first-stage fused score as a prior.
Runs in microseconds, so it adds no noticeable latency.
"""
from .text import tokenize


class LexicalReranker:
    def __init__(self, idf: dict, w_cov: float = 1.0, w_head: float = 0.6, w_prior: float = 25.0):
        self.idf, self.w_cov, self.w_head, self.w_prior = idf, w_cov, w_head, w_prior

    def _weight(self, terms) -> float:
        return sum(self.idf.get(t, 3.0) for t in terms)

    def score(self, question: str, hit) -> float:
        q = set(tokenize(question))
        if not q:
            return hit.score
        total = self._weight(q)
        body = set(tokenize(hit.chunk.text))
        head = set(tokenize(f"{hit.chunk.doc_title} {hit.chunk.section}"))
        coverage = self._weight(q & (body | head)) / total
        heading = self._weight(q & head) / total
        return self.w_cov * coverage + self.w_head * heading + self.w_prior * hit.score

    def rerank(self, question: str, hits):
        for h in hits:
            h.score = self.score(question, h)
        return sorted(hits, key=lambda h: h.score, reverse=True)
