"""Answer generation with numbered source citations.

* LLMGenerator        - GPT model through the OpenAI API, grounded on retrieved
                        context only (used when OPENAI_API_KEY is set).
* ExtractiveGenerator - offline mode: selects the most relevant sentences from
                        the retrieved passages and cites them.
"""
import os

import requests

from .text import split_sentences, tokenize

SYSTEM_PROMPT = """You are the customer support assistant for Northline Home Services.
Answer ONLY from the numbered context passages. Cite every fact with its number like [1].
If the answer is not in the context, say you could not find it and suggest calling the office.
Keep answers short, friendly and specific."""


def _context_block(hits) -> str:
    return "\n\n".join(
        f"[{n}] ({h.chunk.doc_title} > {h.chunk.section})\n{h.chunk.text}"
        for n, h in enumerate(hits, 1)
    )


class LLMGenerator:
    name = "openai"

    def __init__(self, model: str | None = None):
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.key = os.environ["OPENAI_API_KEY"]

    def answer(self, question: str, hits, idf: dict) -> str:
        resp = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {self.key}"},
            json={
                "model": self.model,
                "temperature": 0,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Context:\n{_context_block(hits)}\n\nQuestion: {question}"},
                ],
            },
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()


class ExtractiveGenerator:
    name = "extractive"

    def __init__(self, max_sentences: int = 3, use_top: int = 3):
        self.max_sentences = max_sentences
        self.use_top = use_top

    def answer(self, question: str, hits, idf: dict) -> str:
        q_terms = set(tokenize(question))
        candidates = []
        for n, hit in enumerate(hits[: self.use_top], 1):
            for pos, sent in enumerate(split_sentences(hit.chunk.text)):
                terms = set(tokenize(sent))
                overlap = sum(idf.get(t, 1.0) for t in q_terms & terms)
                section_bonus = 0.5 * len(q_terms & set(tokenize(hit.chunk.section)))
                score = (overlap + section_bonus) / (1 + 0.6 * (n - 1)) - 0.02 * pos
                candidates.append((score, n, pos, sent))

        ranked = sorted(candidates, reverse=True)
        top_score = ranked[0][0] if ranked else 0
        best = [c for c in ranked[: self.max_sentences] if c[0] >= 0.6 * top_score] or ranked[:1]
        best.sort(key=lambda c: (c[1], c[2]))
        parts = []
        for _, n, _, sent in best:
            sent = sent if sent.endswith((".", "!", "?")) else sent + "."
            parts.append(f"{sent} [{n}]")
        return " ".join(parts)


def get_generator():
    return LLMGenerator() if os.getenv("OPENAI_API_KEY") else ExtractiveGenerator()
