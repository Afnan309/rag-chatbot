"""Input and output guardrails for the chatbot."""
import re

from .text import tokenize

INJECTION_PATTERNS = [
    r"ignore (all |any |the )?(previous|prior|above) (instructions|prompts?)",
    r"disregard (the |your )?(system|previous) (prompt|instructions)",
    r"reveal (your|the) (system prompt|instructions)",
    r"you are now (dan|in developer mode)",
    r"act as (an? )?(unrestricted|jailbroken)",
]

FALLBACK = ("I couldn't find that in our company documents, so I don't want to guess. "
            "Please call the office or book a visit online and our team will help you.")

INJECTION_REPLY = ("I can only answer questions about our services, pricing and policies "
                   "using the company knowledge base.")


def is_prompt_injection(message: str) -> bool:
    text = message.lower()
    return any(re.search(p, text) for p in INJECTION_PATTERNS)


def query_coverage(question: str, hit, idf: dict) -> float:
    """Share of the question's (IDF-weighted) meaning found in a passage."""
    q = set(tokenize(question))
    if not q:
        return 0.0
    found = q & set(tokenize(hit.chunk.searchable_text))
    total = sum(idf.get(t, 3.0) for t in q)   # unknown words weigh a lot
    return sum(idf.get(t, 3.0) for t in found) / total


def grounding_score(question: str, hits, idf: dict) -> float:
    """Keyword coverage + semantic similarity of the best evidence (0 to 2)."""
    if not hits:
        return 0.0
    best_cov = max(query_coverage(question, h, idf) for h in hits)
    best_vec = max(h.vector for h in hits)
    return best_cov + best_vec


def is_grounded(question: str, hits, idf: dict, threshold: float) -> bool:
    """Refuse to answer when retrieval evidence is weak (anti-hallucination).

    The threshold is tuned on the dev set (eval/dev_questions.json) and
    favours refusing over guessing.
    """
    return grounding_score(question, hits, idf) >= threshold
