"""Text utilities shared by the retrieval components."""
import re

STOPWORDS = set("""
a an the and or but if of to in on at for from by with about as is are was were be been being
do does did doing have has had having i me my we our you your it its this that these those
what which who whom when where why how can could should would will shall may might must
there their them they he she his her not no so than too very just also into out up down over
any all each some such only own same other more most both few
""".split())

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[.'][a-z0-9]+)*")


ACRONYMS = {
    "pto": "paid time off",
    "hvac": "heating cooling hvac",
    "ev": "electric vehicle ev",
    "co": "carbon monoxide",
}


def stem(word: str) -> str:
    """Small suffix stripper so 'cancel', 'cancelled' and 'cancellation' match."""
    for suffix in ("ations", "ation", "ings", "ing", "ies", "ers", "ed", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            word = word[: -3] + "y" if suffix == "ies" else word[: -len(suffix)]
            break
    if len(word) > 4 and word[-1] == word[-2] and word[-1] not in "aeiousz":
        word = word[:-1]
    return word


# Customer wording -> knowledge-base wording (applied to queries and documents alike)
PHRASES = {
    r"\bsign(ing)? up\b": "join",
    r"\bmoney back\b": "refund",
    r"\bon (the|their|his|her) way\b": "arrival",
    r"\binstall?ments?\b": "financing monthly",
    r"\bpay later\b": "financing",
    r"\bprice[sd]?\b": "cost",
    r"\bhow much\b": "cost",
    r"\bchange my\b": "replace my",
    r"\bfreezing\b": "cold temperature",
    r"\bboiling hot\b": "hot temperature",
    r"\bservice my\b": "tune up my",
    r"\bdown ?payment\b": "deposit",
    r"\bcome out\b": "arrival",
}


def normalize(text: str) -> str:
    text = text.lower()
    for pattern, repl in PHRASES.items():
        text = re.sub(pattern, repl, text)
    return text


def tokenize(text: str) -> list[str]:
    raw = _TOKEN_RE.findall(normalize(text).replace("-", " "))
    expanded = []
    for t in raw:
        expanded.extend(ACRONYMS.get(t, t).split())
    return [stem(t) for t in expanded if t not in STOPWORDS]


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+(?=- )|\n{2,}", text)
    return [p.strip(" -\n") for p in parts if len(p.strip(" -\n")) > 3]
