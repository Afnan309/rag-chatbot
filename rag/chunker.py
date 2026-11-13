"""Section-aware chunking for Markdown knowledge-base documents.

Each chunk keeps its document title and section heading so the retriever
and the citations always know where a passage came from.
"""
from dataclasses import dataclass, field
from pathlib import Path
import re


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    doc_title: str
    section: str
    text: str
    metadata: dict = field(default_factory=dict)

    @property
    def searchable_text(self) -> str:
        return f"{self.doc_title}. {self.section}. {self.text}"


def _split_long(text: str, max_words: int, overlap: int) -> list[str]:
    words = text.split()
    if len(words) <= max_words:
        return [text]
    pieces, start = [], 0
    while start < len(words):
        pieces.append(" ".join(words[start:start + max_words]))
        start += max_words - overlap
    return pieces


def chunk_markdown(path: Path, max_words: int = 120, overlap: int = 20) -> list[Chunk]:
    raw = path.read_text(encoding="utf-8")
    title_match = re.search(r"^#\s+(.+)$", raw, flags=re.M)
    doc_title = title_match.group(1).strip() if title_match else path.stem
    doc_id = path.stem

    sections = re.split(r"^##\s+", raw, flags=re.M)
    chunks: list[Chunk] = []

    intro = re.sub(r"^#\s+.+$", "", sections[0], flags=re.M).strip()
    blocks = [("Overview", intro)] if intro else []
    for sec in sections[1:]:
        heading, _, body = sec.partition("\n")
        blocks.append((heading.strip(), body.strip()))

    for s_idx, (heading, body) in enumerate(blocks):
        for p_idx, piece in enumerate(_split_long(body, max_words, overlap)):
            chunks.append(Chunk(
                chunk_id=f"{doc_id}#{s_idx}.{p_idx}",
                doc_id=doc_id,
                doc_title=doc_title,
                section=heading,
                text=piece,
                metadata={"source": path.name},
            ))
    return chunks


def load_corpus(folder: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(folder.glob("*.md")):
        chunks.extend(chunk_markdown(path))
    return chunks
