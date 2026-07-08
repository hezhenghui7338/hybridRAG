"""Recursive character chunker with Chinese-friendly separators.

Strategy: prefer splitting on the largest semantic boundary that keeps the chunk
under ``chunk_size``. Boundaries (in order of preference):

    1. Paragraph breaks (``\n\n``)
    2. Sentence-ending punctuation (。！？!?;；\n)
    3. Other whitespace
    4. Hard character split (fallback)

We merge small pieces upward so we don't end up with tiny chunks when feasible.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

# Highest → lowest priority separators (first one tried keeps chunks most coherent).
SEPARATORS: list[str] = [
    "\n\n",
    "\n",
    "。",
    "！",
    "？",  # CJK sentence end
    ";",
    "；",  # semicolons
    ".",
    "?",
    "!",  # ASCII sentence end
    " ",  # space
    "",  # character-level fallback
]


@dataclass
class Chunk:
    text: str
    doc_id: str
    source: str
    chunk_id: str  # global unique id: {doc_id}#{i:04d}
    index: int  # position within document
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "doc_id": self.doc_id,
            "source": self.source,
            "chunk_id": self.chunk_id,
            "index": self.index,
            "metadata": self.metadata,
        }


def _split(text: str, sep: str) -> list[str]:
    """Split ``text`` by ``sep``. Empty pieces are dropped."""
    if sep == "":
        return list(text)
    parts = text.split(sep)
    return [p for p in parts if p != ""]


def _recursive_split(text: str, separators: list[str], chunk_size: int) -> list[str]:
    """Recursively split ``text`` using ``separators`` until every piece <= chunk_size."""
    if len(text) <= chunk_size:
        return [text]

    # Pick the first separator that actually appears in text.
    sep: str | None = None
    remaining: list[str] = []
    for cand in separators:
        if cand and cand in text:
            sep = cand
            remaining = [s for s in separators if s != cand]
            break
    if sep is None:
        # No separator found; we'll do a hard character-level split below.
        return _hard_split(text, chunk_size)

    pieces = _split(text, sep)
    pieces = [p.strip() for p in pieces]
    pieces = [p for p in pieces if p]

    merged: list[str] = []
    buf = ""
    sep_for_join = sep if sep not in ("", " ") else "\n"
    for p in pieces:
        candidate = (buf + sep_for_join + p) if buf else p
        if len(candidate) <= chunk_size:
            buf = candidate
        else:
            if buf:
                merged.append(buf)
            if len(p) > chunk_size:
                # Recursively split oversized piece.
                merged.extend(_recursive_split(p, remaining or [""], chunk_size))
                buf = ""
            else:
                buf = p
    if buf:
        merged.append(buf)
    return merged


def _hard_split(text: str, chunk_size: int) -> list[str]:
    return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]


def _add_overlap(chunks: list[str], overlap: int) -> list[str]:
    """Add ``overlap`` chars from the previous chunk to the start of each chunk."""
    if overlap <= 0 or len(chunks) <= 1:
        return chunks
    out = [chunks[0]]
    for i in range(1, len(chunks)):
        prev = chunks[i - 1]
        prefix = prev[-overlap:] if len(prev) > overlap else prev
        out.append(prefix + chunks[i])
    return out


def chunk_text(
    text: str,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    separators: list[str] | None = None,
) -> list[str]:
    seps = separators or SEPARATORS
    chunks = _recursive_split(text, seps, chunk_size)
    chunks = _add_overlap(chunks, chunk_overlap)
    return [c for c in chunks if c.strip()]


def chunk_documents(docs: Iterable) -> list[Chunk]:
    """Chunk every LoadedDoc. Returns flat list of Chunk with stable ids."""
    chunks: list[Chunk] = []
    for doc in docs:
        pieces = chunk_text(doc.text)
        for i, piece in enumerate(pieces):
            chunks.append(
                Chunk(
                    text=piece,
                    doc_id=doc.doc_id,
                    source=doc.source,
                    chunk_id=f"{doc.doc_id}#{i:04d}",
                    index=i,
                    metadata={**doc.metadata, "doc_index": i},
                )
            )
    return chunks
