"""Document loaders: PDF / Markdown / TXT.

Each loader yields ``(text, metadata)`` pairs. The text is the raw content; metadata
includes the source path so we can trace citations back.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import frontmatter
import pymupdf  # PyMuPDF

SUPPORTED_SUFFIXES = {".pdf", ".md", ".markdown", ".txt"}


@dataclass
class LoadedDoc:
    text: str
    source: str  # absolute path
    doc_id: str  # stable id derived from path
    metadata: dict = field(default_factory=dict)


def _doc_id_for(path: Path) -> str:
    # Stable across runs: path relative to a stable root, plus stem+hash.
    return f"{path.stem}::{abs(path)}"


def abs(path: Path) -> str:
    return str(path.resolve())


def _load_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _load_markdown(path: Path) -> tuple[str, dict]:
    post = frontmatter.load(path)
    md_text = post.content
    # Optional: strip excessive whitespace but preserve paragraph breaks.
    md_text = "\n".join(line.rstrip() for line in md_text.splitlines())
    meta = dict(post.metadata) if post.metadata else {}
    return md_text, meta


def _load_pdf(path: Path) -> tuple[str, dict]:
    """Load PDF using PyMuPDF. We extract per-page text and join with double newlines
    so the chunker can use paragraph boundaries. ``page_map`` records char->page so
    chunks can cite the page they came from.
    """
    doc = pymupdf.open(path)
    parts: list[str] = []
    page_breaks: list[int] = []  # cumulative length after each page
    total = 0
    for page in doc:
        text = page.get_text("text") or ""
        text = text.strip()
        if text:
            parts.append(text)
            total += len(text) + 2  # account for "\n\n" joiner
            page_breaks.append(total)
    doc.close()
    return "\n\n".join(parts), {"page_breaks": page_breaks, "n_pages": len(page_breaks)}


def load_file(path: Path) -> LoadedDoc | None:
    """Load a single file. Returns None if format unsupported or content is empty."""
    path = Path(path)
    if not path.exists() or not path.is_file():
        return None
    suf = path.suffix.lower()
    if suf not in SUPPORTED_SUFFIXES:
        return None

    try:
        if suf in {".md", ".markdown"}:
            text, meta = _load_markdown(path)
        elif suf == ".pdf":
            text, meta = _load_pdf(path)
        else:  # .txt
            text, meta = _load_txt(path), {}
    except Exception as e:
        print(f"[loader] failed to load {path}: {e}")
        return None

    text = text.strip()
    if not text:
        return None

    meta.setdefault("filename", path.name)
    meta.setdefault("suffix", suf)
    return LoadedDoc(text=text, source=abs(path), doc_id=_doc_id_for(path), metadata=meta)


def load_directory(root: Path, recursive: bool = True) -> Iterator[LoadedDoc]:
    """Yield LoadedDoc for every supported file under ``root``."""
    root = Path(root)
    if not root.exists():
        return
    pattern = "**/*" if recursive else "*"
    for p in sorted(root.glob(pattern)):
        if not p.is_file():
            continue
        if p.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        doc = load_file(p)
        if doc is not None:
            yield doc
