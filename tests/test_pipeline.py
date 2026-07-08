"""Tests for the hybrid RAG system.

Most integration tests require GPU/MPS and BGE-M3 model weights (~2GB). They are
gated behind a marker so unit tests stay cheap.
"""

from __future__ import annotations

from rag.ingest.chunker import SEPARATORS, chunk_text
from rag.store.bm25_store import tokenize


def test_tokenize_chinese():
    tokens = tokenize("我爱自然语言处理")
    # Should include unigrams and bigrams
    assert "我" in tokens
    assert "我爱" in tokens
    assert "爱自" in tokens


def test_tokenize_mixed():
    tokens = tokenize("Hello world, 你好世界!")
    assert "hello" in tokens
    assert "world" in tokens
    assert any("你好" in t for t in tokens)


def test_chunk_under_size():
    text = "第一段。\n\n第二段比较长一点,用来测试分块效果。\n\n第三段。"
    chunks = chunk_text(text, chunk_size=20, chunk_overlap=5)
    for c in chunks:
        # overlap may push it slightly over
        assert len(c) <= 30


def test_chunk_returns_multiple():
    text = "A。\n\nB。\n\nC。" * 50
    chunks = chunk_text(text, chunk_size=50, chunk_overlap=10)
    assert len(chunks) > 1
    # All non-empty
    assert all(c.strip() for c in chunks)


def test_recursive_split_basic():
    from rag.ingest.chunker import _recursive_split

    text = "Sentence one.\n\nSentence two is longer.\n\nSentence three."
    pieces = _recursive_split(text, SEPARATORS, chunk_size=30)
    # Each piece (without overlap) fits within chunk_size
    assert all(len(p) <= 30 for p in pieces)
    # Coverage: every non-separator char from input is preserved in some piece
    covered = "".join(pieces)
    for ch in text:
        if ch in "\n":
            continue
        assert ch in covered
