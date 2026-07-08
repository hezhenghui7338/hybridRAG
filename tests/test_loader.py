"""Tests for the document loader."""

import textwrap
from pathlib import Path

from rag.ingest.loader import load_directory, load_file


def test_load_txt(tmp_path: Path):
    p = tmp_path / "a.txt"
    p.write_text("hello world", encoding="utf-8")
    doc = load_file(p)
    assert doc is not None
    assert doc.text == "hello world"
    assert doc.metadata["suffix"] == ".txt"


def test_load_markdown_with_frontmatter(tmp_path: Path):
    p = tmp_path / "a.md"
    p.write_text(
        textwrap.dedent(
            """\
            ---
            title: 测试
            tags: [a, b]
            ---

            # 标题

            正文内容。
            """
        ),
        encoding="utf-8",
    )
    doc = load_file(p)
    assert doc is not None
    assert doc.metadata.get("title") == "测试"
    assert "# 标题" in doc.text


def test_unsupported_suffix_returns_none(tmp_path: Path):
    p = tmp_path / "a.png"
    p.write_bytes(b"\x89PNG\r\n")
    assert load_file(p) is None


def test_load_directory_recursive(tmp_path: Path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "a.txt").write_text("a", encoding="utf-8")
    (tmp_path / "sub" / "b.md").write_text("# b", encoding="utf-8")
    docs = list(load_directory(tmp_path))
    assert len(docs) == 2
    assert {d.metadata["suffix"] for d in docs} == {".txt", ".md"}
