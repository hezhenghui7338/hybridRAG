"""Tests for hybrid retrieval fusion."""

from rag.retrieve.hybrid import reciprocal_rank_fusion


def hit(cid, score=None):
    h = {"chunk_id": cid, "text": f"text-{cid}", "metadata": {}}
    if score is not None:
        h["score_dense"] = score
    return h


def test_rrf_basic():
    dense = [hit("a"), hit("b"), hit("c")]
    sparse = [hit("b"), hit("c"), hit("d")]
    fused = reciprocal_rank_fusion([dense, sparse], rrf_k=60)
    # 'b' and 'c' appear in both lists so they should outrank single-list items
    assert fused[0]["chunk_id"] in {"b", "c"}
    assert fused[1]["chunk_id"] in {"b", "c"}


def test_rrf_handles_disjoint():
    dense = [hit("a"), hit("b")]
    sparse = [hit("c"), hit("d")]
    fused = reciprocal_rank_fusion([dense, sparse], rrf_k=60)
    assert {h["chunk_id"] for h in fused} == {"a", "b", "c", "d"}


def test_rrf_top_k_limit():
    dense = [hit(f"x{i}") for i in range(10)]
    sparse = [hit(f"y{i}") for i in range(10)]
    fused = reciprocal_rank_fusion([dense, sparse], rrf_k=60, top_k=5)
    assert len(fused) == 5
