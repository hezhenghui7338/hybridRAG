"""Hybrid retriever: dense (Chroma) + sparse (BM25) with Reciprocal Rank Fusion.

Why RRF?
- Dense scores (cosine similarity in [0, 1]) and BM25 scores (raw term-frequency sums,
  unbounded range) are not comparable. Trying to z-score normalize and weight-sum them
  is fragile across corpora.
- RRF only uses ranks, so it's robust to score scale differences. Empirically it's the
  most reliable off-the-shelf fusion baseline. (Linear combination with calibration is
  better when you can spend the time to tune it.)
- Formula: RRF(d) = Σ_r  1 / (k + rank_r(d))   for each ranker r that returned d.

Note: a doc that appears in only one list still gets a contribution from that list,
so hybrid "fills in" gaps from either side.
"""

from __future__ import annotations

from typing import Any

from rag.embed.bge_m3 import BGEM3Embedder
from rag.store.bm25_store import BM25Store
from rag.store.chroma_store import ChromaStore


def reciprocal_rank_fusion(
    ranked_lists: list[list[dict[str, Any]]],
    rrf_k: int = 60,
    top_k: int | None = None,
) -> list[dict[str, Any]]:
    """Fuse multiple ranked lists (each a list of hit-dicts with 'chunk_id') via RRF.

    Returns a single fused list, sorted descending by RRF score. Each returned item
    carries the merged metadata/text plus per-source score fields (e.g.
    ``score_dense``, ``score_sparse``).
    """
    by_id: dict[str, dict[str, Any]] = {}
    for lst in ranked_lists:
        for rank, hit in enumerate(lst, start=1):
            cid = hit["chunk_id"]
            entry = by_id.setdefault(
                cid,
                {
                    "chunk_id": cid,
                    "text": hit.get("text", ""),
                    "metadata": hit.get("metadata", {}),
                    "score_rrf": 0.0,
                    "ranks": [],
                },
            )
            entry["score_rrf"] += 1.0 / (rrf_k + rank)
            entry["ranks"].append(rank)
            # Carry over any score_* field from this ranker
            for k, v in hit.items():
                if k.startswith("score_") and k not in entry:
                    entry[k] = v

    fused = sorted(by_id.values(), key=lambda x: x["score_rrf"], reverse=True)
    if top_k is not None:
        fused = fused[:top_k]
    return fused


class HybridRetriever:
    def __init__(
        self,
        embedder: BGEM3Embedder,
        chroma: ChromaStore,
        bm25: BM25Store,
        top_k_dense: int = 20,
        top_k_sparse: int = 20,
        candidate_pool: int = 50,
        rrf_k: int = 60,
    ) -> None:
        self.embedder = embedder
        self.chroma = chroma
        self.bm25 = bm25
        self.top_k_dense = top_k_dense
        self.top_k_sparse = top_k_sparse
        self.candidate_pool = candidate_pool
        self.rrf_k = rrf_k

    def retrieve(self, query: str) -> list[dict[str, Any]]:
        # 1) dense
        q_emb = self.embedder.embed_query(query).tolist()
        dense_hits = self.chroma.query(q_emb, top_k=self.top_k_dense)

        # 2) sparse
        sparse_hits = self.bm25.query(query, top_k=self.top_k_sparse)

        # 3) RRF
        fused = reciprocal_rank_fusion(
            [dense_hits, sparse_hits],
            rrf_k=self.rrf_k,
            top_k=self.candidate_pool,
        )
        return fused
