"""Reranker wrapper.

Takes the candidate pool from the hybrid retriever and re-ranks using a
cross-encoder (BGE-reranker-v2-m3 by default). The cross-encoder jointly
encodes query+doc and outputs a single relevance score, which is much sharper
than dot-product similarity — at the cost of one forward pass per candidate.
"""

from __future__ import annotations

from typing import Any

from rag.embed.bge_m3 import BGEReranker


class Reranker:
    def __init__(self, model: BGEReranker | None = None, top_n: int = 30) -> None:
        self.model = model
        self.top_n = top_n

    def rerank(self, query: str, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not candidates:
            return []
        if self.model is None:
            # Rerank disabled. Just slice.
            return candidates[: self.top_n]

        docs = [c.get("text", "") for c in candidates]
        scored = self.model.rerank(query, docs, top_n=self.top_n)
        out = []
        for idx, score in scored:
            c = dict(candidates[idx])
            c["score_rerank"] = float(score)
            out.append(c)
        return out
