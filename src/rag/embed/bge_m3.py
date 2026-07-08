"""BGE-M3 embedder wrapper.

Why BGE-M3?
- One model produces dense + sparse (lexical) + multi-vec (colbert) embeddings.
- Multilingual (strong Chinese + English + code).
- Long context (8192 tokens).
- Open weights, runs locally.

For Hybrid Search we use:
- dense vectors (ChromaDB)
- lexical weights (we feed these to a separate BM25 index; alternatively you can
  fuse the lexical weights directly with BM25 — both are valid and the BM25 route
  is what we wire up here).

The Reranker is the cross-encoder `BAAI/bge-reranker-v2-m3`, same family.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class EmbeddingResult:
    """Holds outputs from BGE-M3 for one batch."""

    dense: np.ndarray  # (n, dim)
    dense_dim: int


class BGEM3Embedder:
    """Lazy-loading wrapper around FlagEmbedding BGEM3."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        device: str = "cpu",
        max_length: int = 8192,
        use_fp16: bool = True,
    ) -> None:
        self.model_name = model_name
        self.device = self._resolve_device(device)
        self.max_length = max_length
        self.use_fp16 = use_fp16 and self.device != "cpu"
        self._model = None

    @staticmethod
    def _resolve_device(d: str) -> str:
        d = d.lower()
        if d == "mps" and torch.backends.mps.is_available():
            return "mps"
        if d == "cuda" and torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def _load(self) -> None:
        if self._model is not None:
            return
        # Imported lazily so the rest of the package stays importable without torch.
        from FlagEmbedding import BGEM3FlagModel

        self._model = BGEM3FlagModel(
            self.model_name,
            use_fp16=self.use_fp16,
            device=self.device,
        )

    # ---- API ----

    def embed(self, texts: list[str]) -> EmbeddingResult:
        """Encode ``texts`` -> dense vectors."""
        if not texts:
            return EmbeddingResult(dense=np.zeros((0, 0), dtype=np.float32), dense_dim=0)

        self._load()
        # BGEM3FlagModel.encode returns dict with dense_vecs / lexical_weights / colbert_vecs
        out = self._model.encode(  # type: ignore[union-attr]
            texts,
            batch_size=12,
            max_length=self.max_length,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        dense = np.asarray(out["dense_vecs"], dtype=np.float32)
        # L2 normalize (BGE recommends cosine)
        norms = np.linalg.norm(dense, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        dense = dense / norms
        return EmbeddingResult(dense=dense, dense_dim=dense.shape[1])

    def embed_query(self, query: str) -> np.ndarray:
        return self.embed([query]).dense[0]


class BGEReranker:
    """Cross-encoder reranker. Higher score = more relevant."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        device: str = "cpu",
        use_fp16: bool = True,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.use_fp16 = use_fp16 and device != "cpu"
        self._model = None

    def _load(self) -> None:
        if self._model is not None:
            return
        from FlagEmbedding import FlagReranker

        self._model = FlagReranker(self.model_name, use_fp16=self.use_fp16, device=self.device)

    def rerank(
        self, query: str, documents: list[str], top_n: int | None = None
    ) -> list[tuple[int, float]]:
        """Returns list of (index, score) sorted by descending score."""
        if not documents:
            return []
        self._load()
        scores = self._model.compute_score(  # type: ignore[union-attr]
            [(query, d) for d in documents],
            normalize=True,
        )
        # FlagReranker may return list or single float depending on input shape
        if isinstance(scores, (float, int)):
            scores = [float(scores)]
        pairs = list(enumerate(scores))
        pairs.sort(key=lambda x: x[1], reverse=True)
        if top_n is not None:
            pairs = pairs[:top_n]
        return pairs
