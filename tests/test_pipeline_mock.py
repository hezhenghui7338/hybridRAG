"""Mock-based pipeline test that exercises the full retrieval path without needing
the (large) BGE-M3 / reranker models.

Useful for:
- CI / sanity checks
- Quickly validating indexing and hybrid retrieval logic
"""

from __future__ import annotations

import numpy as np
import pytest

from rag.config import (
    AppConfig,
    ChunkingConfig,
    EmbeddingConfig,
    GenerateConfig,
    RerankConfig,
    RetrieveConfig,
    StorageConfig,
)
from rag.pipeline import RAGPipeline


@pytest.fixture
def mock_pipeline(tmp_path) -> RAGPipeline:
    """Build a pipeline where the embedder and reranker are replaced by cheap mocks."""
    cfg = AppConfig(
        embedding=EmbeddingConfig(
            model_name="mock",
            reranker_name="mock",
            device="cpu",
            max_length=512,
            use_fp16=False,
        ),
        storage=StorageConfig(
            storage_dir=tmp_path / "storage",
            collection_name="test_hybrid_rag",
        ),
        chunking=ChunkingConfig(chunk_size=200, chunk_overlap=20, keep_separator=True),
        retrieve=RetrieveConfig(
            top_k_dense=5,
            top_k_sparse=5,
            top_k_final=3,
            rrf_k=60,
            candidate_pool=10,
        ),
        rerank=RerankConfig(enabled=True, top_n=10),
        generate=GenerateConfig(
            host="http://127.0.0.1:1",  # never reachable on purpose
            model="mock-model",
            temperature=0.2,
            num_ctx=2048,
            timeout=5,
            system_prompt="mock",
        ),
    )
    pipe = RAGPipeline(cfg)

    # ---- mock embedder ----
    class FakeEmbedder:
        dim = 32

        def embed(self, texts: list[str]) -> EmbeddingResult:
            # Deterministic fake embeddings: hash-based, then L2-normalize.
            from rag.embed.bge_m3 import EmbeddingResult

            vecs = np.zeros((len(texts), self.dim), dtype=np.float32)
            for i, t in enumerate(texts):
                # word overlap similarity is approximated via shared tokens
                tokens = set(t.lower().split())
                for j in range(self.dim):
                    vecs[i, j] = (j + 1) * (1.0 if j % 2 == 0 else 0.5)
                # tweak: longer text, slightly larger norm; tokens affecting nothing
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            vecs = vecs / norms
            return EmbeddingResult(dense=vecs, dense_dim=self.dim)

        def embed_query(self, q: str):
            return self.embed([q]).dense[0]

    # ---- mock reranker ----
    class FakeReranker:
        def rerank(self, query, docs, top_n=None):
            # Boost docs that share any token with the query (very rough).
            q_tokens = set(query.lower().split())
            scored = []
            for i, d in enumerate(docs):
                d_tokens = set(d.lower().split())
                overlap = len(q_tokens & d_tokens)
                # add a small index tiebreaker so output is deterministic
                scored.append((i, overlap + 1e-6 * (len(docs) - i)))
            scored.sort(key=lambda x: x[1], reverse=True)
            if top_n is not None:
                scored = scored[:top_n]
            return scored

    pipe._embedder = FakeEmbedder()  # type: ignore[assignment]
    pipe.retriever.embedder = pipe._embedder  # type: ignore[assignment]
    pipe._reranker_model = FakeReranker()  # type: ignore[assignment]
    pipe.reranker.model = pipe._reranker_model  # type: ignore[assignment]
    return pipe


def test_ingest_then_retrieve_end_to_end(mock_pipeline: RAGPipeline, tmp_path):
    # Create a few docs
    docs_dir = tmp_path / "raw"
    docs_dir.mkdir()
    (docs_dir / "a.md").write_text(
        "# Alpha\n\nRAG combines retrieval and generation. Hybrid search mixes dense and sparse.",
        encoding="utf-8",
    )
    (docs_dir / "b.md").write_text(
        "# Beta\n\nBM25 is a classic sparse retrieval algorithm. Reranker re-scores top candidates.",
        encoding="utf-8",
    )
    (docs_dir / "c.md").write_text(
        "# Gamma\n\nOllama runs LLMs locally. BGE-M3 is a multilingual embedding model.",
        encoding="utf-8",
    )

    n = mock_pipeline.ingest_directory(docs_dir, reset=True)
    assert n > 0
    stats = mock_pipeline.stats
    assert stats["dense_chunks"] == n
    assert stats["sparse_chunks"] == n

    # Retrieve
    hits = mock_pipeline.retrieve("What is hybrid search?")
    assert 1 <= len(hits) <= mock_pipeline.config.retrieve.top_k_final
    assert all(h.text for h in hits)

    # Hit IDs are unique
    ids = [h.chunk_id for h in hits]
    assert len(ids) == len(set(ids))
