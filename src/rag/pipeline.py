"""End-to-end RAG pipeline.

Composes:  Loader -> Chunker -> Embedder -> ChromaStore + BM25Store
          -> HybridRetriever -> Reranker -> OllamaGenerator
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from tqdm import tqdm

from rag.config import AppConfig, load_config
from rag.embed.bge_m3 import BGEM3Embedder, BGEReranker
from rag.generate.ollama_client import OllamaGenerator
from rag.ingest.chunker import chunk_documents
from rag.ingest.loader import LoadedDoc, load_directory, load_file
from rag.retrieve.hybrid import HybridRetriever
from rag.retrieve.reranker import Reranker
from rag.store.bm25_store import BM25Store
from rag.store.chroma_store import ChromaStore


@dataclass
class RetrievalHit:
    chunk_id: str
    text: str
    source: str
    score_rrf: float
    score_dense: float | None = None
    score_sparse: float | None = None
    score_rerank: float | None = None
    metadata: dict = field(default_factory=dict)


@dataclass
class QueryResult:
    question: str
    hits: list[RetrievalHit]
    answer: str | None = None


class RAGPipeline:
    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = config or load_config()

        self.chroma = ChromaStore(
            persist_dir=self.config.chroma_path,
            collection_name=self.config.storage.collection_name,
        )
        self.bm25 = BM25Store(self.config.bm25_path)

        # Embedder / reranker are lazy-loaded so the constructor stays cheap
        # and we don't pay model-load cost when only inspecting stats.
        self._embedder: BGEM3Embedder | None = None
        self._reranker_model: BGEReranker | None = None

        self.retriever = HybridRetriever(
            embedder=None,  # wired in lazily by _ensure_embedder()
            chroma=self.chroma,
            bm25=self.bm25,
            top_k_dense=self.config.retrieve.top_k_dense,
            top_k_sparse=self.config.retrieve.top_k_sparse,
            candidate_pool=self.config.retrieve.candidate_pool,
            rrf_k=self.config.retrieve.rrf_k,
        )

        self.reranker = Reranker(
            model=None,  # loaded lazily
            top_n=self.config.rerank.top_n,
        )

        self.generator = OllamaGenerator(
            host=self.config.generate.host,
            model=self.config.generate.model,
            temperature=self.config.generate.temperature,
            num_ctx=self.config.generate.num_ctx,
            timeout=self.config.generate.timeout,
            system_prompt=self.config.generate.system_prompt,
        )

    # ---- lazy accessors ----

    def _get_embedder(self) -> BGEM3Embedder:
        if self._embedder is None:
            self._embedder = BGEM3Embedder(
                model_name=self.config.embedding.model_name,
                device=self.config.embedding.device,
                max_length=self.config.embedding.max_length,
                use_fp16=self.config.embedding.use_fp16,
            )
            # also wire into retriever (same instance)
            self.retriever.embedder = self._embedder
        return self._embedder

    def _get_reranker_model(self) -> BGEReranker:
        if self._reranker_model is None:
            self._reranker_model = BGEReranker(
                model_name=self.config.embedding.reranker_name,
                device=self.config.embedding.device,
                use_fp16=self.config.embedding.use_fp16,
            )
            self.reranker.model = self._reranker_model
        return self._reranker_model

    def _ensure_ready(self) -> None:
        """Load embedder + (optional) reranker before any retrieval work."""
        self._get_embedder()
        if self.config.rerank.enabled:
            self._get_reranker_model()

    # ---- ingest ----

    def ingest_directory(self, root: str | Path, reset: bool = False) -> int:
        root = Path(root)
        if reset:
            self.chroma.reset()
            self.bm25.reset()
        docs = list(load_directory(root))
        return self._ingest_docs(docs)

    def ingest_files(self, paths: list[str | Path], reset: bool = False) -> int:
        if reset:
            self.chroma.reset()
            self.bm25.reset()
        docs: list[LoadedDoc] = []
        for p in paths:
            d = load_file(Path(p))
            if d is not None:
                docs.append(d)
        return self._ingest_docs(docs)

    def _ingest_docs(self, docs: list[LoadedDoc]) -> int:
        if not docs:
            return 0
        # Make sure the (heavy) embedder is loaded before we encode anything.
        self._ensure_ready()

        # Chunk
        chunks = chunk_documents(docs)
        if not chunks:
            return 0

        # Embed (BGE-M3 dense)
        embedder = self._get_embedder()
        texts = [c.text for c in chunks]
        # batch by 32 to keep memory sane
        all_dense: list[list[float]] = []
        BS = 32
        for i in tqdm(range(0, len(texts), BS), desc="embedding", unit="batch"):
            batch = texts[i : i + BS]
            result = embedder.embed(batch)
            all_dense.extend(result.dense.tolist())

        # Dense store
        self.chroma.upsert(chunks, all_dense)

        # Sparse store (BM25 over the same chunks)
        # Rebuild from scratch (BM25 doesn't support incremental well).
        existing_ids = set(self.bm25.chunk_ids)
        new_ids = [c.chunk_id for c in chunks if c.chunk_id not in existing_ids]
        new_texts = [c.text for c in chunks if c.chunk_id not in existing_ids]
        new_metas = [
            {**c.metadata, "doc_id": c.doc_id, "source": c.source, "index": c.index}
            for c in chunks
            if c.chunk_id not in existing_ids
        ]
        merged_ids = self.bm25.chunk_ids + new_ids
        merged_texts = self.bm25.texts_raw + new_texts
        merged_metas = self.bm25.metas + new_metas
        self.bm25.build(merged_ids, merged_texts, merged_metas)

        return len(chunks)

    # ---- retrieve ----

    def retrieve(self, question: str, top_k: int | None = None) -> list[RetrievalHit]:
        top_k = top_k or self.config.retrieve.top_k_final
        self._ensure_ready()
        fused = self.retriever.retrieve(question)
        # Rerank if enabled
        reranked = self.reranker.rerank(question, fused)
        reranked = reranked[:top_k]
        hits: list[RetrievalHit] = []
        for h in reranked:
            meta = dict(h.get("metadata", {}))
            hits.append(
                RetrievalHit(
                    chunk_id=h["chunk_id"],
                    text=h.get("text", ""),
                    source=meta.get("source", ""),
                    score_rrf=float(h.get("score_rrf", 0.0)),
                    score_dense=h.get("score_dense"),
                    score_sparse=h.get("score_sparse"),
                    score_rerank=h.get("score_rerank"),
                    metadata=meta,
                )
            )
        return hits

    # ---- ask ----

    def ask(self, question: str, top_k: int | None = None, stream: bool = False) -> QueryResult:
        hits = self.retrieve(question, top_k=top_k)
        contexts = [h.text for h in hits]
        if stream:
            # For streaming, we still populate answer lazily — caller should use ask_stream.
            return QueryResult(question=question, hits=hits, answer=None)
        answer = self.generator.generate(question, contexts)
        return QueryResult(question=question, hits=hits, answer=answer)

    def ask_stream(self, question: str, top_k: int | None = None):
        hits = self.retrieve(question, top_k=top_k)
        contexts = [h.text for h in hits]
        for chunk in self.generator.stream(question, contexts):
            yield chunk

    # ---- inspect ----

    @property
    def stats(self) -> dict:
        return {
            "dense_chunks": self.chroma.count(),
            "sparse_chunks": self.bm25.count(),
        }
