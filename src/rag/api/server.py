"""FastAPI server exposing the RAG pipeline over HTTP.

Run:
    uv run python -m rag.api.server
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from rag.pipeline import RAGPipeline

# Lazy global pipeline to avoid model load on import.
_pipeline: RAGPipeline | None = None


def get_pipeline() -> RAGPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline()
    return _pipeline


app = FastAPI(
    title="Hybrid RAG",
    description="BGE-M3 dense + BM25 sparse + RRF + reranker + Ollama",
    version="0.1.0",
)


class IngestRequest(BaseModel):
    data_dir: str | None = Field(default=None, description="Directory to ingest")
    files: list[str] | None = Field(default=None, description="Specific files")
    reset: bool = False


class IngestResponse(BaseModel):
    added_chunks: int
    dense_chunks: int
    sparse_chunks: int


class QueryRequest(BaseModel):
    question: str
    top_k: int | None = None
    rerank: bool = True
    stream: bool = False


class Hit(BaseModel):
    chunk_id: str
    text: str
    source: str
    score_rrf: float
    score_dense: float | None = None
    score_sparse: float | None = None
    score_rerank: float | None = None


class QueryResponse(BaseModel):
    question: str
    hits: list[Hit]
    answer: str | None = None


@app.get("/health")
def health() -> dict[str, Any]:
    pipe = get_pipeline()
    return {
        "status": "ok",
        "stats": pipe.stats,
        "ollama_available": pipe.generator.is_available(),
    }


@app.post("/ingest", response_model=IngestResponse)
def ingest(req: IngestRequest) -> IngestResponse:
    pipe = get_pipeline()
    if req.data_dir:
        n = pipe.ingest_directory(req.data_dir, reset=req.reset)
    elif req.files:
        n = pipe.ingest_files(req.files, reset=req.reset)
    else:
        raise HTTPException(400, "either data_dir or files required")
    stats = pipe.stats
    return IngestResponse(
        added_chunks=n, dense_chunks=stats["dense_chunks"], sparse_chunks=stats["sparse_chunks"]
    )


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    pipe = get_pipeline()
    if not req.rerank:
        pipe.reranker.model = None
    try:
        res = pipe.ask(req.question, top_k=req.top_k)
    except Exception as e:
        raise HTTPException(500, f"query failed: {e}") from e
    finally:
        # Restore reranker for subsequent calls
        pipe._ensure_ready()
    return QueryResponse(
        question=res.question,
        hits=[Hit(**asdict(h)) for h in res.hits],
        answer=res.answer,
    )


@app.post("/retrieve")
def retrieve(req: QueryRequest) -> dict[str, Any]:
    """Retrieve-only endpoint (no LLM generation)."""
    pipe = get_pipeline()
    if not req.rerank:
        pipe.reranker.model = None
    try:
        hits = pipe.retrieve(req.question, top_k=req.top_k)
    finally:
        pipe._ensure_ready()
    return {
        "question": req.question,
        "hits": [asdict(h) for h in hits],
    }
