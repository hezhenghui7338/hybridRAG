"""Configuration loader.

Reads `config.yaml` next to the project root, then layers environment variables on top.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

# Load .env if present (silently skip otherwise)
load_dotenv()


def _env(name: str, default: str | None = None) -> str | None:
    v = os.getenv(name)
    return v if v not in (None, "") else default


def _resolve(path_str: str, base: Path) -> Path:
    p = Path(os.path.expanduser(path_str))
    if not p.is_absolute():
        p = (base / p).resolve()
    return p


@dataclass
class EmbeddingConfig:
    model_name: str
    reranker_name: str
    device: str
    max_length: int
    use_fp16: bool


@dataclass
class StorageConfig:
    storage_dir: Path
    collection_name: str


@dataclass
class ChunkingConfig:
    chunk_size: int
    chunk_overlap: int
    keep_separator: bool


@dataclass
class RetrieveConfig:
    top_k_dense: int
    top_k_sparse: int
    top_k_final: int
    rrf_k: int
    candidate_pool: int


@dataclass
class RerankConfig:
    enabled: bool
    top_n: int


@dataclass
class GenerateConfig:
    host: str
    model: str
    temperature: float
    num_ctx: int
    timeout: int
    system_prompt: str


@dataclass
class AppConfig:
    embedding: EmbeddingConfig
    storage: StorageConfig
    chunking: ChunkingConfig
    retrieve: RetrieveConfig
    rerank: RerankConfig
    generate: GenerateConfig

    @property
    def chroma_path(self) -> Path:
        return self.storage.storage_dir / "chroma"

    @property
    def bm25_path(self) -> Path:
        return self.storage.storage_dir / "bm25.pkl"


def load_config(config_path: str | Path | None = None) -> AppConfig:
    """Load config from YAML + env overrides."""
    project_root = Path(__file__).resolve().parents[2]
    cfg_path = Path(config_path) if config_path else project_root / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"config.yaml not found at {cfg_path}")

    # If HF_HOME is set (via .env), propagate to huggingface_hub cache.
    hf_home = os.getenv("HF_HOME")
    if hf_home:
        hp = Path(os.path.expanduser(hf_home))
        if not hp.is_absolute():
            hp = (project_root / hp).resolve()
        os.environ["HF_HOME"] = str(hp)
        # Also set the older variable used by some libs
        os.environ["HUGGINGFACE_HUB_CACHE"] = str(hp)

    raw: dict[str, Any] = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    # ---- embedding ----
    emb_raw = raw.get("embedding", {})
    embedding = EmbeddingConfig(
        model_name=_env("BGE_M3_MODEL_NAME", emb_raw.get("model_name")) or "BAAI/bge-m3",
        reranker_name=_env("BGE_RERANKER_MODEL_NAME", emb_raw.get("reranker_name"))
        or "BAAI/bge-reranker-v2-m3",
        device=_env("EMBED_DEVICE", emb_raw.get("device", "cpu")) or "cpu",
        max_length=int(emb_raw.get("max_length", 8192)),
        use_fp16=bool(emb_raw.get("use_fp16", True)),
    )

    # ---- storage ----
    st_raw = raw.get("storage", {})
    storage = StorageConfig(
        storage_dir=_resolve(
            _env("STORAGE_DIR", st_raw.get("storage_dir", "./storage")) or "./storage",
            project_root,
        ),
        collection_name=_env("COLLECTION_NAME", st_raw.get("collection_name")) or "hybrid_rag",
    )
    storage.storage_dir.mkdir(parents=True, exist_ok=True)

    # ---- chunking ----
    ch_raw = raw.get("chunking", {})
    chunking = ChunkingConfig(
        chunk_size=int(ch_raw.get("chunk_size", 512)),
        chunk_overlap=int(ch_raw.get("chunk_overlap", 64)),
        keep_separator=bool(ch_raw.get("keep_separator", True)),
    )

    # ---- retrieve ----
    rt_raw = raw.get("retrieve", {})
    retrieve = RetrieveConfig(
        top_k_dense=int(rt_raw.get("top_k_dense", 20)),
        top_k_sparse=int(rt_raw.get("top_k_sparse", 20)),
        top_k_final=int(rt_raw.get("top_k_final", 5)),
        rrf_k=int(rt_raw.get("rrf_k", 60)),
        candidate_pool=int(rt_raw.get("candidate_pool", 50)),
    )

    # ---- rerank ----
    rr_raw = raw.get("rerank", {})
    rerank = RerankConfig(
        enabled=bool(rr_raw.get("enabled", True)),
        top_n=int(rr_raw.get("top_n", 30)),
    )

    # ---- generate ----
    g_raw = raw.get("generate", {})
    generate = GenerateConfig(
        host=_env("OLLAMA_HOST", g_raw.get("host")) or "http://127.0.0.1:11434",
        model=_env("OLLAMA_MODEL", g_raw.get("model")) or "qwen2.5:7b",
        temperature=float(g_raw.get("temperature", 0.2)),
        num_ctx=int(g_raw.get("num_ctx", 4096)),
        timeout=int(_env("OLLAMA_TIMEOUT", str(g_raw.get("timeout", 120))) or 120),
        system_prompt=str(g_raw.get("system_prompt", "")).strip(),
    )

    return AppConfig(
        embedding=embedding,
        storage=storage,
        chunking=chunking,
        retrieve=retrieve,
        rerank=rerank,
        generate=generate,
    )
