# Hybrid Search RAG

> 面向中文场景的本地化 RAG 流水线：稠密（BGE-M3）+ 稀疏（BM25）+ RRF 融合 + Cross-Encoder Rerank + Ollama 推理。
> 全部跑在本地，零 API 费用，零数据外传。

一个简洁、生产形态的 RAG 框架。`BGE-M3` 做语义向量，`BM25` 兜底关键词精确匹配，
两者通过 **Reciprocal Rank Fusion (RRF)** 融合，再用 `BGE-reranker-v2-m3` 二次精排，
最后交给本地 `Ollama` LLM 生成答案。

```
┌─────────────┐     ┌─────────┐     ┌────────────────┐
│  Documents  │ ──▶ │ Chunker │ ──▶ │  Embedder      │
│ PDF/MD/TXT  │     │ Recur.  │     │  BGE-M3        │
└─────────────┘     └─────────┘     └───────┬────────┘
                                            │
                             ┌──────────────┴──────────────┐
                             ▼                             ▼
                    ┌─────────────────┐         ┌─────────────────┐
                    │  ChromaDB       │         │  BM25 Index     │
                    │  dense vectors  │         │  sparse weights │
                    └────────┬────────┘         └────────┬────────┘
                             │      Hybrid Retrieve    │
                             └────────────┬────────────┘
                                          ▼ (RRF fusion, top-N)
                              ┌─────────────────────┐
                              │  Reranker           │
                              │  BGE-reranker-v2-m3 │
                              └──────────┬──────────┘
                                          ▼ (top-K)
                                ┌─────────────────────┐
                                │  Ollama LLM         │
                                │  (qwen / gemma …)   │
                                └──────────┬──────────┘
                                          ▼
                                      Answer
```

## Features

- **Hybrid Retrieval** — 稠密向量 + 经典 BM25，RRF 融合，长尾查询和精确关键词都不漏。
- **多语言 Embedding** — `BGE-M3` 单模型产出 dense / sparse / colbert 多模态向量，中英文都强；8192 tokens 长上下文。
- **Cross-Encoder Rerank** — `BGE-reranker-v2-m3` 在 RRF 候选上二次精排，把 top-1 准确率显著拉高。
- **本地 LLM** — 通过 [Ollama](https://ollama.com) 调用任意 chat 模型（qwen2.5 / qwen3 / gemma …），无 API 费用、可断网运行。
- **多格式文档** — 一键 ingest `PDF`（PyMuPDF，含分页定位）/ `Markdown`（含 frontmatter）/ `TXT`。
- **中文友好分块** — 递归分块器，保留段落分隔，配置 `chunk_size` / `chunk_overlap`。
- **两种使用形态** — `rag-query` CLI 直接问；`uvicorn` 起 FastAPI 服务，HTTP 端点开箱即用。
- **可配置可扩展** — 全部参数走 `config.yaml` + `.env`，Fusion / Rerank / LLM 都可以关掉或换实现。
- **持久化** — Chroma + BM25 索引落到 `storage/`，重启不重算；HF 模型缓存也归一管。
- **测试覆盖** — 13 个 pytest 用例覆盖分块 / BM25 / RRF / loader / mock e2e。

## Stack

| Layer        | Choice                                              | Why                                              |
| ------------ | --------------------------------------------------- | ------------------------------------------------ |
| Embedding    | `BAAI/bge-m3` (via FlagEmbedding)                   | Multilingual, dense+sparse+colbert in one model  |
| Vector store | ChromaDB (persistent, local)                        | Zero-config, embedded, easy to inspect           |
| Sparse store | `rank_bm25` (pickle on disk)                        | Classical BM25 Okapi, fast & deterministic       |
| Fusion       | Reciprocal Rank Fusion (RRF)                        | Score-free, robust, well-studied                 |
| Reranker     | `BAAI/bge-reranker-v2-m3`                           | Strong multilingual reranker                     |
| LLM          | Ollama local (any chat model, e.g. qwen / gemma)    | Zero API cost, full privacy                      |
| API          | FastAPI + Uvicorn                                   | Async-friendly, OpenAPI 自动生成                 |

## Quick start

```bash
# 1. install (Python 3.12 via uv)
uv sync --extra dev

# 2. drop documents into ./data/raw (pdf / md / txt)
cp -r my-docs/* data/raw/

# 3. start ollama and pull a model
ollama serve &
ollama pull qwen2.5:7b        # or any chat model

# 4. build the index (downloads BGE-M3 + reranker on first run, ~2.5GB)
uv run rag-ingest --data ./data/raw

# 5. ask
uv run rag-query "你的问题"

# 6. (optional) run as an HTTP service
uv run python -m rag.api.server --port 8000
```

> 首次 ingest 会下载 `BAAI/bge-m3` (~2.27 GB) 和 `BAAI/bge-reranker-v2-m3` (~568 MB) 到
> `./storage/models/`。之后可完全离线运行。

## HTTP API

启动后访问 `http://127.0.0.1:8000/docs` 看到 OpenAPI 文档。

| Method | Path       | 说明                                                       |
| ------ | ---------- | ---------------------------------------------------------- |
| GET    | `/health`  | 健康检查 + 索引规模 + Ollama 可用性                         |
| POST   | `/ingest`  | 触发 ingest；支持 `data_dir` 或 `files`，可选 `reset`       |
| POST   | `/query`   | 完整 RAG：retrieve + rerank + generate，返回 `hits` + `answer` |
| POST   | `/retrieve` | 仅检索（不调 LLM），调试重排 / 召回用                      |

`POST /query` request:

```json
{ "question": "什么是混合检索?", "top_k": 5, "rerank": true, "stream": false }
```

`POST /retrieve` request:

```json
{ "question": "混合检索", "top_k": 5, "rerank": true }
```

## Layout

```
src/rag/
├── config.py              # YAML + env config (dotenv, overrides)
├── pipeline.py            # end-to-end RAGPipeline (orchestrator)
├── ingest/
│   ├── loader.py          # PDF (PyMuPDF) / Markdown (frontmatter) / TXT
│   └── chunker.py         # recursive Chinese-aware splitter + overlap
├── embed/
│   └── bge_m3.py          # dense encoder + BGE cross-encoder reranker
├── store/
│   ├── chroma_store.py    # dense vector index (cosine)
│   └── bm25_store.py      # sparse BM25 index + pickle persistence
├── retrieve/
│   ├── hybrid.py          # RRF fusion of dense + sparse ranked lists
│   └── reranker.py        # cross-encoder rerank wrapper
├── generate/
│   └── ollama_client.py   # chat with local Ollama (think=False by default)
├── api/
│   └── server.py          # FastAPI HTTP service
└── scripts/
    ├── ingest.py          # CLI: build index
    └── query.py           # CLI: ask questions
tests/                     # pytest, 13 cases (unit + mock e2e)
data/raw/                  # drop source documents here
storage/                   # index + model cache (gitignored)
config.yaml                # tunable knobs (chunk_size, top_k, RRF k, ...)
.env                       # local env overrides (HF_HOME, OLLAMA_HOST, ...)
```

## Why Hybrid Search?

纯稠密检索会漏掉精确关键词（产品 SKU、错误码、人名）。纯 BM25 又抓不到语义改写。
Hybrid 把两边的强项都拿过来。**RRF** 融合两个 ranked list 不需要归一化分数——这是它
作为默认 fusion 策略的原因。

Rerank 在 hybrid 候选之上再做一次精排，把 top-K 再剃一刀。代价是对每对 (query, candidate)
多一次前向，但只跑在 20–50 个候选上，不是全语料。

## Tuning knobs (`config.yaml`)

```yaml
chunking:
  chunk_size: 512         # characters
  chunk_overlap: 64
retrieve:
  top_k_dense: 20         # how many from dense
  top_k_sparse: 20        # how many from BM25
  top_k_final: 5          # after rerank
  rrf_k: 60               # RRF constant
  candidate_pool: 50      # sent to reranker before final slice
rerank:
  enabled: true
  top_n: 30               # rerank this many, then slice top_k_final
generate:
  model: qwen3.5:4b
  temperature: 0.2
  num_ctx: 4096
  timeout: 300            # seconds; qwen3.5 needs longer for first call
```

## End-to-end behavior (verified)

```
$ uv run rag-query "什么是混合检索?它为什么有效?"

[index] dense=14 sparse=14
=== Hits (5) ===
[1] rrf=0.0328 rerank=0.9967 source=.../02-hybrid-search.md
[2] rrf=0.0315 rerank=0.4924 source=.../02-hybrid-search.md
[3] rrf=0.0149 rerank=0.0964 source=.../03-bge-m3.md
[4] rrf=0.0308 rerank=0.0578 source=.../03-bge-m3.md
[5] rrf=0.0308 rerank=0.0512 source=.../03-bge-m3.md

=== Answer ===
**什么是混合检索？**  混合检索（Hybrid Search）是指同时使用稠密检索
（Dense Retrieval）和稀疏检索（Sparse Retrieval）…… [1]
…
[1][2][4][5]
```

Reranker 把 top hit 拉到 0.997 — cross-encoder 把相关 chunk 和弱相关 chunk 拉开了
一个量级的距离。

## Performance (Apple M-series MPS)

- BGE-M3 first load: ~25s (2.27 GB)
- 5 docs → 14 chunks → full embed: ~85s
- Single retrieve + rerank (30 candidates): ~3s
- Ollama qwen3.5:4b generation: ~10–30s (depends on `think`)

## Gotchas (read this before deploying)

1. **transformers pin**: FlagEmbedding 1.2.x 与 `transformers 5.x` 不兼容（依赖已移除的
   `prepare_for_model`）。`pyproject.toml` 锁在 `<5.0`。
2. **hf_xet**: 新版 xet downloader 在某些代理下会卡住。如果模型下载卡死，卸掉它：
   `uv pip uninstall hf_xet`。
3. **HF cache dir**: 默认 `HF_HOME=./storage/models`（通过 `.env`），模型和索引放一起，
   备份 / 清理都方便。
4. **qwen3 / qwen3.5 thinking mode**: Ollama 在填充 `thinking` 时会返回空 `content`。
   `OllamaGenerator` 默认传 `think=False` 规避。
5. **Chroma cosine ↔ similarity**: Chroma 返回的是 *distance*，不是 similarity。
   已统一转成 `similarity = 1 - distance`。

## Tests

```bash
uv run pytest tests/ -v
# 13 passed
```

覆盖：tokenizer（CJK + ASCII）、recursive chunker、BM25 store、hybrid RRF 融合、
loader（PDF / MD / TXT / frontmatter）、以及一个 mock 的端到端流水线（不加载真实模型）。

## Extending

- **BGE-M3 lexical weights (sparse from the model)**: 跟 BM25 并联走 3-way RRF
  （BGE sparse + BM25 + dense）。在 `BGEM3Embedder` 里设 `return_sparse=True`。
- **ColBERT late-interaction**: 设 `return_colbert_vecs=True`，挂一个 ColBERT 索引
  （例如 `pylate`）。
- **Evaluation**: 拿 Ragas (`ragas.metrics.*`) 包住 retrieve + generate 即可。
- **API auth**: 在 `rag/api/server.py` 加一个 API-key dependency，对外暴露前必加。
- **Stream answer**: `pipeline.ask_stream(question)` 已经把 `Ollama.stream` 串起来了，
  HTTP 层加个 `text/event-stream` 路由就能用。
