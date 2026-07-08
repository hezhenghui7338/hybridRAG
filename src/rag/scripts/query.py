"""CLI: ask a question against the hybrid RAG index.

Usage:
    uv run rag-query "你的问题"
    uv run rag-query "..." --top-k 10 --no-rerank
    uv run rag-query "..." --no-generate       # just retrieve
    uv run rag-query "..." --stream            # stream LLM output
"""

from __future__ import annotations

import argparse

from rag.config import load_config
from rag.pipeline import RAGPipeline


def main() -> int:
    p = argparse.ArgumentParser(description="Ask a question to the hybrid RAG system.")
    p.add_argument("question", type=str, help="Question to ask")
    p.add_argument("--top-k", type=int, default=None, help="Final top-K (overrides config)")
    p.add_argument(
        "--no-generate", action="store_true", help="Skip LLM, just print retrieved chunks"
    )
    p.add_argument("--no-rerank", action="store_true", help="Disable reranker for this query")
    p.add_argument("--stream", action="store_true", help="Stream the LLM response")
    p.add_argument("--config", type=str, default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    pipe = RAGPipeline(cfg)
    print(f"[index] dense={pipe.chroma.count()} sparse={pipe.bm25.count()}")

    if args.no_rerank:
        pipe.reranker.model = None

    if args.no_generate:
        hits = pipe.retrieve(args.question, top_k=args.top_k)
        if not hits:
            print("(no hits)")
            return 0
        for i, h in enumerate(hits, 1):
            print(
                f"\n--- hit #{i}  chunk={h.chunk_id}  rrf={h.score_rrf:.4f}  rerank={h.score_rerank} ---"
            )
            print(f"source: {h.source}")
            print(h.text[:600] + ("…" if len(h.text) > 600 else ""))
        return 0

    if args.stream:
        print(f"\nQ: {args.question}\nA: ", end="", flush=True)
        for piece in pipe.ask_stream(args.question, top_k=args.top_k):
            print(piece, end="", flush=True)
        print()
        return 0

    res = pipe.ask(args.question, top_k=args.top_k)
    print(f"\n=== Hits ({len(res.hits)}) ===")
    for i, h in enumerate(res.hits, 1):
        print(f"[{i}] rrf={h.score_rrf:.4f} rerank={h.score_rerank} source={h.source}")
    print(f"\n=== Answer ===\n{res.answer}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
