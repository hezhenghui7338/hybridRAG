"""CLI: build the index from a directory of documents.

Usage:
    uv run rag-ingest --data ./data/raw
    uv run rag-ingest --data ./data/raw --reset
    uv run rag-ingest --file doc.pdf doc2.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rag.config import load_config
from rag.pipeline import RAGPipeline


def main() -> int:
    p = argparse.ArgumentParser(description="Ingest documents into the hybrid RAG index.")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--data", type=str, help="Directory of documents to ingest recursively.")
    g.add_argument("--file", nargs="+", help="Specific file(s) to ingest.")
    p.add_argument("--reset", action="store_true", help="Reset existing indexes before ingest.")
    p.add_argument("--config", type=str, default=None, help="Path to config.yaml")
    args = p.parse_args()

    cfg = load_config(args.config)
    pipe = RAGPipeline(cfg)

    if args.reset:
        print("[ingest] resetting existing indexes …")
    if args.data:
        root = Path(args.data).expanduser().resolve()
        if not root.exists():
            print(f"[ingest] data dir not found: {root}", file=sys.stderr)
            return 2
        n = pipe.ingest_directory(root, reset=args.reset)
    else:
        n = pipe.ingest_files(args.file, reset=args.reset)

    stats = pipe.stats
    print(f"[ingest] added {n} chunks. totals: {stats}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
