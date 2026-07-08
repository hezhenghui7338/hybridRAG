"""Run the FastAPI server via:  python -m rag.api.server"""

from __future__ import annotations

import argparse

import uvicorn

from rag.api.server import app


def main() -> None:
    p = argparse.ArgumentParser(description="Run the Hybrid RAG HTTP server.")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--reload", action="store_true")
    args = p.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
