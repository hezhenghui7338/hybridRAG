"""ChromaDB dense vector store.

Chroma is local-embedded, persistent, and exposes a small Python API. We use it as
a thin wrapper around a persistent collection.

Note: BGE-M3 dense vectors are 1024-dim and L2-normalized. Chroma's default cosine
similarity works for that.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import chromadb
from chromadb.api.models.Collection import Collection

from rag.ingest.chunker import Chunk


class ChromaStore:
    def __init__(self, persist_dir: Path, collection_name: str) -> None:
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(self.persist_dir))
        self.collection: Collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self.collection_name = collection_name

    # ---- write ----

    def upsert(
        self,
        chunks: list[Chunk],
        embeddings: list[list[float]],
    ) -> None:
        if not chunks:
            return
        ids = [c.chunk_id for c in chunks]
        docs = [c.text for c in chunks]
        metas = []
        for c in chunks:
            m = dict(c.metadata)
            m["doc_id"] = c.doc_id
            m["source"] = c.source
            m["index"] = c.index
            metas.append(m)
        # Chroma batches upserts; chunk to avoid giant calls
        BATCH = 256
        for i in range(0, len(chunks), BATCH):
            self.collection.upsert(
                ids=ids[i : i + BATCH],
                documents=docs[i : i + BATCH],
                embeddings=embeddings[i : i + BATCH],
                metadatas=metas[i : i + BATCH],
            )

    # ---- read ----

    def query(
        self,
        query_embedding: list[float],
        top_k: int,
        where: dict | None = None,
    ) -> list[dict[str, Any]]:
        if not query_embedding:
            return []
        res = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        hits = []
        ids = res.get("ids", [[]])[0]
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]
        for cid, doc, meta, dist in zip(ids, docs, metas, dists):
            # cosine distance -> similarity
            sim = 1.0 - float(dist) if dist is not None else 0.0
            hits.append(
                {
                    "chunk_id": cid,
                    "text": doc,
                    "metadata": meta,
                    "score_dense": sim,
                }
            )
        return hits

    def count(self) -> int:
        return self.collection.count()

    def reset(self) -> None:
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
