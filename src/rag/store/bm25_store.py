"""BM25 sparse store.

We persist the BM25 index plus a parallel list of chunk texts/metadatas via pickle.
For 10k-100k chunks this is fine. For larger corpora consider Lucene/Elasticsearch.
"""

from __future__ import annotations

import pickle
import re
from dataclasses import dataclass
from pathlib import Path

from rank_bm25 import BM25Okapi

# Lightweight tokenizer: split on non-word chars for latin, and per-character for CJK.
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_TOKEN_SPLIT_RE = re.compile(r"[^A-Za-z0-9\u4e00-\u9fff]+")


def tokenize(text: str) -> list[str]:
    """Mixed CJK + ASCII tokenizer.

    Strategy:
    - Split text into runs of CJK chars and runs of latin/digit chars.
    - For latin runs, lowercase and split on non-word chars.
    - For CJK runs, emit per-character bigrams (good recall for Chinese without
      requiring jieba). Optional: use jieba for word-level segmentation if you have
      it installed.
    """
    text = text.lower()
    tokens: list[str] = []
    for piece in _TOKEN_SPLIT_RE.split(text):
        if not piece:
            continue
        if _CJK_RE.search(piece) is None:
            tokens.append(piece)
            continue
        # CJK: emit unigrams + bigrams
        chars = [c for c in piece if _CJK_RE.match(c)]
        if not chars:
            continue
        tokens.extend(chars)
        tokens.extend([chars[i] + chars[i + 1] for i in range(len(chars) - 1)])
    return tokens


@dataclass
class BM25Hit:
    chunk_id: str
    score: float


class BM25Store:
    """Pickle-backed BM25 index.

    Stores:
      - bm25: the rank_bm25.BM25Okapi object
      - chunk_ids: parallel list of chunk ids
      - texts: parallel list of tokenized docs (kept for inspection if needed)
      - metas: parallel list of metadata dicts (doc_id, source, original text)
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.bm25: BM25Okapi | None = None
        self.chunk_ids: list[str] = []
        self.texts_raw: list[str] = []  # original text for re-hydration
        self.metas: list[dict] = []
        if self.path.exists():
            self.load()

    # ---- write ----

    def build(self, chunk_ids: list[str], texts: list[str], metas: list[dict]) -> None:
        assert len(chunk_ids) == len(texts) == len(metas)
        tokenized = [tokenize(t) for t in texts]
        self.bm25 = BM25Okapi(tokenized)
        self.chunk_ids = chunk_ids
        self.texts_raw = texts
        self.metas = metas
        self.save()

    def save(self) -> None:
        with self.path.open("wb") as f:
            pickle.dump(
                {
                    "bm25": self.bm25,
                    "chunk_ids": self.chunk_ids,
                    "texts_raw": self.texts_raw,
                    "metas": self.metas,
                },
                f,
            )

    def load(self) -> bool:
        try:
            with self.path.open("rb") as f:
                data = pickle.load(f)
        except Exception:
            return False
        self.bm25 = data.get("bm25")
        self.chunk_ids = data.get("chunk_ids", [])
        self.texts_raw = data.get("texts_raw", [])
        self.metas = data.get("metas", [])
        return True

    # ---- read ----

    def query(self, query: str, top_k: int) -> list[dict]:
        if self.bm25 is None or not self.chunk_ids:
            return []
        q_tokens = tokenize(query)
        scores = self.bm25.get_scores(q_tokens)
        # Argpartition-style: get top-k indices by score
        import numpy as np

        scores_arr = np.asarray(scores)
        if top_k >= len(scores_arr):
            top_idx = np.argsort(-scores_arr)
        else:
            top_idx = np.argpartition(-scores_arr, top_k)[:top_k]
            top_idx = top_idx[np.argsort(-scores_arr[top_idx])]

        hits = []
        for idx in top_idx:
            i = int(idx)
            if scores_arr[i] <= 0:
                continue
            hits.append(
                {
                    "chunk_id": self.chunk_ids[i],
                    "text": self.texts_raw[i],
                    "metadata": self.metas[i],
                    "score_sparse": float(scores_arr[i]),
                }
            )
            if len(hits) >= top_k:
                break
        return hits

    def count(self) -> int:
        return len(self.chunk_ids)

    def reset(self) -> None:
        self.bm25 = None
        self.chunk_ids = []
        self.texts_raw = []
        self.metas = []
        if self.path.exists():
            self.path.unlink()
