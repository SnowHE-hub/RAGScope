"""
BM25Index: rank_bm25-backed sparse retrieval index with NLTK stopword filtering.
"""

from __future__ import annotations

import pickle
import re
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger
from rank_bm25 import BM25Okapi
from tqdm import tqdm
from ragscope.indexing.identity import atomic_json, signature

# ------------------------------------------------------------------
# Fixed, explicitly prepared NLTK English stopword snapshot (no import downloads).
# ------------------------------------------------------------------

_STOPWORDS_PATH = Path(__file__).with_name("stopwords_en.txt")
_STOPWORDS: frozenset = frozenset(_STOPWORDS_PATH.read_text(encoding="utf-8").splitlines())
_TOKEN_RE = re.compile(r"[a-z0-9]+")   # keep alphanumeric only


# ------------------------------------------------------------------
# BM25Index
# ------------------------------------------------------------------

class BM25Index:
    def __init__(self, index_path: str = "data/indexes/bm25.pkl") -> None:
        self.index_path = Path(index_path)
        self.bm25: Optional[BM25Okapi] = None
        # Lightweight chunk records kept in RAM for result assembly
        self.chunks: List[Dict] = []

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self, chunks_df: pd.DataFrame) -> None:
        """Tokenize all chunks and build a BM25Okapi index, then save to disk."""
        logger.info(f"Building BM25 index — {len(chunks_df):,} chunks")

        tokenized_corpus: List[List[str]] = []
        self.chunks = []

        for _, row in tqdm(chunks_df.iterrows(), total=len(chunks_df), desc="BM25 tokenise", unit="chunk"):
            tokens = self._tokenize(str(row["text"]))
            tokenized_corpus.append(tokens)
            self.chunks.append(
                {
                    "chunk_id": str(row["chunk_id"]),
                    "doc_id": str(row["doc_id"]),
                    "text": str(row["text"]),
                }
            )

        self.bm25 = BM25Okapi(tokenized_corpus)
        self.save()
        logger.info(f"BM25 index built and saved → {self.index_path}")

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query: str, k: int = 10) -> List[Dict]:
        """Return top-k chunks by BM25 score.

        Each result: {chunk_id, doc_id, text, score}
        """
        if self.bm25 is None:
            raise RuntimeError("BM25 index not built or loaded.  Call build() or load() first.")

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        scores: np.ndarray = self.bm25.get_scores(query_tokens)
        top_idxs = np.argsort(scores)[::-1][:k]

        results: List[Dict] = []
        for idx in top_idxs:
            score = float(scores[idx])
            if score <= 0.0:
                break   # BM25 scores are non-negative; 0 means no term overlap
            chunk = self.chunks[idx]
            results.append(
                {
                    "chunk_id": chunk["chunk_id"],
                    "doc_id": chunk["doc_id"],
                    "text": chunk["text"],
                    "score": score,
                }
            )

        return results

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | None = None) -> None:
        target = Path(path) if path else self.index_path
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise FileExistsError(f"Refusing to replace index: {target}")
        payload = {"format": "ragscope-bm25-json-v2", "chunks": self.chunks,
                   "stopwords": sorted(_STOPWORDS), "tokenizer": "legacy-alnum-v2",
                   "parameters": {"k1": 1.5, "b": .75, "epsilon": .25}}
        payload["signature"] = signature(payload)
        atomic_json(target, payload)
        logger.info(f"BM25 index saved → {target}")

    @classmethod
    def load(cls, path: str, *, trusted_legacy: bool = False) -> "BM25Index":
        """Load a previously saved index from disk."""
        target = Path(path)
        if not target.exists():
            raise FileNotFoundError(f"BM25 index file not found: {target}")
        instance = cls(index_path=str(target))
        with target.open("rb") as header:
            is_pickle = header.read(1) == b"\x80"
        if is_pickle:
            if not trusted_legacy:
                raise ValueError("Legacy pickle needs explicit trusted_legacy=True in isolated migration")
            with open(target, "rb") as fh:
                data = pickle.load(fh)
            instance.bm25, instance.chunks = data["bm25"], data["chunks"]
        else:
            data = json.loads(target.read_text(encoding="utf-8"))
            expected = data.pop("signature")
            if signature(data) != expected:
                raise ValueError("BM25 JSON signature mismatch")
            instance.chunks = data["chunks"]
            words = frozenset(data["stopwords"])
            tokens = [[t for t in _TOKEN_RE.findall(c["text"].lower()) if t not in words] or ["__empty__"] for c in instance.chunks]
            instance.bm25 = BM25Okapi(tokens, **data["parameters"])
        logger.info(f"BM25 index loaded ← {target} ({len(instance.chunks):,} chunks)")
        return instance

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Lowercase → keep alphanumeric → remove NLTK stopwords."""
        return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS]
