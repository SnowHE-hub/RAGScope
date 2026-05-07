"""
HybridRetriever — Reciprocal Rank Fusion of dense + sparse results.

RRF score formula (1-indexed ranks, constant k=60):
    score(d) = alpha * 1/(60 + dense_rank(d))
             + (1-alpha) * 1/(60 + bm25_rank(d))

alpha=1.0 → pure dense
alpha=0.0 → pure sparse
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

from ragscope.retrieval.base import BaseRetriever
from ragscope.retrieval.bm25_retriever import BM25Retriever
from ragscope.retrieval.dense_retriever import DenseRetriever

_RRF_K = 60  # standard constant


class HybridRetriever(BaseRetriever):
    def __init__(
        self,
        dense_retriever: DenseRetriever,
        bm25_retriever: BM25Retriever,
        alpha: float = 0.5,
    ) -> None:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        self.dense = dense_retriever
        self.bm25 = bm25_retriever
        self.alpha = alpha

    def retrieve(self, query: str, k: int = 5) -> List[Dict]:
        fetch_k = k * 2   # oversample from each source before fusion

        dense_hits = self.dense.retrieve(query, k=fetch_k)
        bm25_hits  = self.bm25.retrieve(query, k=fetch_k)

        rrf_scores: Dict[str, float] = defaultdict(float)
        chunk_store: Dict[str, Dict] = {}

        # Dense contribution  (ranks are 1-indexed: rank 1 → denominator 61)
        for rank, hit in enumerate(dense_hits, start=1):
            cid = hit["chunk_id"]
            rrf_scores[cid] += self.alpha * (1.0 / (_RRF_K + rank))
            chunk_store.setdefault(cid, hit)

        # BM25 contribution
        for rank, hit in enumerate(bm25_hits, start=1):
            cid = hit["chunk_id"]
            rrf_scores[cid] += (1.0 - self.alpha) * (1.0 / (_RRF_K + rank))
            chunk_store.setdefault(cid, hit)

        # Sort by fused RRF score, return top-k
        top = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:k]

        results: List[Dict] = []
        for cid, score in top:
            item = dict(chunk_store[cid])
            item["score"] = score
            results.append(item)

        return results
