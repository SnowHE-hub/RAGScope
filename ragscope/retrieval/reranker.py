"""
Reranker — cross-encoder reranking of candidate chunks.

Uses sentence_transformers.CrossEncoder so any HuggingFace
cross-encoder model works (BAAI/bge-reranker-*, cross-encoder/*, …).
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
from loguru import logger
from sentence_transformers import CrossEncoder


class Reranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-base") -> None:
        self.model_name = model_name
        logger.info(f"Loading reranker: {model_name}")
        self.model = CrossEncoder(model_name, max_length=512)
        logger.info("Reranker ready.")

    def rerank(
        self,
        query: str,
        candidates: List[Dict],
        top_k: int = 5,
    ) -> List[Dict]:
        """Score every (query, candidate_text) pair and return top_k by score.

        Adds a *rerank_score* field to each result dict.
        """
        if not candidates:
            return []

        pairs = [(query, c["text"]) for c in candidates]
        raw_scores: np.ndarray = self.model.predict(pairs, show_progress_bar=False)

        top_k = min(top_k, len(candidates))
        ranked_idxs = np.argsort(raw_scores)[::-1][:top_k]

        results: List[Dict] = []
        for idx in ranked_idxs:
            item = dict(candidates[idx])
            item["rerank_score"] = float(raw_scores[idx])
            results.append(item)

        return results
