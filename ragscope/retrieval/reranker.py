"""Query-inclusive cross-encoder windows with max aggregation; no hidden truncation."""
from __future__ import annotations

import os
from typing import Dict, List

import numpy as np

from ragscope.ingestion.chunkers import TokenizerAdapter


class Reranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-base", *, revision: str | None = None,
                 device: str = "cpu", max_length: int = 512, window_overlap: int = 32,
                 batch_size: int = 4, model=None, tokenizer=None, local_files_only: bool = False) -> None:
        if not isinstance(max_length, int) or isinstance(max_length, bool) or not 4 <= max_length <= 512:
            raise ValueError("Reranker max_length must be within [4,512]")
        if not isinstance(window_overlap, int) or not 0 <= window_overlap < max_length:
            raise ValueError("Invalid window_overlap")
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be positive")
        if device in {None, "auto"} or (device != "cpu" and os.environ.get("CUDA_VISIBLE_DEVICES") == ""):
            raise ValueError("Reranker requires an explicit permitted device")
        self.model_name, self.device = model_name, device
        if model is None:
            from sentence_transformers import CrossEncoder
            model = CrossEncoder(model_name, max_length=max_length, device=device, revision=revision,
                                 trust_remote_code=False, local_files_only=local_files_only)
        self.model = model
        self.tokenizer = tokenizer or getattr(model, "tokenizer", None)
        if self.tokenizer is None:
            raise ValueError("Reranker tokenizer is required")
        actual_limit = getattr(model, "max_length", max_length)
        self.max_length = min(max_length, actual_limit) if isinstance(actual_limit, int) else max_length
        tokenizer_limit = getattr(self.tokenizer, "model_max_length", None)
        if isinstance(tokenizer_limit, int) and 0 < tokenizer_limit < 10**6:
            self.max_length = min(self.max_length, tokenizer_limit)
        self.window_overlap, self.batch_size = window_overlap, batch_size
        revision = getattr(getattr(getattr(model, "model", None), "config", None), "_commit_hash", None) or revision
        if not revision:
            raise ValueError("A pinned reranker revision or resolved commit hash is required")
        self.identity = {"model": model_name, "revision": revision, "device": device,
                         "max_length": self.max_length, "window_overlap": window_overlap,
                         "aggregation": "windowed_max", "batch_size": batch_size}
        self.last_window_records: list[dict] = []
        self.text_tokenizer = TokenizerAdapter(self.tokenizer, model_name, revision)

    def _pair_tokens(self, query: str, text: str) -> int:
        encoded = self.tokenizer(query, text, add_special_tokens=True, truncation=False)
        return len(encoded["input_ids"])

    def _windows(self, query: str, text: str) -> list[tuple[int, int, int]]:
        if self._pair_tokens(query, "") >= self.max_length:
            raise ValueError("Question plus pair special tokens leaves no reranker document budget")
        if not text:
            return [(0, 0, self._pair_tokens(query, ""))]
        windows, start = [], 0
        while start < len(text):
            # Full pair is counted, including question, pair separators and content.
            low, high = start, min(len(text), start + self.max_length * 4)
            if self._pair_tokens(query, text[start:high]) <= self.max_length:
                low = high
                while high < len(text):
                    high = min(len(text), start + (high - start) * 2)
                    if self._pair_tokens(query, text[start:high]) > self.max_length:
                        break
                    low = high
            while low + 1 < high:
                mid = (low + high) // 2
                if self._pair_tokens(query, text[start:mid]) <= self.max_length:
                    low = mid
                else:
                    high = mid
            end = low
            if end <= start:
                raise ValueError("Reranker budget cannot fit one document Unicode code point")
            if windows and end <= windows[-1][1]:
                start = windows[-1][1]
                continue
            windows.append((start, end, self._pair_tokens(query, text[start:end])))
            if end == len(text):
                break
            left, right = start + 1, end
            while left < right:
                mid = (left + right) // 2
                if self.text_tokenizer.count(text[mid:end]) <= self.window_overlap:
                    right = mid
                else:
                    left = mid + 1
            start = left
        return windows

    def rerank(self, query: str, candidates: List[Dict], top_k: int = 10) -> List[Dict]:
        if not isinstance(top_k, int) or top_k < 1:
            raise ValueError("top_k must be positive")
        self.last_window_records = []
        if not candidates:
            return []
        if len({candidate["chunk_id"] for candidate in candidates}) != len(candidates):
            raise ValueError("Reranker candidate IDs must be unique")
        pairs, owners = [], []
        for i, candidate in enumerate(candidates):
            for start, end, count in self._windows(query, candidate["text"]):
                pairs.append((query, candidate["text"][start:end]))
                owners.append(i)
                self.last_window_records.append({"chunk_id": candidate["chunk_id"], "start": start,
                                                 "end": end, "pair_tokens": count})
        raw = np.asarray(self.model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False))
        if raw.size != len(pairs) or not np.isfinite(raw).all():
            raise ValueError("Invalid reranker scores")
        scores = np.full(len(candidates), -np.inf)
        for i, score, record in zip(owners, raw.reshape(-1), self.last_window_records):
            scores[i] = max(scores[i], float(score))
            record["score"] = float(score)
        ranking = sorted(range(len(candidates)), key=lambda i: (-scores[i], candidates[i]["chunk_id"]))[:top_k]
        return [{**candidates[i], "rerank_score": float(scores[i]),
                 "rerank_aggregation": "windowed_max",
                 "rerank_windows": [dict(record) for record in self.last_window_records if record["chunk_id"] == candidates[i]["chunk_id"]]}
                for i in ranking]

    def retrieve(self, query: str, retriever, *, candidate_depth: int = 50, top_k: int = 10) -> List[Dict]:
        if candidate_depth < top_k or candidate_depth < 1:
            raise ValueError("candidate_depth must be >= top_k")
        return self.rerank(query, retriever.retrieve(query, k=candidate_depth), top_k=top_k)
