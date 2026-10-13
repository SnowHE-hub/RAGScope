"""Explicit-device SentenceTransformer embeddings with verified npy caches."""
from __future__ import annotations

import os
from pathlib import Path
from typing import List

import numpy as np

from ragscope.indexing.array_index import ArrayCache
from ragscope.indexing.identity import signature

_BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
_DEFAULT_EMB_CACHE = os.path.join(os.environ.get("RAGSCOPE_HOME", "D:/RAGScope_Runtime" if os.name == "nt" else "/mnt/d/RAGScope_Runtime"), "cache", "ragscope", "embeddings")


class EmbeddingService:
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", device: str = "cpu",
                 cache_dir: str = _DEFAULT_EMB_CACHE, *, revision: str | None = None,
                 model=None, tokenizer=None, max_input_tokens: int | None = None,
                 query_prefix: str | None = None, pooling=None, precision: str = "float32",
                 local_files_only: bool = False) -> None:
        if device in {None, "auto"}:
            raise ValueError("Embedding device must be explicit")
        if device != "cpu" and os.environ.get("CUDA_VISIBLE_DEVICES") == "":
            raise ValueError("GPU use is prohibited by this CPU-only process")
        self.model_name, self.device = model_name, device
        if model is None:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(model_name, device=device, revision=revision,
                                        cache_folder=os.environ.get("SENTENCE_TRANSFORMERS_HOME"),
                                        trust_remote_code=False, local_files_only=local_files_only)
        self.model = model
        self.tokenizer = tokenizer or getattr(model, "tokenizer", None)
        if self.tokenizer is None:
            raise ValueError("The real embedding tokenizer is required to prevent silent truncation")
        module = model._first_module() if hasattr(model, "_first_module") else None
        config = getattr(getattr(module, "auto_model", None), "config", None)
        revision = getattr(config, "_commit_hash", None) or revision
        if not revision:
            raise ValueError("A pinned embedding revision or resolved commit hash is required")
        model_limit = getattr(model, "max_seq_length", None)
        tokenizer_limit = getattr(self.tokenizer, "model_max_length", None)
        limits = [int(value) for value in (model_limit, tokenizer_limit) if isinstance(value, (int, float)) and 0 < value < 10**6]
        if max_input_tokens is not None:
            if not isinstance(max_input_tokens, int) or isinstance(max_input_tokens, bool) or max_input_tokens < 1:
                raise ValueError("max_input_tokens must be positive")
            limits.append(max_input_tokens)
        if not limits:
            raise ValueError("An explicit embedding input token limit is required")
        self.max_input_tokens = min(limits)
        self.query_prefix = query_prefix if query_prefix is not None else (_BGE_QUERY_PREFIX if "bge-" in model_name.lower() else "")
        if hasattr(model, "get_embedding_dimension"):
            self._dim = int(model.get_embedding_dimension())
        else:
            self._dim = int(model.get_sentence_embedding_dimension())
        if pooling is None:
            pooling = [module.get_config_dict() for module in getattr(model, "_modules", {}).values()
                       if module.__class__.__name__ == "Pooling"] or "model_config"
        self.identity = {"model": model_name, "revision": revision,
                         "tokenizer": getattr(self.tokenizer, "name_or_path", model_name),
                         "tokenizer_revision": revision, "pooling": pooling,
                         "normalize": True, "query_prefix": self.query_prefix,
                         "max_input_tokens": self.max_input_tokens, "precision": str(getattr(model, "dtype", precision)),
                         "backend": "sentence_transformers", "automatic_prompt": "disabled"}
        self.cache = ArrayCache(Path(cache_dir).expanduser().resolve(), self.identity)
        self.last_input_token_counts: list[int] = []
        self.last_cache_hits = 0

    def _count_input(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=True, truncation=False))

    def embed(self, texts: List[str], batch_size: int = 64, show_progress: bool = True) -> np.ndarray:
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.last_input_token_counts = [self._count_input(text) for text in texts]
        oversized = [(i, count) for i, count in enumerate(self.last_input_token_counts) if count > self.max_input_tokens]
        if oversized:
            raise ValueError(f"Embedding input exceeds {self.max_input_tokens} tokens including special tokens: {oversized[:5]}")
        self.last_cache_hits = 0
        if not texts:
            return np.empty((0, self._dim), dtype=np.float32)
        results, pending = [None] * len(texts), {}
        for i, text in enumerate(texts):
            cached = self.cache.get(text)
            if cached is not None:
                if cached.shape != (self._dim,):
                    raise ValueError("Cached embedding dimension mismatch")
                results[i] = cached
                self.last_cache_hits += 1
            else:
                pending.setdefault(text, []).append(i)
        if pending:
            missing = list(pending)
            embeddings = np.asarray(self.model.encode(missing, batch_size=batch_size,
                show_progress_bar=show_progress and len(missing) > batch_size,
                normalize_embeddings=True, convert_to_numpy=True, prompt="", device=self.device), dtype=np.float32)
            if embeddings.shape != (len(missing), self._dim) or not np.isfinite(embeddings).all():
                raise ValueError("Invalid model embedding shape or non-finite output")
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            if np.any(norms <= 0):
                raise ValueError("Model returned a zero embedding")
            embeddings = embeddings / norms
            for text, vector in zip(missing, embeddings):
                self.cache.put(text, vector)
                for i in pending[text]:
                    results[i] = vector
        return np.stack(results).astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        return self.embed([self.query_prefix + query], show_progress=False)[0]

    def get_dimension(self) -> int:
        return self._dim

    def _cache_key(self, text: str) -> str:
        return signature({"text": text, "model": self.identity})
