"""
EmbeddingService: SentenceTransformer wrapper with L2 normalisation and diskcache.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import List

import diskcache
import numpy as np
from loguru import logger
from sentence_transformers import SentenceTransformer

# BGE v1.5 retrieval instruction for query encoding
_BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# Use home-relative path so diskcache works on non-ASCII working dirs (Windows)
_DEFAULT_EMB_CACHE = os.path.join(os.path.expanduser("~"), ".cache", "ragscope", "embeddings")


class EmbeddingService:
    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        device: str = "cpu",
        cache_dir: str = _DEFAULT_EMB_CACHE,
    ) -> None:
        self.model_name = model_name
        self.device = device

        # Resolve to absolute path before passing to diskcache
        resolved = str(Path(cache_dir).expanduser().resolve())
        Path(resolved).mkdir(parents=True, exist_ok=True)
        self.cache = diskcache.Cache(resolved)

        logger.info(f"Loading embedding model '{model_name}' on {device}")
        self.model = SentenceTransformer(model_name, device=device)
        # sentence-transformers ≥ 3.x renamed the method
        if hasattr(self.model, "get_embedding_dimension"):
            self._dim: int = self.model.get_embedding_dimension()
        else:
            self._dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"Model ready — embedding dim: {self._dim}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed(
        self,
        texts: List[str],
        batch_size: int = 64,
        show_progress: bool = True,
    ) -> np.ndarray:
        """Embed a list of texts.  Returns L2-normalised float32 array (N, D)."""
        if not texts:
            return np.empty((0, self._dim), dtype=np.float32)

        results: List[np.ndarray | None] = [None] * len(texts)
        miss_texts: List[str] = []
        miss_idxs: List[int] = []

        for i, text in enumerate(texts):
            cached = self.cache.get(self._cache_key(text))
            if cached is not None:
                results[i] = cached
            else:
                miss_texts.append(text)
                miss_idxs.append(i)

        if miss_texts:
            embeddings = self.model.encode(
                miss_texts,
                batch_size=batch_size,
                show_progress_bar=show_progress and len(miss_texts) > batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
            ).astype(np.float32)

            for local_i, (text, emb) in enumerate(zip(miss_texts, embeddings)):
                self.cache.set(self._cache_key(text), emb)
                results[miss_idxs[local_i]] = emb

        return np.stack(results).astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single retrieval query with BGE instruction prefix. Returns (D,)."""
        return self.embed([_BGE_QUERY_PREFIX + query], show_progress=False)[0]

    def get_dimension(self) -> int:
        return self._dim

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _cache_key(self, text: str) -> str:
        return hashlib.sha256((text + self.model_name).encode("utf-8")).hexdigest()
