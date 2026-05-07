"""
RAGCache — diskcache wrapper for RAGResult persistence.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

import diskcache
from loguru import logger

# Default to user home so diskcache works on paths with non-ASCII characters
_DEFAULT_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "ragscope", "rag")


class RAGCache:
    def __init__(self, cache_dir: str = _DEFAULT_CACHE_DIR) -> None:
        # Resolve to absolute path to avoid issues with non-ASCII working dirs
        resolved = str(Path(cache_dir).expanduser().resolve())
        Path(resolved).mkdir(parents=True, exist_ok=True)
        self._cache = diskcache.Cache(resolved)
        self._hits = 0
        self._misses = 0

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def get(self, key: str) -> Optional[Any]:
        value = self._cache.get(key)
        if value is not None:
            self._hits += 1
        else:
            self._misses += 1
        return value

    def set(self, key: str, value: Any, ttl_hours: int = 168) -> None:
        self._cache.set(key, value, expire=ttl_hours * 3600)

    # ------------------------------------------------------------------
    # Housekeeping
    # ------------------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        total = self._hits + self._misses
        return {
            "hit_count":  self._hits,
            "miss_count": self._misses,
            "hit_rate":   round(self._hits / total, 4) if total else 0.0,
        }

    def clear(self) -> None:
        self._cache.clear()
        self._hits = 0
        self._misses = 0
        logger.info("RAG cache cleared.")

    def size(self) -> int:
        """Number of entries currently in the cache."""
        return len(self._cache)
