"""
RAGCache — diskcache wrapper for RAGResult persistence.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

import diskcache
from loguru import logger

def default_cache_dir():
    runtime = os.environ.get("RAGSCOPE_HOME", "D:/RAGScope_Runtime" if os.name == "nt" else "/mnt/d/RAGScope_Runtime")
    return str(Path(runtime) / "cache/ragscope/rag-v2")


_DEFAULT_CACHE_DIR = default_cache_dir()


class RAGCache:
    def __init__(self, cache_dir: str | None = None) -> None:
        # Resolve to absolute path to avoid issues with non-ASCII working dirs
        resolved = str(Path(cache_dir or default_cache_dir()).expanduser().resolve())
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
        if getattr(value, "error", None) or (isinstance(value, dict) and value.get("error")):
            return
        self._cache.set(key, value, expire=ttl_hours * 3600)

    def close(self) -> None:
        self._cache.close()

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
