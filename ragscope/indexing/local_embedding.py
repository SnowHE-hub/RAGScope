"""Qwen official GGUF CPU embedding adapter; never silently truncates inputs."""
from __future__ import annotations
import time
import urllib.parse
from pathlib import Path
import numpy as np
import requests
from ragscope.indexing.array_index import ArrayCache

QUERY_INSTRUCTION = "Instruct: Given a question about a scientific paper, retrieve relevant passages that answer the query\nQuery:"


class HFJsonTokenizer:
    def __init__(self, path: str | Path, tokenizer_id: str, revision: str):
        from tokenizers import Tokenizer
        self.backend = Tokenizer.from_file(str(path))
        self.tokenizer_id, self.revision = tokenizer_id, revision

    def encode(self, text: str):
        return self.backend.encode(text, add_special_tokens=False).ids

    def decode(self, ids):
        return self.backend.decode(ids, skip_special_tokens=False)


class LocalEmbeddingService:
    def __init__(self, base_url: str, identity: dict, tokenizer, cache_dir: Path, max_input_tokens: int = 1100):
        if urllib.parse.urlparse(base_url).hostname not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError("Only project loopback embedding backend is allowed")
        self.base_url, self.identity, self.tokenizer = base_url.rstrip("/"), identity, tokenizer
        self.cache = ArrayCache(cache_dir, identity)
        self.max_input_tokens = max_input_tokens
        self.last_metrics = []

    def embed(self, texts: list[str]):
        rows = []
        self.last_metrics = []
        for text in texts:
            tokens = len(self.tokenizer.encode(text))
            if tokens > self.max_input_tokens:
                raise ValueError(f"Embedding input {tokens} exceeds {self.max_input_tokens}; no truncation allowed")
            started = time.perf_counter()
            vector = self.cache.get(text)
            cached = vector is not None
            if vector is None:
                response = requests.post(self.base_url + "/v1/embeddings", json={"input": text}, timeout=180)
                response.raise_for_status()
                payload = response.json()
                vector = np.asarray(payload["data"][0]["embedding"], dtype=np.float32)
                norm = np.linalg.norm(vector)
                if not np.isfinite(norm) or norm == 0:
                    raise ValueError("Invalid backend embedding")
                vector /= norm
                self.cache.put(text, vector)
            rows.append(vector)
            self.last_metrics.append({"input_tokens": tokens, "cache_hit": cached, "seconds": time.perf_counter() - started})
        return np.stack(rows) if rows else np.empty((0, self.identity.get("dimensions", 1024)), dtype=np.float32)

    def embed_query(self, text: str):
        return self.embed([QUERY_INSTRUCTION + text])[0]

    def get_dimension(self):
        return self.identity.get("dimensions", 1024)
