"""Manifest-bound numpy vector index and JSON sparse index for v2 scoped retrieval."""
from __future__ import annotations
import io
import json
import os
import re
import time
from pathlib import Path
import numpy as np
from rank_bm25 import BM25Okapi
from ragscope.indexing.identity import artifact, atomic_json, signature, verify_artifacts


class ArrayCache:
    def __init__(self, path: Path, model_identity: dict):
        self.path = Path(path) / signature(model_identity)
        self.path.mkdir(parents=True, exist_ok=True)
        self.model_identity = model_identity

    def key(self, text: str) -> str:
        return signature({"text": text, "model": self.model_identity})

    def get(self, text: str):
        path = self.path / (self.key(text) + ".npy")
        manifest = path.with_suffix(".json")
        if not path.exists() or not manifest.exists():
            return None
        info = json.loads(manifest.read_text(encoding="utf-8"))
        if (info.get("model") != self.model_identity or info.get("key") != self.key(text)
                or Path(info["array"]["path"]).resolve() != path.resolve()):
            raise ValueError("Embedding cache identity/path mismatch")
        if not verify_artifacts([info["array"]]):
            raise ValueError("Embedding cache checksum mismatch")
        result = np.load(path, allow_pickle=False)
        if result.ndim != 1 or result.dtype != np.float32 or not np.isfinite(result).all():
            raise ValueError("Invalid embedding cache array")
        return result

    def put(self, text: str, vector):
        vector = np.asarray(vector, dtype=np.float32)
        if vector.ndim != 1 or not np.isfinite(vector).all():
            raise ValueError("Embedding must be finite one dimensional vector")
        path = self.path / (self.key(text) + ".npy")
        temporary = path.with_suffix(f".{os.getpid()}.tmp")
        with temporary.open("wb") as handle:
            np.save(handle, vector, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        atomic_json(path.with_suffix(".json"), {"array": artifact(path), "key": self.key(text), "model": self.model_identity})


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+|[\u3400-\u9fff]", text.casefold())


class ArrayIndex:
    def __init__(self, chunks: list[dict], identity: dict, vectors=None):
        self.chunks = chunks
        self.identity = identity
        self.signature = signature(identity)
        self.vectors = vectors
        self.by_doc = {}
        for i, item in enumerate(chunks):
            self.by_doc.setdefault(item["doc_id"], []).append(i)
        self.bm25 = BM25Okapi([tokenize(row["text"]) or ["__empty__"] for row in chunks]) if chunks else None
        if vectors is not None and (len(vectors) != len(chunks) or vectors.ndim != 2):
            raise ValueError("Vector/chunk shape mismatch")

    def save(self, directory: Path):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        manifest = directory / "manifest.json"
        if manifest.exists():
            existing = self.load(directory, self.identity)
            if signature(existing.chunks) != signature(self.chunks):
                raise ValueError("Existing index content changed")
            return
        chunks_path = directory / "chunks.json"
        atomic_json(chunks_path, self.chunks)
        atomic_json(directory / "bm25.json", {"tokenizer": "alnum-cjk-v1", "parameters": {"k1": 1.5, "b": .75, "epsilon": .25}, "corpus_hash": signature(self.chunks)})
        outputs = [artifact(chunks_path), artifact(directory / "bm25.json")]
        if self.vectors is not None:
            with (directory / "vectors.npy").open("wb") as handle:
                np.save(handle, np.asarray(self.vectors, dtype=np.float32), allow_pickle=False)
            outputs.append(artifact(directory / "vectors.npy"))
        atomic_json(manifest, {"signature": self.signature, "identity": self.identity, "count": len(self.chunks), "artifacts": outputs})

    @classmethod
    def load(cls, directory: Path, expected_identity: dict):
        directory = Path(directory)
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        expected_files = {"chunks.json", "bm25.json"} | ({"vectors.npy"} if (directory / "vectors.npy").exists() else set())
        paths = [Path(item["path"]).resolve() for item in manifest["artifacts"]]
        if ({p.name for p in paths} != expected_files or len(paths) != len(expected_files)
                or any(p.parent != directory.resolve() for p in paths)):
            raise ValueError("Index artifact path mismatch")
        if (manifest["signature"] != signature(expected_identity) or manifest["identity"] != expected_identity
                or not verify_artifacts(manifest["artifacts"])):
            raise ValueError("Index identity/content validation failed")
        chunks = json.loads((directory / "chunks.json").read_text(encoding="utf-8"))
        if len(chunks) != manifest["count"]:
            raise ValueError("Index count mismatch")
        vectors = np.load(directory / "vectors.npy", allow_pickle=False, mmap_mode="r") if (directory / "vectors.npy").exists() else None
        return cls(chunks, expected_identity, vectors)

    def search(self, question: str, *, method: str = "bm25", query_vector=None, embedding_identity=None, scope: str | None = None, k: int = 10, candidate_depth: int = 50):
        started = time.perf_counter()
        self.last_timings = {"bm25_ms": None, "dense_search_ms": None, "fusion_ms": None}
        ids = self.by_doc.get(scope, []) if scope is not None else range(len(self.chunks))
        ids = list(ids)
        if not ids:
            return []
        if k <= 0 or candidate_depth < k:
            raise ValueError("Invalid retrieval depth")
        if method in {"bm25", "hybrid"}:
            sparse_start = time.perf_counter()
            sparse = self.bm25.get_scores(tokenize(question))
            sparse_rank = sorted(ids, key=lambda i: (-float(sparse[i]), self.chunks[i]["chunk_id"]))[:candidate_depth]
            self.last_timings["bm25_ms"] = (time.perf_counter() - sparse_start) * 1000
        if method == "bm25":
            ranked, scores = sparse_rank, {i: float(sparse[i]) for i in sparse_rank}
        else:
            if self.vectors is None or query_vector is None:
                raise ValueError("Dense retrieval requires vectors")
            if embedding_identity != self.identity.get("embedding"):
                raise ValueError("Query vector space identity mismatch")
            dense_start = time.perf_counter()
            dense = self.vectors @ np.asarray(query_vector, dtype=np.float32)
            dense_rank = sorted(ids, key=lambda i: (-float(dense[i]), self.chunks[i]["chunk_id"]))[:candidate_depth]
            self.last_timings["dense_search_ms"] = (time.perf_counter() - dense_start) * 1000
            if method == "dense":
                ranked, scores = dense_rank, {i: float(dense[i]) for i in dense_rank}
            elif method == "hybrid":
                fusion_start = time.perf_counter()
                scores = {}
                for ranking in (sparse_rank, dense_rank):
                    for rank, i in enumerate(ranking, start=1):
                        scores[i] = scores.get(i, 0.0) + .5 / (60 + rank)
                ranked = sorted(scores, key=lambda i: (-scores[i], self.chunks[i]["chunk_id"]))
                self.last_timings["fusion_ms"] = (time.perf_counter() - fusion_start) * 1000
            else:
                raise ValueError(method)
        results = [dict(self.chunks[i], score=scores[i]) for i in ranked[:k]]
        self.last_timings["retrieval_total_ms"] = (time.perf_counter() - started) * 1000
        return results
