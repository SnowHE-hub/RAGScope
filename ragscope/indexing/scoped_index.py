"""Explicit paper-only arrays with BM25 statistics pinned to a full reference.

Only selected papers require neural vectors. Sparse IDF/average length and
scoped candidate ordering retain the full reference corpus semantics.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from ragscope.indexing.array_index import ArrayIndex, tokenize
from ragscope.indexing.identity import artifact, atomic_json, signature, verify_artifacts


ENGINE = "ScopedArrayIndex-v1-full-reference-bm25"


class _ReferenceScores:
    def __init__(self, reference, positions):
        self.reference = reference
        self.positions = positions

    def get_scores(self, tokens):
        return self.reference.get_scores(tokens)[self.positions]


class ScopedArrayIndex(ArrayIndex):
    def __init__(self, chunks, identity, vectors=None):
        if identity.get("engine") != ENGINE or identity.get("scope") != "explicit_paper_only":
            raise ValueError("Scoped index requires an explicit paper-only identity")
        selected = identity.get("selected_doc_ids", [])
        if not selected or selected != sorted(set(selected)) or signature(selected) != identity.get("selected_doc_ids_sha256"):
            raise ValueError("Scoped document membership/hash is invalid")
        source = identity["sparse_reference"]
        if not verify_artifacts([source["chunks"], source["chunk_manifest"]]):
            raise ValueError("Full sparse reference source changed")
        manifest = json.loads(Path(source["chunk_manifest"]["path"]).read_text(encoding="utf-8"))
        if (manifest["signature"] != source["chunk_signature"]
                or manifest["signature"] != signature(manifest["identity"])
                or source["chunks"] not in manifest["artifacts"]
                or not verify_artifacts(manifest["artifacts"])):
            raise ValueError("Full sparse reference manifest differs")
        reference = [json.loads(line) for line in Path(source["chunks"]["path"]).read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(reference) != source["count"] or signature(reference) != source["chunks_content_sha256"]:
            raise ValueError("Full sparse reference content differs")
        if sorted({c["doc_id"] for c in reference}) != source["doc_ids"]:
            raise ValueError("Full sparse reference document membership differs")
        wanted = set(selected)
        positions = [i for i, row in enumerate(reference) if row["doc_id"] in wanted]
        if [reference[i] for i in positions] != chunks or {c["doc_id"] for c in chunks} != wanted:
            raise ValueError("Scoped chunks must contain all unchanged reference chunks of every selected paper")
        super().__init__(chunks, identity, vectors)
        reference_bm25 = BM25Okapi([tokenize(row["text"]) or ["__empty__"] for row in reference])
        self.bm25 = _ReferenceScores(reference_bm25, positions)

    def search(self, question, *, scope=None, **kwargs):
        if scope is None or scope not in self.by_doc:
            raise ValueError("Scoped-only index requires one supported explicit paper; global search is forbidden")
        return super().search(question, scope=scope, **kwargs)

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        manifest_path = directory / "manifest.json"
        if manifest_path.exists():
            existing = self.load(directory, self.identity)
            if existing.chunks != self.chunks:
                raise ValueError("Existing scoped index content differs")
            return
        atomic_json(directory / "chunks.json", self.chunks)
        atomic_json(directory / "bm25.json", {"tokenizer": "alnum-cjk-v1",
            "parameters": {"k1": 1.5, "b": .75, "epsilon": .25},
            "statistics_scope": "full_frozen_reference", "reference": self.identity["sparse_reference"]})
        outputs = [artifact(directory / name) for name in ("chunks.json", "bm25.json")]
        if self.vectors is not None:
            with (directory / "vectors.npy").open("wb") as handle:
                np.save(handle, np.asarray(self.vectors, dtype=np.float32), allow_pickle=False)
            outputs.append(artifact(directory / "vectors.npy"))
        atomic_json(manifest_path, {"signature": self.signature, "identity": self.identity,
            "count": len(self.chunks), "artifacts": outputs, "scoped_only": True,
            "neural_vector_count": 0 if self.vectors is None else len(self.vectors),
            "sparse_reference_count": self.identity["sparse_reference"]["count"]})

    @classmethod
    def load(cls, directory, expected_identity):
        directory = Path(directory)
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        expected_files = {"chunks.json", "bm25.json"} | ({"vectors.npy"} if manifest["neural_vector_count"] else set())
        paths = [Path(item["path"]).resolve() for item in manifest["artifacts"]]
        if ({p.name for p in paths} != expected_files or len(paths) != len(expected_files)
                or any(p.parent != directory.resolve() for p in paths)
                or manifest["signature"] != signature(expected_identity)
                or manifest["identity"] != expected_identity or manifest.get("scoped_only") is not True
                or not verify_artifacts(manifest["artifacts"])):
            raise ValueError("Scoped index identity/content validation failed")
        chunks = json.loads((directory / "chunks.json").read_text(encoding="utf-8"))
        if len(chunks) != manifest["count"]:
            raise ValueError("Scoped index count differs")
        vectors = np.load(directory / "vectors.npy", allow_pickle=False, mmap_mode="r") if manifest["neural_vector_count"] else None
        if vectors is not None and (len(vectors) != manifest["neural_vector_count"] or vectors.dtype != np.float32 or not np.isfinite(vectors).all()):
            raise ValueError("Scoped vectors differ from declared arrays")
        return cls(chunks, expected_identity, vectors)
