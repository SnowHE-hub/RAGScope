"""Chroma vector store with checked manifests and explicit vector-space identity."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from ragscope.indexing.identity import atomic_json, canonical_json, signature


class DenseIndex:
    def __init__(self, collection_name: str, persist_dir: str = "data/indexes/chroma", *, client=None) -> None:
        self.collection_name, self.persist_dir = collection_name, str(persist_dir)
        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        self.manifest_path = Path(persist_dir) / (signature(collection_name)[:24] + ".manifest.json")
        if client is None:
            import chromadb
            client = chromadb.PersistentClient(path=str(persist_dir))
        self._client = client
        self.collection = client.get_or_create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})

    def _manifest(self, *, require_complete: bool = True):
        if not self.manifest_path.exists():
            return None
        data = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if data.get("signature") != signature(data.get("identity")):
            raise ValueError("Dense manifest signature is invalid")
        if require_complete and (data.get("status") != "complete" or self.collection.count() != data["identity"]["count"]):
            raise ValueError("Dense index is incomplete or its count differs from manifest")
        return data

    def build(self, chunks_df, embedding_service, batch_size: int = 500) -> None:
        """Reuse only an identical, complete index; resume owned partial batches."""
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be positive")
        rows = chunks_df.to_dict("records") if hasattr(chunks_df, "to_dict") else list(chunks_df)
        ids = [row["chunk_id"] for row in rows]
        if len(set(ids)) != len(ids):
            raise ValueError("Chunk IDs must be unique within an index")
        identity = {"engine": "chroma", "distance": "cosine", "collection": self.collection_name,
                    "embedding": embedding_service.identity, "dimension": embedding_service.get_dimension(),
                    "count": len(rows), "chunks_digest": signature(rows)}
        expected_signature = signature(identity)
        current = self._manifest(require_complete=False)
        if current is not None and current["signature"] != expected_signature:
            raise ValueError("Dense identity mismatch; use a new isolated collection, never overwrite")
        if current is None and self.collection.count():
            raise ValueError("Existing legacy collection has no verified manifest; preserve it and build a new collection")
        if current is not None and current.get("status") == "complete":
            self._manifest()
            return
        manifest = {"identity": identity, "signature": expected_signature, "status": "building"}
        atomic_json(self.manifest_path, manifest)
        existing_ids = set()
        if self.collection.count():
            existing = self.collection.get(include=["documents", "metadatas"])
            by_id = {row["chunk_id"]: row for row in rows}
            for cid, text, metadata in zip(existing["ids"], existing["documents"], existing["metadatas"]):
                if cid not in by_id or text != by_id[cid]["text"] or metadata.get("payload_hash") != signature(by_id[cid]):
                    raise ValueError("Partial index content mismatch; no rows were overwritten")
                existing_ids.add(cid)
        pending = [row for row in rows if row["chunk_id"] not in existing_ids]
        for start in range(0, len(pending), batch_size):
            batch = pending[start:start + batch_size]
            vectors = np.asarray(embedding_service.embed([row["text"] for row in batch], show_progress=False), dtype=np.float32)
            if vectors.shape != (len(batch), identity["dimension"]) or not np.isfinite(vectors).all():
                raise ValueError("Embedding shape or values invalid")
            metadata = [{"doc_id": row["doc_id"], "chunk_id": row["chunk_id"],
                         "payload_hash": signature(row), "payload_json": canonical_json(row)} for row in batch]
            self.collection.add(ids=[row["chunk_id"] for row in batch], embeddings=vectors.tolist(),
                                documents=[row["text"] for row in batch], metadatas=metadata)
        if self.collection.count() != len(rows):
            raise ValueError("Dense index final count mismatch")
        manifest["status"] = "complete"
        atomic_json(self.manifest_path, manifest)

    def search(self, query_embedding: np.ndarray, k: int = 10, *, embedding_identity=None) -> List[Dict]:
        if not isinstance(k, int) or k < 1:
            raise ValueError("k must be positive")
        manifest = self._manifest()
        if manifest is None:
            raise ValueError("Search requires a verified index manifest")
        identity = manifest["identity"]
        if embedding_identity != identity["embedding"]:
            raise ValueError("Query vector space identity mismatch, even when dimensions match")
        vector = np.asarray(query_embedding, dtype=np.float32)
        if vector.shape != (identity["dimension"],) or not np.isfinite(vector).all():
            raise ValueError("Invalid query embedding")
        count = self.collection.count()
        if not count:
            return []
        found = self.collection.query(query_embeddings=[vector.tolist()], n_results=min(k, count),
                                      include=["documents", "metadatas", "distances"])
        result = []
        for cid, text, meta, distance in zip(found["ids"][0], found["documents"][0], found["metadatas"][0], found["distances"][0]):
            payload = json.loads(meta["payload_json"])
            if signature(payload) != meta["payload_hash"] or payload["chunk_id"] != cid or payload["text"] != text:
                raise ValueError("Retrieved chunk payload integrity failure")
            result.append({**payload, "score": float(1.0 - distance)})
        return result

    def exists(self, expected_identity: dict | None = None) -> bool:
        manifest = self._manifest()
        return manifest is not None and (expected_identity is None or manifest["identity"] == expected_identity)
