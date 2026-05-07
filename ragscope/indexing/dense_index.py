"""
DenseIndex: ChromaDB-backed vector store for chunk retrieval.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm

from ragscope.indexing.embedding import EmbeddingService


class DenseIndex:
    def __init__(
        self,
        collection_name: str,
        persist_dir: str = "data/indexes/chroma",
    ) -> None:
        import chromadb

        self.collection_name = collection_name
        self.persist_dir = persist_dir
        Path(persist_dir).mkdir(parents=True, exist_ok=True)

        self._client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(
        self,
        chunks_df: pd.DataFrame,
        embedding_service: EmbeddingService,
        batch_size: int = 500,
    ) -> None:
        """Embed all chunks and upsert into the ChromaDB collection."""
        if self.collection.count() > 0:
            logger.warning(
                f"Collection '{self.collection_name}' already has "
                f"{self.collection.count():,} vectors — skipping build."
            )
            return

        n = len(chunks_df)
        logger.info(f"Building DenseIndex '{self.collection_name}' — {n:,} chunks")

        for start in tqdm(range(0, n, batch_size), desc="DenseIndex build", unit="batch"):
            batch = chunks_df.iloc[start : start + batch_size]

            texts = batch["text"].tolist()
            embeddings = embedding_service.embed(texts, show_progress=False)

            ids = batch["chunk_id"].tolist()
            metadatas = [
                {
                    "doc_id": str(doc_id),
                    "chunk_id": str(chunk_id),
                    "n_tokens": int(n_tok),
                    "position_in_doc": int(pos),
                    "source": str(doc_id).split("_")[0],
                }
                for doc_id, chunk_id, n_tok, pos in zip(
                    batch["doc_id"],
                    batch["chunk_id"],
                    batch["n_tokens"],
                    batch["position_in_doc"],
                )
            ]

            self.collection.add(
                ids=ids,
                embeddings=embeddings.tolist(),
                documents=texts,
                metadatas=metadatas,
            )

        logger.info(
            f"DenseIndex '{self.collection_name}' built — "
            f"{self.collection.count():,} vectors stored"
        )

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query_embedding: np.ndarray, k: int = 10) -> List[Dict]:
        """Return top-k chunks by cosine similarity.

        Each result: {chunk_id, doc_id, text, score, metadata}
        Score is cosine similarity in [−1, 1]; higher = more similar.
        """
        count = self.collection.count()
        if count == 0:
            return []
        k = min(k, count)

        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )

        return [
            {
                "chunk_id": chunk_id,
                "doc_id": meta.get("doc_id", ""),
                "text": doc,
                "score": float(1.0 - dist),   # cosine distance → similarity
                "metadata": meta,
            }
            for chunk_id, doc, meta, dist in zip(
                results["ids"][0],
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            )
        ]

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def exists(self) -> bool:
        """True if the collection has been populated with at least one vector."""
        return self.collection.count() > 0
