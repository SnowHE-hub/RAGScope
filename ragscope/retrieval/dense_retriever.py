from __future__ import annotations

from typing import Dict, List, Optional

from ragscope.indexing.dense_index import DenseIndex
from ragscope.indexing.embedding import EmbeddingService
from ragscope.retrieval.base import BaseRetriever

_COLLECTION_MAP: Dict[str, str] = {
    "fixed":    "ragscope_fixed",
    "sentence": "ragscope_sentence",
    "semantic": "ragscope_semantic",
}


class DenseRetriever(BaseRetriever):
    """Vector-similarity retrieval backed by ChromaDB."""

    def __init__(
        self,
        chunking_strategy: str = "fixed",
        embedding_service: Optional[EmbeddingService] = None,
        persist_dir: str = "data/indexes/chroma",
        collection_name: Optional[str] = None,
    ) -> None:
        self.embedding_service = embedding_service or EmbeddingService()
        cname = collection_name or _COLLECTION_MAP.get(chunking_strategy, chunking_strategy)
        self.index = DenseIndex(collection_name=cname, persist_dir=persist_dir)

    def retrieve(self, query: str, k: int = 5) -> List[Dict]:
        query_emb = self.embedding_service.embed_query(query)
        return self.index.search(query_emb, k=k, embedding_identity=self.embedding_service.identity)
