from __future__ import annotations

from typing import Dict, List, Optional

from ragscope.indexing.sparse_index import BM25Index
from ragscope.retrieval.base import BaseRetriever

_BM25_PATH_MAP: Dict[str, str] = {
    "fixed":    "data/indexes/bm25_fixed.pkl",
    "sentence": "data/indexes/bm25_sentence.pkl",
    "semantic": "data/indexes/bm25_semantic.pkl",
}


class BM25Retriever(BaseRetriever):
    """Sparse BM25 retrieval."""

    def __init__(
        self,
        chunking_strategy: str = "fixed",
        index_path: Optional[str] = None,
    ) -> None:
        path = index_path or _BM25_PATH_MAP[chunking_strategy]
        self.index = BM25Index.load(path)

    def retrieve(self, query: str, k: int = 5) -> List[Dict]:
        return self.index.search(query, k=k)
