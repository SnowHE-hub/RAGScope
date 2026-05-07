from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List


class BaseRetriever(ABC):
    """Common interface for all retrieval strategies."""

    @abstractmethod
    def retrieve(self, query: str, k: int = 5) -> List[Dict]:
        """Return top-k results for *query*.

        Each result is a dict with at minimum:
            chunk_id  (str)
            doc_id    (str)
            text      (str)
            score     (float)  — higher is more relevant
            metadata  (dict)
        """
        ...
