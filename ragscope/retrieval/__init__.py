"""Public retrievers; backend dependencies load only when explicitly selected."""
from importlib import import_module

__all__ = ["DenseRetriever", "BM25Retriever", "HybridRetriever", "Reranker"]

_MODULES = {"DenseRetriever": "dense_retriever", "BM25Retriever": "bm25_retriever",
            "HybridRetriever": "hybrid_retriever", "Reranker": "reranker"}


def __getattr__(name):
    if name not in _MODULES:
        raise AttributeError(name)
    value = getattr(import_module(f"{__name__}.{_MODULES[name]}"), name)
    globals()[name] = value
    return value
