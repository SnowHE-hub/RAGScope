"""
End-to-end retrieval smoke test.

Builds tiny in-memory test indexes (no full ingest pipeline required),
then exercises all four retrieval strategies with a fixed query.

Run:
    python scripts/test_retrieval.py
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from loguru import logger

from ragscope.indexing.dense_index import DenseIndex
from ragscope.indexing.embedding import EmbeddingService
from ragscope.indexing.sparse_index import BM25Index
from ragscope.retrieval.bm25_retriever import BM25Retriever
from ragscope.retrieval.dense_retriever import DenseRetriever
from ragscope.retrieval.hybrid_retriever import HybridRetriever
from ragscope.retrieval.reranker import Reranker

# ---------------------------------------------------------------------------
# Tiny test corpus
# ---------------------------------------------------------------------------

_TEST_DOCS = [
    {
        "chunk_id": "d0_c0", "doc_id": "d0",
        "text": "Machine learning is a subset of artificial intelligence that enables "
                "computers to learn patterns from data without being explicitly programmed.",
        "n_tokens": 24, "position_in_doc": 0,
        "prev_chunk_id": None, "next_chunk_id": None, "metadata": "{}",
    },
    {
        "chunk_id": "d1_c0", "doc_id": "d1",
        "text": "Deep learning uses neural networks with many layers to automatically "
                "learn hierarchical representations from raw data.",
        "n_tokens": 20, "position_in_doc": 0,
        "prev_chunk_id": None, "next_chunk_id": None, "metadata": "{}",
    },
    {
        "chunk_id": "d2_c0", "doc_id": "d2",
        "text": "Supervised learning trains models on labelled examples so they can "
                "make accurate predictions on unseen data.",
        "n_tokens": 19, "position_in_doc": 0,
        "prev_chunk_id": None, "next_chunk_id": None, "metadata": "{}",
    },
    {
        "chunk_id": "d3_c0", "doc_id": "d3",
        "text": "Unsupervised learning discovers hidden patterns and structure in "
                "unlabelled data without any explicit supervision.",
        "n_tokens": 18, "position_in_doc": 0,
        "prev_chunk_id": None, "next_chunk_id": None, "metadata": "{}",
    },
    {
        "chunk_id": "d4_c0", "doc_id": "d4",
        "text": "Reinforcement learning trains agents to select actions in an "
                "environment so as to maximise a cumulative reward signal.",
        "n_tokens": 22, "position_in_doc": 0,
        "prev_chunk_id": None, "next_chunk_id": None, "metadata": "{}",
    },
    {
        "chunk_id": "d5_c0", "doc_id": "d5",
        "text": "The transformer architecture revolutionised natural language "
                "processing through self-attention mechanisms.",
        "n_tokens": 16, "position_in_doc": 0,
        "prev_chunk_id": None, "next_chunk_id": None, "metadata": "{}",
    },
]

QUERY = "what is machine learning"
K = 3


def _banner(title: str) -> None:
    print(f"\n{'='*55}")
    print(f"  {title}")
    print(f"{'='*55}")


def _show(results, label: str, score_key: str = "score") -> None:
    print(f"[{label}] top-{len(results)} for: {QUERY!r}")
    for i, r in enumerate(results, 1):
        print(f"  {i}. {r['chunk_id']}  {score_key}={r[score_key]:.6f}  {r['text'][:60]!r}")


def main() -> None:
    chunks_df = pd.DataFrame(_TEST_DOCS)

    # Temporary directories — cleaned up at end
    dense_dir = tempfile.mkdtemp(prefix="ragscope_dense_test_")
    bm25_tmp = tempfile.NamedTemporaryFile(
        prefix="ragscope_bm25_test_", suffix=".pkl", delete=False
    )
    bm25_tmp.close()
    bm25_path = bm25_tmp.name

    try:
        # ── Build test indexes ────────────────────────────────────────────
        _banner("Building test indexes")

        svc = EmbeddingService()

        dense_idx = DenseIndex(collection_name="test_ml", persist_dir=dense_dir)
        dense_idx.build(chunks_df, svc)
        logger.info(f"Dense index: {dense_idx.collection.count()} vectors, dim={svc.get_dimension()}")

        bm25_idx = BM25Index(index_path=bm25_path)
        bm25_idx.build(chunks_df)
        logger.info(f"BM25 index: {len(bm25_idx.chunks)} chunks")

        # ── Instantiate retrievers ────────────────────────────────────────
        dense_r  = DenseRetriever(embedding_service=svc, persist_dir=dense_dir,
                                  collection_name="test_ml")
        bm25_r   = BM25Retriever(index_path=bm25_path)
        hybrid_r = HybridRetriever(dense_r, bm25_r, alpha=0.5)

        # ── 1. DenseRetriever ─────────────────────────────────────────────
        _banner("1. DenseRetriever")
        dense_results = dense_r.retrieve(QUERY, k=K)
        _show(dense_results, "Dense", "score")
        assert len(dense_results) == K, f"Expected {K} results, got {len(dense_results)}"
        assert all("chunk_id" in r and "score" in r and "text" in r for r in dense_results)
        print("  [Dense assertions: PASS]")

        # ── 2. BM25Retriever ──────────────────────────────────────────────
        _banner("2. BM25Retriever")
        bm25_results = bm25_r.retrieve(QUERY, k=K)
        _show(bm25_results, "BM25", "score")
        assert len(bm25_results) > 0, "BM25 returned no results"
        assert all("chunk_id" in r and "score" in r and "text" in r for r in bm25_results)
        # BM25 should find d0 ("machine learning") as top result
        assert bm25_results[0]["chunk_id"] == "d0_c0", \
            f"Expected d0_c0 as BM25 top-1, got {bm25_results[0]['chunk_id']}"
        print("  [BM25 assertions: PASS]")

        # ── 3. HybridRetriever (RRF) ──────────────────────────────────────
        _banner("3. HybridRetriever (RRF, alpha=0.5)")
        hybrid_results = hybrid_r.retrieve(QUERY, k=K)
        _show(hybrid_results, "Hybrid", "score")
        assert len(hybrid_results) == K
        # Scores must be strictly decreasing
        scores = [r["score"] for r in hybrid_results]
        assert scores == sorted(scores, reverse=True), "Hybrid results not sorted by score"
        print("  [Hybrid assertions: PASS]")

        # alpha edge cases
        pure_dense  = HybridRetriever(dense_r, bm25_r, alpha=1.0).retrieve(QUERY, k=K)
        pure_sparse = HybridRetriever(dense_r, bm25_r, alpha=0.0).retrieve(QUERY, k=K)
        assert len(pure_dense) == K and len(pure_sparse) == K
        print("  [Alpha edge-cases (0.0 / 1.0): PASS]")

        # ── 4. Reranker ───────────────────────────────────────────────────
        _banner("4. Reranker (BAAI/bge-reranker-base)")
        candidates = dense_r.retrieve(QUERY, k=6)
        reranker = Reranker(model_name="BAAI/bge-reranker-base")
        reranked = reranker.rerank(QUERY, candidates, top_k=K)
        _show(reranked, "Reranker", "rerank_score")
        assert len(reranked) == K
        assert all("rerank_score" in r for r in reranked)
        rs = [r["rerank_score"] for r in reranked]
        assert rs == sorted(rs, reverse=True), "Reranked results not sorted"
        # d0 (machine learning definition) should be most relevant
        assert reranked[0]["chunk_id"] == "d0_c0", \
            f"Expected d0_c0 as reranker top-1, got {reranked[0]['chunk_id']}"
        print("  [Reranker assertions: PASS]")

        # ── Summary ───────────────────────────────────────────────────────
        _banner("All checks PASSED")

    finally:
        shutil.rmtree(dense_dir, ignore_errors=True)
        Path(bm25_path).unlink(missing_ok=True)
        logger.info("Temp indexes cleaned up.")


if __name__ == "__main__":
    main()
