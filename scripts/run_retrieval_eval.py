"""
Retrieval evaluation script for RAGScope.

Loads eval_queries.jsonl, runs run_retrieval_benchmark over 4 configs
(Dense, BM25, Hybrid, Hybrid+Rerank) using the fixed-chunking indexes,
and saves results to reports/retrieval_results.csv.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Allow running as `python scripts/run_retrieval_eval.py` from project root
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from loguru import logger

from ragscope.evaluation.benchmark import run_retrieval_benchmark
from ragscope.indexing.embedding import EmbeddingService
from ragscope.retrieval.bm25_retriever import BM25Retriever
from ragscope.retrieval.dense_retriever import DenseRetriever
from ragscope.retrieval.hybrid_retriever import HybridRetriever
from ragscope.retrieval.reranker import Reranker

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

EVAL_QUERIES_PATH = _ROOT / "data" / "processed" / "eval_queries.jsonl"
REPORTS_DIR       = _ROOT / "reports"
OUTPUT_CSV        = REPORTS_DIR / "retrieval_results.csv"


def _load_eval_queries(path: Path) -> list:
    queries = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                queries.append(json.loads(line))
    logger.info(f"Loaded {len(queries)} eval queries from {path}")
    return queries


def main() -> None:
    # ── Validate inputs ────────────────────────────────────────────────────
    if not EVAL_QUERIES_PATH.exists():
        logger.error(
            f"Eval queries not found: {EVAL_QUERIES_PATH}\n"
            "Run `python scripts/ingest.py` first."
        )
        sys.exit(1)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load queries ───────────────────────────────────────────────────────
    eval_queries = _load_eval_queries(EVAL_QUERIES_PATH)

    # ── Build retrievers (fixed chunking) ──────────────────────────────────
    logger.info("Initialising retrievers …")

    emb_service = EmbeddingService()

    dense   = DenseRetriever(chunking_strategy="fixed", embedding_service=emb_service)
    bm25    = BM25Retriever(chunking_strategy="fixed")
    hybrid  = HybridRetriever(
        dense_retriever=dense,
        bm25_retriever=bm25,
        alpha=0.5,
    )
    reranker = Reranker()

    # ── Configs ────────────────────────────────────────────────────────────
    configs = [
        {
            "name":         "Dense",
            "retriever":    dense,
            "use_reranker": False,
        },
        {
            "name":         "BM25",
            "retriever":    bm25,
            "use_reranker": False,
        },
        {
            "name":         "Hybrid",
            "retriever":    hybrid,
            "use_reranker": False,
        },
        {
            "name":         "Hybrid+Rerank",
            "retriever":    hybrid,
            "use_reranker": True,
            "reranker":     reranker,
        },
    ]

    # ── Run benchmark ──────────────────────────────────────────────────────
    df = run_retrieval_benchmark(configs, eval_queries, n_queries=200, seed=42)

    # ── Save results ───────────────────────────────────────────────────────
    df.to_csv(OUTPUT_CSV, index=False)
    logger.info(f"Results saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
