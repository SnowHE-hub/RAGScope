"""
Retrieval benchmark runner for RAGScope.

Evaluates multiple retrieval configurations against a set of labelled queries
and prints a formatted comparison table.
"""

from __future__ import annotations

import random
import time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm

from ragscope.evaluation.retrieval_metrics import (
    hit_rate_at_k,
    mrr,
    ndcg_at_k,
    recall_at_k,
)

# ---------------------------------------------------------------------------
# Config schema (duck-typed)
# ---------------------------------------------------------------------------
# Each config dict must contain:
#   name             str          — display name in the table
#   retriever        BaseRetriever — pre-built retriever instance
#   use_reranker     bool
#   reranker         Reranker | None   — required when use_reranker=True
#
# Alternatively, lazy construction is supported via:
#   retriever_class  type
#   retriever_kwargs dict

# ---------------------------------------------------------------------------
# Table printing
# ---------------------------------------------------------------------------

_COL_WIDTHS = {
    "Config":      22,
    "Recall@5":     8,
    "Recall@10":    9,
    "HitRate@5":    9,
    "MRR":          6,
    "NDCG@10":      7,
    "Latency(ms)": 11,
}
_COLS = list(_COL_WIDTHS.keys())


def _fmt_row(values: Dict[str, Any]) -> str:
    parts = []
    for col, w in _COL_WIDTHS.items():
        v = values[col]
        if col == "Config":
            parts.append(f"{str(v):<{w}}")
        elif col == "Latency(ms)":
            parts.append(f"{v:>{w}.1f}")
        else:
            parts.append(f"{v:>{w}.3f}")
    return " | ".join(parts)


def _print_table(df: pd.DataFrame) -> None:
    header = " | ".join(
        f"{col:<{w}}" if col == "Config" else f"{col:>{w}}"
        for col, w in _COL_WIDTHS.items()
    )
    sep = "-+-".join("-" * w for w in _COL_WIDTHS.values())
    print()
    print(header)
    print(sep)
    for _, row in df.iterrows():
        print(_fmt_row(row.to_dict()))
    print(sep)


# ---------------------------------------------------------------------------
# Core benchmark function
# ---------------------------------------------------------------------------

def run_retrieval_benchmark(
    configs: List[Dict],
    eval_queries: List[Dict],
    n_queries: int = 500,
    seed: int = 42,
) -> pd.DataFrame:
    """Evaluate multiple retrieval configurations on a sample of eval queries.

    Parameters
    ----------
    configs:
        List of config dicts.  Each must include ``name``, ``retriever`` (or
        ``retriever_class`` + ``retriever_kwargs``), ``use_reranker``, and
        optionally ``reranker``.
    eval_queries:
        List of query dicts with keys ``question`` and ``gold_doc_id``.
    n_queries:
        Number of queries to sample (uses all if smaller than dataset).
    seed:
        Random seed for reproducible sampling.

    Returns
    -------
    pd.DataFrame
        One row per config, columns: Config, Recall@5, Recall@10,
        HitRate@5, MRR, NDCG@10, Latency(ms).
    """
    random.seed(seed)
    sampled = (
        random.sample(eval_queries, n_queries)
        if len(eval_queries) > n_queries
        else eval_queries
    )
    logger.info(f"Benchmark: {len(configs)} configs × {len(sampled)} queries")

    rows: List[Dict] = []

    for cfg in configs:
        name = cfg["name"]

        # Instantiate retriever (pre-built or lazy)
        if "retriever" in cfg:
            retriever = cfg["retriever"]
        else:
            retriever = cfg["retriever_class"](**cfg.get("retriever_kwargs", {}))

        use_reranker: bool = cfg.get("use_reranker", False)
        reranker     = cfg.get("reranker") if use_reranker else None

        per_query: List[Dict] = []
        latencies: List[float] = []
        errors = 0

        for q in tqdm(sampled, desc=f"  {name}", unit="q", leave=False):
            question    = q["question"]
            gold_doc_id = q["gold_doc_id"]

            try:
                t0 = time.perf_counter()
                retrieved = retriever.retrieve(question, k=10)
                if reranker and retrieved:
                    retrieved = reranker.rerank(question, retrieved, top_k=10)
                latency_ms = (time.perf_counter() - t0) * 1000
            except Exception as exc:
                logger.warning(f"[{name}] query failed: {exc}")
                errors += 1
                continue

            doc_ids     = [r["doc_id"] for r in retrieved]
            relevant    = {gold_doc_id}
            latencies.append(latency_ms)

            per_query.append(
                {
                    "recall@5":  recall_at_k(doc_ids, relevant, k=5),
                    "recall@10": recall_at_k(doc_ids, relevant, k=10),
                    "hitrate@5": hit_rate_at_k(doc_ids, relevant, k=5),
                    "mrr":       mrr(doc_ids, relevant),
                    "ndcg@10":   ndcg_at_k(doc_ids, relevant, k=10),
                }
            )

        if not per_query:
            logger.error(f"[{name}] no successful queries — skipping")
            continue

        if errors:
            logger.warning(f"[{name}] {errors}/{len(sampled)} queries failed")

        avg = {k: float(np.mean([m[k] for m in per_query])) for k in per_query[0]}
        rows.append(
            {
                "Config":      name,
                "Recall@5":    avg["recall@5"],
                "Recall@10":   avg["recall@10"],
                "HitRate@5":   avg["hitrate@5"],
                "MRR":         avg["mrr"],
                "NDCG@10":     avg["ndcg@10"],
                "Latency(ms)": float(np.mean(latencies)),
            }
        )
        logger.info(
            f"[{name}] Recall@5={avg['recall@5']:.3f} Recall@10={avg['recall@10']:.3f} "
            f"HitRate@5={avg['hitrate@5']:.3f} MRR={avg['mrr']:.3f} "
            f"NDCG@10={avg['ndcg@10']:.3f} Latency={np.mean(latencies):.1f}ms"
        )

    df = pd.DataFrame(rows, columns=_COLS)
    _print_table(df)
    return df
