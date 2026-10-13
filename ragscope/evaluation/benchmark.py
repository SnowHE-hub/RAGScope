"""Legacy retrieval benchmark with explicit failure denominators and row audit."""
from __future__ import annotations

import json
import random
import time
from pathlib import Path
from statistics import mean

from ragscope.evaluation.retrieval_metrics import hit_rate_at_k, mrr, ndcg_at_k, recall_at_k

_COLS = ["Config", "Recall@5", "Recall@10", "HitRate@5", "MRR", "NDCG@10", "Latency(ms)"]


def run_retrieval_benchmark(configs, eval_queries, n_queries=500, seed=42, *, output_path=None):
    """Return summaries plus df.attrs['records']; retain every frozen query.

    k denotes returned chunks; duplicate documents never earn repeated relevance.
    Reranking retrieves 50 candidates and returns 10 by default. Failures score
    zero and stay in n_total, including all-failed configurations.
    """
    import pandas as pd
    if n_queries < 0:
        raise ValueError("n_queries must be nonnegative")
    rng = random.Random(seed)
    sampled = rng.sample(eval_queries, min(n_queries, len(eval_queries)))
    records, rows = [], []
    output = None
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        output = Path(output_path).open("x", encoding="utf-8")
    try:
        for cfg in configs:
            name = cfg["name"]
            init_error = None
            try:
                retriever = cfg["retriever"] if "retriever" in cfg else cfg["retriever_class"](**cfg.get("retriever_kwargs", {}))
                reranker = cfg.get("reranker") if cfg.get("use_reranker") else None
                if cfg.get("use_reranker") and reranker is None:
                    raise ValueError("use_reranker requires a ready reranker")
            except Exception as exc:
                init_error = {"kind": "initialization_error", "message": str(exc)}
            per_query = []
            for position, query in enumerate(sampled):
                started, error, retrieved = time.perf_counter(), init_error, []
                if error is None:
                    try:
                        depth = cfg.get("candidate_k", 50) if reranker else 10
                        retrieved = retriever.retrieve(query["question"], k=depth)
                        if reranker and retrieved:
                            retrieved = reranker.rerank(query["question"], retrieved, top_k=10)
                    except Exception as exc:
                        error = {"kind": "retrieval_error", "message": str(exc)}
                        retrieved = []
                latency = (time.perf_counter()-started)*1000
                docs = [chunk["doc_id"] for chunk in retrieved]
                relevant = set(query.get("relevant_doc_ids") or [query["gold_doc_id"]])
                scores = {"Recall@5": recall_at_k(docs, relevant, 5), "Recall@10": recall_at_k(docs, relevant, 10),
                          "HitRate@5": hit_rate_at_k(docs, relevant, 5), "MRR": mrr(docs, relevant),
                          "NDCG@10": ndcg_at_k(docs, relevant, 10)}
                record = {"config": name, "query_id": query.get("query_id", str(position)),
                          "question": query["question"], "retrieved_ids": docs, "error": error,
                          "latency_ms": latency, "scores": scores}
                per_query.append(record)
                records.append(record)
                if output:
                    output.write(json.dumps(record, ensure_ascii=False, allow_nan=False)+"\n")
                    output.flush()
            success = [r for r in per_query if r["error"] is None]
            summary = {"Config": name, **{metric: mean(r["scores"][metric] for r in per_query) if per_query else None
                                           for metric in _COLS[1:-1]},
                       "Latency(ms)": mean(r["latency_ms"] for r in per_query) if per_query else None,
                       "n_total": len(per_query), "n_success": len(success), "n_failed": len(per_query)-len(success),
                       "Recall@10_success_conditional": mean(r["scores"]["Recall@10"] for r in success) if success else None}
            rows.append(summary)
    finally:
        if output:
            output.close()
    frame = pd.DataFrame(rows)
    frame.attrs["records"] = records
    return frame
