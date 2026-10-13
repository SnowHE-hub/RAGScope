from ragscope.evaluation.retrieval_metrics import (
    recall_at_k,
    precision_at_k,
    hit_rate_at_k,
    mrr,
    ndcg_at_k,
)
from ragscope.evaluation.generation_metrics import (
    exact_match,
    token_f1,
    rouge_l,
)
from ragscope.evaluation.evidence import evidence_metrics, context_statistics
from ragscope.evaluation.v2 import score_generation, summarize_records, cluster_bootstrap, paired_cluster_bootstrap


def run_retrieval_benchmark(*args, **kwargs):
    from ragscope.evaluation.benchmark import run_retrieval_benchmark as run
    return run(*args, **kwargs)

__all__ = [
    "recall_at_k",
    "precision_at_k",
    "hit_rate_at_k",
    "mrr",
    "ndcg_at_k",
    "exact_match",
    "token_f1",
    "rouge_l",
    "run_retrieval_benchmark",
    "evidence_metrics",
    "context_statistics",
    "score_generation",
    "summarize_records",
    "cluster_bootstrap",
    "paired_cluster_bootstrap",
]
