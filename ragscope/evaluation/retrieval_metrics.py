"""
Retrieval quality metrics for RAGScope evaluation.

All functions treat *retrieved_ids* as an **ordered** list (rank matters) and
*relevant_ids* as an **unordered** set of ground-truth identifiers.
"""

from __future__ import annotations

import math
from typing import List, Set, Union

_IDs = Union[List[str], List[int]]
_Set = Union[Set[str], Set[int], list]


def recall_at_k(retrieved_ids: _IDs, relevant_ids: _Set, k: int) -> float:
    """Fraction of relevant items found in the top-k retrieved results.

    Formula::

        Recall@k = |relevant ∩ retrieved[:k]| / |relevant|

    Returns 0.0 when *relevant_ids* is empty.
    """
    if not relevant_ids or k <= 0:
        return 0.0
    rel_set = set(relevant_ids)
    top_k   = set(retrieved_ids[:k])
    return len(top_k & rel_set) / len(rel_set)


def precision_at_k(retrieved_ids: _IDs, relevant_ids: _Set, k: int) -> float:
    """Fraction of the top-k retrieved results that are relevant.

    Formula::

        Precision@k = |relevant ∩ retrieved[:k]| / k

    Returns 0.0 when k == 0 or *retrieved_ids* is empty.
    """
    if k <= 0 or not retrieved_ids:
        return 0.0
    rel_set  = set(relevant_ids)
    top_k    = retrieved_ids[:k]
    n_rel    = len(set(top_k) & rel_set)
    return n_rel / k


def hit_rate_at_k(retrieved_ids: _IDs, relevant_ids: _Set, k: int) -> float:
    """Binary indicator: 1 if at least one relevant item appears in top-k.

    Formula::

        HitRate@k = 1  if |relevant ∩ retrieved[:k]| > 0  else 0

    Equal to Recall@k only when there is one relevant item per query.
    """
    if k <= 0:
        return 0.0
    rel_set = set(relevant_ids)
    top_k   = set(retrieved_ids[:k])
    return 1.0 if top_k & rel_set else 0.0


def mrr(retrieved_ids: _IDs, relevant_ids: _Set) -> float:
    """Reciprocal rank of the first relevant item in the retrieved list.

    Formula::

        MRR = 1 / rank_of_first_relevant

    where rank is 1-indexed.  Returns 0.0 if no relevant item is found.
    """
    rel_set = set(relevant_ids)
    for rank, rid in enumerate(retrieved_ids, start=1):
        if rid in rel_set:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_ids: _IDs, relevant_ids: _Set, k: int) -> float:
    """Normalised Discounted Cumulative Gain at k (binary relevance).

    Formula::

        DCG@k  = Σ rel_i / log2(i+1)       i = 1..k
        IDCG@k = Σ 1 / log2(i+1)           i = 1..min(k, |relevant|)
        NDCG@k = DCG@k / IDCG@k

    *rel_i* is 1 if retrieved_ids[i-1] ∈ relevant_ids else 0.
    Returns 0.0 when IDCG is 0 (no relevant items or k == 0).
    """
    if k <= 0 or not relevant_ids:
        return 0.0

    rel_set = set(relevant_ids)

    seen, dcg = set(), 0.0
    for rank, rid in enumerate(retrieved_ids[:k], start=1):
        if rid in rel_set and rid not in seen:
            dcg += 1.0 / math.log2(rank + 1)
        seen.add(rid)

    ideal_len = min(k, len(rel_set))
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_len + 1))

    return dcg / idcg if idcg > 0.0 else 0.0
