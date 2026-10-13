import math

import pytest

from scripts.summarize_retrieval_auxiliary_v2 import rank_metrics


def test_auxiliary_binary_ranks_use_full_index_ideal_not_retrieved_pool():
    actual = rank_metrics(["noise", "a", "b"], {"a", "b", "missing"}, answerability="answerable")
    assert actual["mrr_at_10"] == .5
    assert actual["ndcg_at_10"] == pytest.approx((1 / math.log2(3) + .5) / (1 + 1 / math.log2(3) + .5))
    assert rank_metrics(["noise"], {"outside_top10"}, answerability="answerable") == {"mrr_at_10": 0., "ndcg_at_10": 0.}


def test_auxiliary_na_failures_and_duplicate_contract():
    assert rank_metrics([], set(), answerability="unanswerable", failed=True) == {"mrr_at_10": None, "ndcg_at_10": None}
    assert rank_metrics(["a"], {"a"}, answerability="answerable", failed=True) == {"mrr_at_10": 0., "ndcg_at_10": 0.}
    with pytest.raises(ValueError, match="repeated"):
        rank_metrics(["a", "a"], {"a"}, answerability="answerable")
    with pytest.raises(ValueError, match="no relevant"):
        rank_metrics([], set(), answerability="answerable")
