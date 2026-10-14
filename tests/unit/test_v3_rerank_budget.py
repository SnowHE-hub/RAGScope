import pytest
from ragscope.v3.rerank_budget import RerankBudget


def test_pair_reservations_count_failure_and_reject_duplicate_and_over_limit(tmp_path):
    budget=RerankBudget(tmp_path)
    budget.reserve('first_failed',200000,{'model':'fixture'})
    with pytest.raises(ValueError,match='already reserved'):
        budget.reserve('first_failed',1,{'model':'fixture'})
    with pytest.raises(RuntimeError,match='exhausted'):
        budget.reserve('next',1,{'model':'fixture'})
