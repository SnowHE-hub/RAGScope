from copy import deepcopy
import pytest

from ragscope.routing.core import ROUTES
from ragscope.routing.laya_report import study


def rows():
    output = []
    for qid in ("one", "two"):
        for index in range(3):
            probs = dict(zip(ROUTES, (.1, .2, .7) if index != 1 else (.6, .2, .2)))
            route = max(probs, key=probs.get)
            output.append({"query_id": qid, "order_index": index, "decision": {
                "probabilities": probs, "fallback": False, "route": route, "metadata": {
                    "raw_decision": {"answers": {"retrieval_path": {"choice": route}}}}}})
    return output


def test_permutations_count_distribution_changes_without_selecting_order():
    report = study(rows(), {"one": "hybrid_rerank", "two": "bm25"}, ["one", "two"])
    assert report["n_decisions"] == 6
    assert report["queries_with_selected_route_change"] == 2
    assert report["queries_with_raw_choice_change_among_all_valid"] == 2
    assert report["pairwise_probability_l1_n"] == 6
    assert report["default_order_index"] == 0
    assert all(order["calibration"]["n"] == 2 for order in report["orders"])


def test_invalid_probabilities_are_counted_without_imputation():
    data = rows()
    data[0]["decision"] = {"fallback": True, "route": "hybrid", "probabilities": {}, "metadata": {"error": {"type": "X"}}}
    result = study(data, {"one": "hybrid", "two": "bm25"}, ["one", "two"])
    assert result["n_decisions"] == 6
    assert result["orders"][0]["n_total"] == 2
    assert result["orders"][0]["n_invalid_output"] == 1
    assert result["orders"][0]["calibration"]["n"] == 1
    assert result["queries_with_all_three_valid_probabilities"] == 1
    assert result["pairwise_probability_l1_n"] == 3


@pytest.mark.parametrize("mutate", ["missing", "duplicate", "foreign", "labels", "query_ids"])
def test_incomplete_study_rejected(mutate):
    data, labels, ids = rows(), {"one": "bm25", "two": "hybrid"}, ["one", "two"]
    if mutate == "missing": data.pop()
    if mutate == "duplicate": data[-1] = deepcopy(data[0])
    if mutate == "foreign": data[-1]["query_id"] = "foreign"
    if mutate == "labels": labels.pop("two")
    if mutate == "query_ids": ids.append("one")
    with pytest.raises(ValueError): study(data, labels, ids)
