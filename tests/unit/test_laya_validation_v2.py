"""Synthetic contract tests only; these fixtures are never real Laya results."""
from copy import deepcopy
from dataclasses import asdict
import pytest

from ragscope.indexing.identity import signature
from ragscope.routing.core import ROUTES, RoutingRequest
from ragscope.routing.laya_adapter import PERMUTATIONS
from ragscope.routing.laya_validation import validate_decision, validate_population


def decision(identity, order=ROUTES, *, low=False, error=False):
    probs = dict(zip(ROUTES, (.3, .4, .3) if low else (.1, .2, .7)))
    metadata = {"identity": identity, "candidate_order": list(order), "raw_decision": {
        "usage": {"truncated": False}, "answers": {"retrieval_path": {
            "probabilities": probs, "choice": "hybrid" if low else "hybrid_rerank"}}}}
    row = {"route": "hybrid" if low else "hybrid_rerank", "elapsed_s": .1, "metadata": metadata,
           "reason": "laya_low_confidence" if low else "laya_three_way", "fallback": low, "probabilities": probs}
    if error:
        metadata["error"] = {"type": "RuntimeError", "message": "retained model failure"}
        row.update(route="hybrid", reason="laya_invalid_or_unavailable", fallback=True, probabilities={})
    return row


def seal(row):
    row["record_hash"] = signature({k: v for k, v in row.items() if k != "record_hash"})
    return row


def population():
    inputs = {"expected_queries": 360, "expected_decisions": 1080, "no_quality_labels_read": True,
              "seed": 42, "min_confidence": .5, "orders": [list(o) for o in PERMUTATIONS], "populations": {}}
    for split, count in (("train", 200), ("dev", 60), ("test", 100)):
        visible = [{"query_id": f"{split}{i}", "split": split, "question": f"Question {i}?",
                    "scope": {"kind": "document", "doc_ids": ["d"]}, "metadata": {}}
                   for i in range(count)]
        inputs["populations"][split] = {"query_ids": [q["query_id"] for q in visible], "visible_queries": visible}
    inputs["input_hash"] = signature(inputs)
    identity = {"unit_fixture": True}
    config = {"input_hash": inputs["input_hash"], "expected_decisions": 1080, "warmup_queries": 3,
              "router_identity": identity}
    rows = []
    for split, pop in inputs["populations"].items():
        for query in pop["visible_queries"]:
            for index, order in enumerate(PERMUTATIONS):
                rows.append(seal({"split": split, "query_id": query["query_id"], "order_index": index,
                    "order": list(order), "record_id": f"{split}:{query['query_id']}:{index}",
                    "visible_input_hash": signature(asdict(RoutingRequest.from_query(query))),
                    "real_model": True, "model_identity": identity, "config_hash": signature(config),
                    "decision": decision(identity, order)}))
    warmups = [seal({"record_id": qid, "real_model": True, "model_identity": identity,
                    "config_hash": signature({**config, "purpose": "excluded_warmup"}), "decision": decision(identity)})
               for qid in inputs["populations"]["dev"]["query_ids"][:3]]
    return inputs, config, rows, warmups


def test_complete_population_keeps_failed_and_low_confidence_denominators():
    inputs, config, rows, warmups = population()
    rows[0]["decision"] = decision(config["router_identity"], error=True)
    rows[1]["decision"] = decision(config["router_identity"], PERMUTATIONS[1], low=True)
    seal(rows[0]); seal(rows[1])
    result = validate_population(inputs, config, rows, warmups)
    assert result == {"n_decisions": 1080, "n_queries": 360, "warmups_excluded": 3,
                      "errors": 1, "fallbacks": 2, "queries_with_order_dependent_selected_route": 1}


@pytest.mark.parametrize("change", ["missing", "duplicate", "wrong_query", "order", "visible", "hash",
                                    "config", "model", "mock", "warmup", "warmup_error", "input"])
def test_population_fail_closed(change):
    inputs, config, rows, warmups = population()
    if change == "missing": rows.pop()
    elif change == "duplicate": rows[-1] = deepcopy(rows[0])
    elif change == "wrong_query": rows[-1]["query_id"] = "foreign"
    elif change == "order": rows[0]["order"] = list(PERMUTATIONS[1])
    elif change == "visible": rows[0]["visible_input_hash"] = "wrong"
    elif change == "hash": rows[0].pop("record_hash")
    elif change == "config": rows[0]["config_hash"] = "wrong"
    elif change == "model": rows[0]["model_identity"] = {"other": True}
    elif change == "mock": rows[0]["real_model"] = False
    elif change == "warmup": warmups[-1]["record_id"] = "foreign"
    elif change == "warmup_error":
        warmups[0]["decision"] = decision(config["router_identity"], error=True)
        seal(warmups[0])
    elif change == "input": inputs["seed"] = 7
    if change in {"config", "model", "mock", "order", "visible"}: seal(rows[0])
    with pytest.raises(ValueError): validate_population(inputs, config, rows, warmups)


@pytest.mark.parametrize("change", ["negative_time", "nan", "wrong_order", "truncated", "collapsed",
    "probability_sum", "nan_probability", "normalized", "wrong_choice", "low_confidence", "error_fallback"])
def test_decision_semantic_corruption(change):
    identity = {"test": True}
    row = decision(identity)
    if change == "negative_time": row["elapsed_s"] = -1
    elif change == "nan": row["elapsed_s"] = float("nan")
    elif change == "wrong_order": row["metadata"]["candidate_order"] = list(PERMUTATIONS[1])
    elif change == "truncated": row["metadata"]["raw_decision"]["usage"]["truncated"] = True
    elif change == "collapsed": row["metadata"]["raw_decision"]["usage"]["options"] = {"retrieval_path": {"distinct": 2, "total": 3}}
    elif change == "probability_sum": row["metadata"]["raw_decision"]["answers"]["retrieval_path"]["probabilities"]["bm25"] = .6
    elif change == "nan_probability": row["metadata"]["raw_decision"]["answers"]["retrieval_path"]["probabilities"]["bm25"] = float("nan")
    elif change == "normalized": row["probabilities"] = {"bm25": .7, "hybrid": .2, "hybrid_rerank": .1}
    elif change == "wrong_choice": row["metadata"]["raw_decision"]["answers"]["retrieval_path"]["choice"] = "bm25"
    elif change == "low_confidence": row = decision(identity, low=True); row["fallback"] = False
    elif change == "error_fallback": row = decision(identity, error=True); row["route"] = "bm25"
    with pytest.raises(ValueError): validate_decision(row, identity, ROUTES)
