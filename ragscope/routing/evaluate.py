"""Compare final answer quality; replay estimates and actual online time stay separate."""
from __future__ import annotations

import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np

from .core import ROUTES, RoutingRequest, proxy_quality


def calibration(probabilities, labels, bins=10):
    values = np.asarray(probabilities, dtype=float)
    if values.shape != (len(labels), len(ROUTES)) or not len(labels) or bins < 1:
        raise ValueError("Calibration requires aligned nonempty three-way probabilities")
    if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any() or not np.allclose(values.sum(1), 1, atol=1e-5):
        raise ValueError("Invalid calibration probabilities")
    actual = np.asarray([ROUTES.index(label) for label in labels])
    top, confidence = values.argmax(1), values.max(1)
    correctness = (top == actual).astype(float)
    bucket_ids = np.minimum((confidence * bins).astype(int), bins - 1)
    rows, ece = [], 0.0
    for index in range(bins):
        members = bucket_ids == index
        if members.any():
            count = int(members.sum())
            mean, accuracy = float(confidence[members].mean()), float(correctness[members].mean())
            ece += count / len(labels) * abs(mean - accuracy)
            rows.append({"bin": index, "n": count, "confidence": mean, "proxy_accuracy": accuracy})
    onehot = np.eye(len(ROUTES))[actual]
    return {"n": len(labels), "ece": float(ece), "brier": float(np.mean(np.sum((values - onehot) ** 2, axis=1))),
            "bins": rows, "target": "generation_F1_cost_proxy_label_not_absolute_optimal_route"}


def summarize(rows):
    if not rows:
        raise ValueError("No routing results")
    if len({row["latency_kind"] for row in rows}) != 1:
        raise ValueError("Actual and replay latency must be summarized separately")
    evidence = [row["evidence_coverage"] for row in rows if row.get("evidence_coverage") is not None]
    f1 = [row['answer_f1'] for row in rows if row.get('answer_f1') is not None]
    unanswerable = [row for row in rows if row.get('answerability') == 'unanswerable']
    answerable = [row for row in rows if row.get('answerability') != 'unanswerable']
    for row in rows:
        proxy_quality(row)
    return {"n": len(rows), "answer_f1": float(np.mean(f1)) if f1 else None,
            "answer_f1_n": len(f1), "unanswerable_n": len(unanswerable),
            "correct_abstention_rate": float(np.mean([row['correct_abstention'] for row in unanswerable])) if unanswerable else None,
            "evidence_coverage": float(np.mean(evidence)) if evidence else None,
            "evidence_n": len(evidence), "failure_rate": float(np.mean([bool(row.get("error")) for row in rows])),
            "wrong_abstain_rate": float(np.mean([bool(row.get("wrong_abstain")) for row in answerable])) if answerable else None,
            "fallback_rate": float(np.mean([bool(row.get("fallback")) for row in rows])),
            "p50_latency_s": float(np.median([row["latency_s"] for row in rows])),
            "p95_latency_s": float(np.quantile([row["latency_s"] for row in rows], .95)),
            "latency_kind": rows[0]["latency_kind"]}


def replay(router, queries, route_records, *, query_ids, comparison_config_hash):
    records = {}
    for record in route_records:
        if record["query_id"] in query_ids:
            if record.get("real_model") is not True or record.get("comparison_config_hash") != comparison_config_hash:
                raise ValueError("Replay needs exact real generation records")
            key = (record["query_id"], record["path"])
            if key in records:
                raise ValueError("Duplicate route record")
            records[key] = record
    if len(query_ids) != len(set(query_ids)):
        raise ValueError("Duplicate replay query IDs")
    query_map = {query["query_id"]: query for query in queries}
    rows = []
    for query_id in query_ids:
        if any((query_id, route) not in records for route in ROUTES):
            raise ValueError("Replay requires all frozen paths, including failed records")
        decision = router.choose(RoutingRequest.from_query(query_map[query_id]))
        record = records[(query_id, decision.route)]
        rows.append({"query_id": query_id, "selected_route": decision.route,
                     "answer_f1": record["answer_f1"],
                     "answerability": record.get('answerability', 'answerable'),
                     "correct_abstention": record.get('correct_abstention'),
                     "evidence_coverage": record.get("evidence_coverage"), "error": record.get("error"),
                     "wrong_abstain": record.get("wrong_abstain", False), "fallback": decision.fallback,
                     "latency_s": record["total_latency_s"] + decision.elapsed_s,
                     "latency_kind": "component_replay_estimate", "router_decision": decision.to_dict()})
    return rows, summarize(rows)


def online(router, queries, execute, output, *, model_residency="resident", minimum_n=30):
    """execute(request,route) performs the real path and includes any needed load."""
    if len(queries) < minimum_n:
        raise ValueError("Online routing validation requires at least the frozen minimum N")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with output.open("x", encoding="utf-8") as handle:
        for query in queries:
            request = RoutingRequest.from_query(query)
            started = perf_counter()
            decision = router.choose(request)
            record = execute(request, decision.route)
            elapsed = perf_counter() - started
            if record.get("real_model") is not True:
                raise ValueError("Online experiment cannot use a mock generator")
            row = {"query_id": query["query_id"], "selected_route": decision.route,
                   "answer_f1": record["answer_f1"],
                   "answerability": record.get('answerability', 'answerable'),
                   "correct_abstention": record.get('correct_abstention'),
                   "evidence_coverage": record.get("evidence_coverage"), "error": record.get("error"),
                   "wrong_abstain": record.get("wrong_abstain", False), "fallback": decision.fallback,
                   "latency_s": elapsed, "latency_kind": "actual_online", "model_residency": model_residency,
                   "router_decision": decision.to_dict(), "generation_record": record}
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            rows.append(row)
    return rows, summarize(rows)


def deployment_gate(router_dev, best_fixed_dev, *, online_n, quality_tolerance=.01, latency_reduction=.1):
    if router_dev["latency_kind"] != "actual_online" or best_fixed_dev["latency_kind"] != "actual_online" or online_n < 30:
        return {"deploy": False, "reason": "actual_online_dev_evidence_required", "default": "fixed"}
    quality_ok = (router_dev.get('answer_f1') is not None and best_fixed_dev.get('answer_f1') is not None
                  and router_dev["answer_f1"] >= best_fixed_dev["answer_f1"] - quality_tolerance)
    latency_ok = router_dev["p50_latency_s"] <= best_fixed_dev["p50_latency_s"] * (1 - latency_reduction)
    return {"deploy": bool(quality_ok and latency_ok), "quality_ok": quality_ok, "latency_ok": latency_ok,
            "default": "router" if quality_ok and latency_ok else "fixed",
            "note": "engineering_threshold_not_statistical_significance"}
