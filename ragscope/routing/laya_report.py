"""Post-freeze Laya permutation and calibration diagnostics; no model calls."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from ragscope.experiments.ledger import read_records
from ragscope.indexing.identity import artifact, signature, verify_artifacts
from .campaign import (REQUIRED_ROUTERS, freeze_routing, immutable_json, load_route_records,
                       validate_decisions, evaluate_decisions)
from .core import ROUTES, build_proxy_labels
from .evaluate import calibration, summarize
from .laya_adapter import PERMUTATIONS
from .laya_campaign import decisions_to_protocol


def study(rows, labels, query_ids):
    """Fixed three-order diagnostics. Missing probabilities remain explicitly counted."""
    if len(query_ids) != len(set(query_ids)) or set(labels) != set(query_ids):
        raise ValueError("Calibration labels must align with the entire frozen population")
    wanted = {(qid, index) for qid in query_ids for index in range(3)}
    by_key = {(row["query_id"], row["order_index"]): row for row in rows}
    if len(rows) != len(by_key) or set(by_key) != wanted:
        raise ValueError("Permutation population is incomplete or duplicated")
    orders = []
    for index, order in enumerate(PERMUTATIONS):
        selected = [by_key[(qid, index)] for qid in query_ids]
        valid = [row for row in selected if not row["decision"]["metadata"].get("error")]
        probabilities = [[row["decision"]["probabilities"][r] for r in ROUTES] for row in valid]
        result = calibration(probabilities, [labels[row["query_id"]] for row in valid]) if valid else None
        if result is not None:
            result["target"] = "generation_quality_cost_proxy;answerable_F1_or_unanswerable_correct_abstention"
        orders.append({"order_index": index, "order": list(order), "n_total": len(selected),
            "n_probabilities_available": len(valid), "n_invalid_output": len(selected) - len(valid),
            "fallbacks": sum(row["decision"]["fallback"] for row in selected),
            "selected_route_counts": dict(Counter(row["decision"]["route"] for row in selected)),
            "raw_choice_counts": dict(Counter(row["decision"]["metadata"]["raw_decision"]["answers"]["retrieval_path"]["choice"] for row in valid)),
            "mean_probabilities": dict(zip(ROUTES, np.mean(probabilities, axis=0).tolist())) if valid else None,
            "calibration": result})
    decisions_changed = raw_choices_changed = all_valid = 0
    pair_distances = []
    for qid in query_ids:
        decisions = [by_key[(qid, index)]["decision"] for index in range(3)]
        decisions_changed += len({d["route"] for d in decisions}) > 1
        if any(d["metadata"].get("error") for d in decisions):
            continue
        all_valid += 1
        raw_choices_changed += len({d["metadata"]["raw_decision"]["answers"]["retrieval_path"]["choice"] for d in decisions}) > 1
        for first, second in ((0, 1), (0, 2), (1, 2)):
            pair_distances.append(sum(abs(decisions[first]["probabilities"][r] - decisions[second]["probabilities"][r]) for r in ROUTES))
    return {"n_queries": len(query_ids), "n_decisions": len(rows), "orders": orders,
            "queries_with_selected_route_change": decisions_changed,
            "queries_with_all_three_valid_probabilities": all_valid,
            "queries_with_raw_choice_change_among_all_valid": raw_choices_changed,
            "pairwise_probability_l1_mean": float(np.mean(pair_distances)) if pair_distances else None,
            "pairwise_probability_l1_n": len(pair_distances),
            "confidence_threshold": .5, "default_order_index": 0,
            "selection_policy": "Original order 0 fixed before path labels; other orders are descriptive only.",
            "calibration_policy": "10 fixed bins; raw normalized probabilities; no fitted temperature or test selection.",
            "failure_policy": "All calls count in n_total; invalid distributions cannot supply ECE/Brier and are counted separately."}


def report(campaign, routing_directory, raw_directory):
    campaign, routing_directory, raw_directory = map(Path, (campaign, routing_directory, raw_directory))
    acceptance = json.loads((raw_directory / "RAW_ACCEPTANCE.json").read_text(encoding="utf-8"))
    if acceptance.get("status") != "PASS" or acceptance.get("n_decisions") != 1080 or not verify_artifacts(acceptance["artifacts"]):
        raise ValueError("Independent complete raw Laya acceptance is required")
    protocol = freeze_routing(campaign, routing_directory)
    # Commit the prespecified Laya order before reading any path quality rows.
    decisions = decisions_to_protocol(raw_directory, protocol, routing_directory)
    decision_sources = []
    for name in (*REQUIRED_ROUTERS, "laya"):
        path = routing_directory / f"DECISIONS_{name}.json"
        validate_decisions(json.loads(path.read_text(encoding="utf-8")), protocol, name)
        decision_sources.append(artifact(path))
    # This is the first read of quality rows. No test labels feed training/selection.
    records, sources = load_route_records(campaign, protocol, ("dev", "test"))
    by_key = {(row["query_id"], row["path"]): row for row in records}
    raw = read_records(raw_directory / "records.jsonl")
    groups = {}
    for split in ("dev", "test"):
        pop = protocol["populations"][split]
        queries = [{**query, "answerability": by_key[(query["query_id"], "bm25")]["answerability"]}
                   for query in pop["visible_queries"]]
        labels = build_proxy_labels(queries, records, split=split, query_ids=pop["query_ids"],
                                   expected_config_hash=protocol["comparison_config_hash"], tolerance=.01)
        selected = [row for row in raw if row["split"] == split]
        group = study(selected, {row["query_id"]: row["label"] for row in labels}, pop["query_ids"])
        group["proxy_label_counts"] = dict(Counter(row["label"] for row in labels))
        group["all_paths_failed_proxy_queries"] = sum(row["all_paths_failed"] for row in labels)
        group["labels_hash"] = signature(labels)
        for index in range(3):
            replay = []
            for row in selected:
                if row["order_index"] != index:
                    continue
                decision = row["decision"]
                record = by_key[(row["query_id"], decision["route"])]
                replay.append({**record, "selected_route": decision["route"], "fallback": decision["fallback"],
                               "latency_s": record["total_latency_s"] + decision["elapsed_s"], "router_decision": decision})
            group["orders"][index]["descriptive_replay"] = summarize(replay)
        groups[split] = group
    result = {"status": "COMPLETE", "protocol_hash": protocol["protocol_hash"], "groups": groups,
              "model_calls_added": 0, "probability_calibration_is_task_specific_test": True,
              "not_absolute_optimal_route_ground_truth": True,
              "source_artifacts": [artifact(raw_directory / "RAW_ACCEPTANCE.json"), artifact(raw_directory / "records.jsonl")]
                                  + decision_sources + sources}
    immutable_json(routing_directory / "LAYA_PERMUTATION_CALIBRATION.json", result)
    evaluate_decisions(campaign, protocol, decisions, routing_directory)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--routing-dir", type=Path, required=True)
    parser.add_argument("--laya-dir", type=Path, required=True)
    args = parser.parse_args()
    result = report(args.campaign, args.routing_dir, args.laya_dir)
    print(json.dumps({"status": result["status"], "groups": {k: v["n_decisions"] for k, v in result["groups"].items()}}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
