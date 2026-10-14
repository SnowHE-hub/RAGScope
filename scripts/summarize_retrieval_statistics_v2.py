"""Actual frozen-test evidence coverage and paired paper-cluster intervals.

The protocol is fixed in configs/v2/retrieval_statistics.json. This script never
selects configurations and never reads generation answers. Missing/incomplete
runs remain explicit; an available paired difference retains every frozen ID.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ragscope.evaluation.v2 import cluster_bootstrap, paired_cluster_bootstrap
from ragscope.indexing.identity import artifact, atomic_json, file_hash, signature
from ragscope.experiments.rerank_sources import accepted_gpu_sources


def read_jsonl(path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def verified_rows(run, run_id, queries, metrics):
    """Validate immutable records, full sample set and source clustering."""
    summary_path = run / "summaries" / (run_id + ".json")
    path = run / "records" / (run_id + ".jsonl")
    config_path = path.with_suffix(".config.json")
    if not all(p.exists() for p in (summary_path, path, config_path)):
        return None, {"status": "UNAVAILABLE", "reason": "No complete frozen run", "run_id": run_id}
    config = json.loads(config_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = read_jsonl(path)
    expected = {q["query_id"]: q for q in queries}
    if len(rows) != len(expected) or {r["query_id"] for r in rows} != set(expected):
        raise ValueError("Completed run does not match full frozen denominator: " + run_id)
    if set(config["query_ids"]) != set(expected) or summary["n_total"] != len(expected):
        raise ValueError("Frozen query identity mismatch: " + run_id)
    prepared = []
    for row in rows:
        if row["config_hash"] != signature(config):
            raise ValueError("Config hash mismatch")
        payload = {k: v for k, v in row.items() if k != "record_hash"}
        if signature(payload) != row["record_hash"]:
            raise ValueError("Record hash mismatch")
        query = expected[row["query_id"]]
        # Ground truth is read only here in an offline statistical report. It is
        # never sent to retrieval; global query scope stays global in run files.
        cluster = query.get("doc_id") or query.get("metadata", {}).get("gold_doc_id_evaluation_only")
        cluster = query.get("metadata", {}).get("article_cluster_id", cluster)
        if not cluster:
            raise ValueError("Missing original source document cluster")
        scores = {}
        for metric in metrics:
            observed = row["scores"][metric]
            if query["answerability"] == "unanswerable":
                if observed is not None:
                    raise ValueError("Unanswerable evidence must remain N/A")
                scores[metric] = None
            else:
                if observed is None or not 0 <= observed <= 1:
                    raise ValueError("Invalid answerable evidence score")
                if row.get("error") and observed != 0:
                    raise ValueError("Failure was excluded or assigned positive evidence")
                scores[metric] = observed
        prepared.append({"query_id": row["query_id"], "doc_id": cluster,
                         "scores": scores, "error": row.get("error")})
    return prepared, {"status": "PASS", "run_id": run_id, "n_total": len(prepared),
                      "n_failed": sum(bool(r["error"]) for r in prepared),
                      "artifacts": [artifact(p) for p in (path, config_path, summary_path)]}


def report(home, campaign):
    project = Path(__file__).resolve().parents[1]
    protocol_path = project / "configs/v2/retrieval_statistics.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    run = home / "runs" / (campaign + "_retrieval")
    selection_path, context_path = run / "CANDIDATE_SELECTION.json", run / "CONTEXT_SELECTION.json"
    result = {"protocol": protocol, "protocol_sha256": file_hash(protocol_path),
              "test_used_for_selection": False, "runs": {}, "comparisons": {}}
    if not selection_path.exists() or not context_path.exists():
        result.update(status="WAITING_FOR_DEV_SELECTION", reason="Final neural dev/context selections are not both frozen")
        atomic_json(run / "RETRIEVAL_STATISTICS.json", result)
        return result
    selected = json.loads(selection_path.read_text(encoding="utf-8"))
    contexts = json.loads(context_path.read_text(encoding="utf-8"))
    if selected["split"] != "dev" or contexts["candidate_selection_sha256"] != file_hash(selection_path):
        raise ValueError("Candidate/context selection provenance differs")
    result["selection_artifacts"] = [artifact(selection_path), artifact(context_path)]
    corpus = home / "corpora" / campaign
    query_sets = {"paper": read_jsonl(corpus / "queries_test.jsonl"),
                  "global": read_jsonl(corpus / "queries_global_test.jsonl"),
                  "squad": read_jsonl(corpus / "squad_e4/queries_test.jsonl")}
    expected_sizes = {"paper": 200, "global": 100, "squad": 100}
    if any(len(query_sets[name]) != size for name, size in expected_sizes.items()):
        raise ValueError("Unexpected frozen test sample size")
    rows_by_key = {}
    gpu_sources = None
    if (run / 'R3_GPU_COMPLETION_VALIDATION.json').is_file():
        gpu_sources = accepted_gpu_sources(home, campaign)
        result['gpu_r3_source_acceptance'] = {
            'artifacts': gpu_sources['evidence'], 'comparison_limit': gpu_sources['comparison_limit']}
    options = [("paper", family, method) for family in ("simple", "structured") for method in ("bm25", "dense", "hybrid")]
    options += [("global", "structured", method) for method in ("bm25", "dense", "hybrid")]
    # R3 is a required existing route, included descriptively when its real
    # CPU-reranked records arrive. This adds no post-hoc selection or contrast.
    options += [(task, "structured", "hybrid_rerank") for task in ("paper", "global")]
    options.append(("squad", "structured", "hybrid"))
    params = {"n_resamples": protocol["n_resamples"], "seed": protocol["seed"]}
    for task, family, method in options:
        group = selected["selected"][family]
        prefix = {"paper": "TEST_CONTEXT", "global": "E8", "squad": "E4_CONTEXT"}[task]
        run_id = f"{prefix}_{group}_{method}_{contexts['selected'][group]}"
        if task == 'global' and method == 'hybrid_rerank' and gpu_sources is not None:
            run_id = gpu_sources['paths']['E8'].stem
        rows, evidence = verified_rows(run, run_id, query_sets[task], protocol["metrics"])
        key = (task, family, method)
        rows_by_key[key] = rows
        if rows is not None:
            evidence["metrics"] = {metric: cluster_bootstrap(rows, metric, **params) for metric in protocol["metrics"]}
        result["runs"][run_id] = evidence
    for comparison in protocol["comparisons"]:
        left = rows_by_key[(comparison["task"], *comparison["left"])]
        right = rows_by_key[(comparison["task"], *comparison["right"])]
        item = {**comparison, "classification": "exploratory"}
        if left is None or right is None:
            item.update(status="UNAVAILABLE", reason="One or both complete paired runs unavailable")
        else:
            item.update(status="PASS", metrics={metric: paired_cluster_bootstrap(left, right, metric, **params)
                                                 for metric in protocol["metrics"]})
        result["comparisons"][comparison["id"]] = item
    result["status"] = "PASS" if all(r["status"] == "PASS" for r in result["runs"].values()) else "PARTIAL_UNAVAILABLE"
    atomic_json(run / "RETRIEVAL_STATISTICS.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--home", type=Path, default=Path("D:/RAGScope_Runtime"))
    parser.add_argument("--campaign", default="20261014_v2")
    args = parser.parse_args()
    result = report(args.home, args.campaign)
    print(json.dumps({"status": result["status"], "runs": len(result["runs"]), "comparisons": len(result["comparisons"])}))
