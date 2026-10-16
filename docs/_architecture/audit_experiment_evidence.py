"""Read-only, standard-library recount for the architecture documentation.

No model imports, server requests, downloads, training, or writes to runtime data.
All writes are restricted to this documentation evidence directory.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNS = Path("D:/RAGScope_Runtime/runs")
OUT = Path(__file__).with_name("experiment_audit.json")
sources: dict[str, dict] = {}


def identity(p: Path) -> dict:
    name = str(p)
    if name not in sources:
        h = hashlib.sha256()
        with p.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        sources[name] = {"bytes": p.stat().st_size, "sha256": h.hexdigest()}
    return {"path": name, **sources[name]}


def rows(p: Path):
    identity(p)
    with p.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def mean(xs):
    ys = [float(x) for x in xs if x is not None]
    return statistics.mean(ys) if ys else None


def percentile(xs, q):
    ys = sorted(float(x) for x in xs if x is not None)
    if not ys:
        return None
    pos = (len(ys) - 1) * q
    a, b = math.floor(pos), math.ceil(pos)
    return ys[a] * (b - pos) + ys[b] * (pos - a) if a != b else ys[a]


def grouped(seq, keys):
    d = defaultdict(list)
    for r in seq:
        d[":".join(str(r.get(k, "UNKNOWN")) for k in keys)].append(r)
    return d


audit = {"scope": "Existing raw records only; no new inference or benchmark", "checked_on": "2026-10-15", "limitations": ["Aggregates stored per-record metric fields; does not independently adjudicate scientific factual correctness.", "Retains failed records and skips only explicit metric null/N/A.", "V1 original model revision is UNKNOWN; reconstructed fixed-revision E0 is separate."]}

p = ROOT / "rerun/per_query.csv"
identity(p)
with p.open(encoding="utf-8-sig", newline="") as f:
    old = list(csv.DictReader(f))
audit["V1_original_per_query"] = {
    k: {"n": len(rs), "unique_questions": len({r["qid"] for r in rs}), **{m: mean(float(r[m]) for r in rs) for m in ["r5", "r10", "h5", "mrr", "ndcg10", "lat"]}}
    for k, rs in grouped(old, ["cfg"]).items()
}
audit["E0_fixed_revision_reconstruction"] = {}
for method in ["dense", "hybrid"]:
    p = RUNS / "20261014_v2_legacy_reconstructed" / (method + ".jsonl")
    rs = list(rows(p))
    audit["E0_fixed_revision_reconstruction"][method] = {"n": len(rs), "real_model_n": sum(r.get("real_model") is True for r in rs), "failures": sum(bool(r.get("error")) for r in rs), **{m: mean(r["metrics"].get(m) for r in rs) for m in ["recall5", "recall10", "mrr", "ndcg10"]}, "identity": rs[0]["model_identity"]}

v2files = []
for run, patterns in [("20261014_v2_generation_e2", ["E2_*.jsonl"]), ("20261014_v2_generation_e3", ["E3_*.jsonl"]), ("20261014_v2_generation", ["E4_*.jsonl", "E5_*.jsonl", "E6_*.jsonl", "E7_*.jsonl", "ROUTE_*_qwen_2b.jsonl"])]:
    for pattern in patterns:
        v2files.extend((RUNS / run / "records").glob(pattern))
v2groups = defaultdict(list)
for p in v2files:
    for r in rows(p):
        v2groups[f"{r['experiment_id']}:{r['model_id']}:{r['method']}"].append(r)
audit["V2_generation"] = {}
for k, rs in sorted(v2groups.items()):
    audit["V2_generation"][k] = {"n": len(rs), "unique_questions": len({r["query_id"] for r in rs}), "real_model_n": sum(r.get("real_model") is True for r in rs), "failures": sum(r["scores"].get("failed") is True for r in rs), "answerable_n": sum(r["scores"].get("token_f1") is not None for r in rs), "token_f1": mean(r["scores"].get("token_f1") for r in rs), "evidence_full_coverage": mean((r["scores"].get("context_evidence") or {}).get("full_reference_coverage") for r in rs), "generation_p50_ms": percentile((r.get("latency_ms") for r in rs), .5)}
audit["V2_generation_total"] = {"n": sum(x["n"] for x in audit["V2_generation"].values()), "groups": len(v2groups), "failures": sum(x["failures"] for x in audit["V2_generation"].values())}

audit["V2_dev_chunking_selected"] = {}
for recipe in ["F512_O0", "R512"]:
    for expansion in ["none", "neighbor"]:
        name = f"E1_{recipe}_hybrid_{expansion}"
        rs = list(rows(RUNS / "20261014_v2_retrieval/records" / (name + ".jsonl")))
        audit["V2_dev_chunking_selected"][name] = {"n": len(rs), "evidence_n": sum(r["scores"].get("full_reference_coverage") is not None for r in rs), **{m: mean(r["scores"].get(m) for r in rs) for m in ["full_reference_coverage", "continuous_coverage"]}}
audit["V2_E1_all_nonrerank"] = {}
for p in (RUNS / "20261014_v2_retrieval/records").glob("E1_*.jsonl"):
    if "rerank" in p.name:
        continue
    rs = list(rows(p))
    audit["V2_E1_all_nonrerank"][p.stem] = {"n": len(rs), "unique_questions": len({r["query_id"] for r in rs}), "failures": sum(bool(r.get("error")) for r in rs), "full_reference_coverage": mean(r["scores"].get("full_reference_coverage") for r in rs)}
audit["V2_E8_global_retrieval"] = {}
for method in ["bm25", "dense", "hybrid", "hybrid_rerank"]:
    suffix = "_gpu" if method == "hybrid_rerank" else ""
    p = RUNS / "20261014_v2_retrieval/records" / f"E8_R512_{method}_neighbor{suffix}.jsonl"
    rs = list(rows(p))
    audit["V2_E8_global_retrieval"][method] = {"n": len(rs), "failures": sum(bool(r.get("error")) for r in rs), "full_reference_coverage": mean(r["scores"].get("full_reference_coverage") for r in rs)}

audit["V2_LoRA_same_BF16"] = {}
for condition in ["baseline", "adapted"]:
    rs = list(rows(RUNS / "20261014_v2_lora/train_2b_1024_seed42/paired_test100" / (condition + ".jsonl")))
    audit["V2_LoRA_same_BF16"][condition] = {"n": len(rs), "real_model_n": sum(r.get("real_model") is True for r in rs), "failures": sum(r["scores"]["failed"] is True for r in rs), "answerable_n": sum(r["scores"].get("token_f1") is not None for r in rs), "wrong_abstention_n": sum(r["scores"].get("wrong_abstention") == 1 for r in rs), "token_f1": mean(r["scores"].get("token_f1") for r in rs), "generation_p50_ms": percentile((r.get("latency_ms") for r in rs), .5)}

audit["V2_context_stress"] = {}
for window in [4096, 8192]:
    rs = list(rows(RUNS / f"20261014_v2_context_stress/records/window_{window}.jsonl"))
    audit["V2_context_stress"][str(window)] = {"n": len(rs), "real_model_n": sum(r["real_model"] is True for r in rs), "failed_n": sum(bool(r.get("error")) for r in rs), "generation_p95_s": percentile((r["timings"]["generation_ms"] / 1000 for r in rs), .95), "generation_mean_s": mean(r["timings"]["generation_ms"] / 1000 for r in rs), "actual_prompt_tokens": sorted(set(r["backend_metadata"].get("actual_input_tokens") for r in rs)), "excluded_from_quality_ranking_n": sum(r["quality_ranking_eligible"] is False for r in rs)}

rs = list(rows(RUNS / "20261014_v2_laya/records.jsonl"))
audit["V2_Laya_actual_CPU"] = {"n_decisions": len(rs), "n_questions": len({r["query_id"] for r in rs}), "real_model_n": sum(r["real_model"] is True for r in rs), "fallback_n": sum(r["decision"]["fallback"] is True for r in rs)}
audit["V2_router_replays"] = {}
for policy in ["fixed_bm25", "fixed_hybrid", "fixed_hybrid_rerank", "rules", "linear", "mlp", "lstm", "laya"]:
    p = RUNS / "20261014_v2_routing" / ("REPLAY_" + policy + ".json")
    identity(p)
    data = json.loads(p.read_text(encoding="utf-8"))
    audit["V2_router_replays"][policy] = {split: {"n": len(g["rows"]), "route_counts": {k: len(v) for k, v in grouped(g["rows"], ["selected_route"]).items()}, "mean_answer_f1": mean(r.get("answer_f1") for r in g["rows"]), "p50_component_replay_s": percentile((r["latency_s"] for r in g["rows"]), .5)} for split, g in data["groups"].items()}

rs = list(rows(RUNS / "20261014_v2_routing/online/records.jsonl"))
audit["V2_actual_online"] = {k: {"n": len(rr), "actual_online_n": sum(r["latency_kind"] == "actual_online" for r in rr), "real_model_n": sum(r.get("real_model") is True for r in rr), "p50_s": percentile((r["latency_s"] for r in rr), .5), "p95_s": percentile((r["latency_s"] for r in rr), .95), "answer_f1": mean(r.get("answer_f1") for r in rr), "answerable_n": sum(r.get("answer_f1") is not None for r in rr)} for k, rr in grouped(rs, ["policy"]).items()}

v3 = RUNS / "v3_20261015_014506"
rs = list(rows(v3 / "BERTSCORE_PAIRS_nli_base_cpu_smallbatch.jsonl"))
audit["V3_BERTScore_raw_pairs"] = {"n": len(rs), "real_encoder_n": sum(r["real_encoder"] is True for r in rs), "truncated_n": sum(r["truncated_by_official_implementation"] is True for r in rs)}
rs = list(rows(v3 / "CLAIM_ALL_OLD_COVERAGE_AND_UNKNOWNS.jsonl"))
audit["V3_old_fact_proxy_availability"] = {"n": len(rs), "status_counts": {k: len(v) for k, v in grouped(rs, ["status"]).items()}, "claim_f1_unknown_n": sum(r.get("claim_f1_proxy") is None for r in rs)}
rs = list(rows(v3 / "S2_PROTOCOL_SCORED864.jsonl"))
audit["V3_diagnostic_protocol"] = {k: {"n": len(rr), "unique_questions": len({r["query_id"] for r in rr}), "failures": sum(r["failed"] is True for r in rr), "word_f1": mean(r.get("legacy_token_f1") for r in rr)} for k, rr in grouped(rs, ["model_id", "context_kind", "protocol"]).items()}
# The raw new-request record plus separately stored reused G0 record proves execution,
# while the score projection is the bounded metric input, not 864 new requests.
for name in ["S2_PROTOCOL_RESULTS.jsonl", "S2_G0_REUSED_RESULTS.jsonl", "S2_SUPPLEMENT_RESULTS_verified.jsonl"]:
    rr = list(rows(v3 / name))
    audit.setdefault("V3_protocol_request_files", {})[name] = {"n": len(rr), "first_record_keys": list(rr[0]) if rr else [], "context_kind_counts": {k: len(v) for k, v in grouped(rr, ["context_kind"]).items()}, "dataset_role_counts": {k: len(v) for k, v in grouped(rr, ["dataset_role"]).items()}, "explicit_reused_actual_n": sum(r.get("reused_actual_model_result") is True for r in rr)}

rr = list(rows(v3 / "S4_WORKFLOW_GENERATION.jsonl"))
audit["V3_workflow_actual_generations"] = {"n": len(rr), "condition_counts": {k: len(v) for k, v in grouped(rr, ["condition"]).items()}}
rr = list(rows(v3 / "S4_WORKFLOW_UTILITY_VECTORS.jsonl"))
audit["V3_workflow_proxy_recount"] = {"n_questions": len(rr), "mapping": mean(r["mapping_proxy_quality"] for r in rr), "best_fixed": mean(r["fixed_proxy_quality"] for r in rr), "hindsight_oracle": mean(r["oracle_proxy_quality"] for r in rr), "mapping_minus_fixed": mean(r["mapping_proxy_quality"] - r["fixed_proxy_quality"] for r in rr)}

audit["V3_BF16_runtime_probe"] = {}
for condition in ["base", "unmerged"]:
    rr = list(rows(v3 / ("S5_RUNTIME_" + condition + ".jsonl")))
    audit["V3_BF16_runtime_probe"][condition] = {"n": len(rr), "failures": sum(not r["success"] for r in rr), "p50_s": percentile((r["wall_time_ms"] / 1000 for r in rr), .5), "output_tokens_p50": percentile((r["output_tokens"] for r in rr), .5), "stop_counts": {k: len(v) for k, v in grouped(rr, ["stop_reason"]).items()}}

finish = RUNS / "v3_finish_20261015_1415"
audit["V3_finish_retrieval_development"] = {}
rr = list(rows(finish / "DEV_RETRIEVAL.jsonl"))
for k, rs in grouped(rr, ["split", "recipe"]).items():
    audit["V3_finish_retrieval_development"][k] = {"n": len(rs), "full_reference_coverage": mean((r.get("evidence") or {}).get("full_reference_coverage") for r in rs)}
audit["V3_finish_generation_development"] = {}
for name in ["DEV_GENERATION.jsonl", "QA_PROMPT_CONTROL.jsonl"]:
    rr = list(rows(finish / name))
    for k, rs in grouped(rr, ["model", "condition"]).items():
        audit["V3_finish_generation_development"][k] = {"n": len(rs), "real_model_n": sum(r.get("real_model") is True for r in rs), "failures": sum(not r["success"] for r in rs), "p50_s": percentile((r["elapsed_s"] for r in rs), .5)}

rr = list(rows(finish / "FINAL_SCORES_RESIDENT8K.jsonl"))
audit["V3_finish_scored"] = {}
for k, rs in grouped(rr, ["split", "condition"]).items():
    metrics = ["token_f1", "exact_match", "rouge_l", "semantic_cosine", "evidence_full_coverage", "correct_abstain", "wrong_abstain", "citation_visibility"]
    audit["V3_finish_scored"][k] = {"n": len(rs), "unique_questions": len({r["query_id"] for r in rs}), "failures": sum(r["metrics"].get("failure") is True for r in rs), **{m: {"mean": mean(r["metrics"].get(m) for r in rs), "n": sum(r["metrics"].get(m) is not None for r in rs)} for m in metrics}, "all_factual_unknown": all(r["factual_correctness"] == "UNKNOWN" for r in rs), "all_human_not_reviewed": all(r["human_review"] == "NOT_REVIEWED" for r in rs)}
# Summary correct-abstain/wrong-abstain intentionally uses only relevant populations;
# record booleans exist on all rows. Recompute the intended populations explicitly.
for k, rs in grouped(rr, ["split", "condition"]).items():
    a = [r for r in rs if r["metrics"].get("token_f1") is not None]
    u = [r for r in rs if r["metrics"].get("token_f1") is None]
    audit["V3_finish_scored"][k]["correct_abstain_relevant"] = {"n": len(u), "mean": mean(r["metrics"].get("correct_abstain") for r in u)}
    audit["V3_finish_scored"][k]["wrong_abstain_relevant"] = {"n": len(a), "mean": mean(r["metrics"].get("wrong_abstain") for r in a)}
audit["V3_finish_raw_generation"] = {}
for name in ["FINAL_GENERATION.jsonl", "DEPLOYMENT_FINAL_ANSWERS_RESIDENT8K.jsonl"]:
    rs = list(rows(finish / name))
    audit["V3_finish_raw_generation"][name] = {"n": len(rs), "real_model_n": sum(r.get("real_model") is True for r in rs), "success_n": sum(r.get("success") is True for r in rs), "groups": {k: len(v) for k, v in grouped(rs, ["split", "condition"]).items()}}

rs = list(rows(finish / "FINAL_GLOBAL_RETRIEVAL.jsonl"))
audit["V3_finish_global"] = {k: {"n": len(rr), "doc_hit_at10": mean(r["doc_hit_at10"] for r in rr), "evidence_n": sum((r["evidence"] or {}).get("full_reference_coverage") is not None for r in rr), "full_reference_coverage": mean((r["evidence"] or {}).get("full_reference_coverage") for r in rr)} for k, rr in grouped(rs, ["condition", "method"]).items()}
audit["V3_finish_online"] = {}
for name in ["FINAL_ONLINE.jsonl", "FINAL_ONLINE_RESIDENT8K.jsonl"]:
    rs = list(rows(finish / name))
    audit["V3_finish_online"][name] = {k: {"n": len(rr), "real_model_n": sum(r.get("real_model") is True for r in rr), "failures": sum(not r["success"] for r in rr), "answer_cache_false_n": sum(r["answer_cache"] is False for r in rr), "query_embedding_cache_false_n": sum(r["query_embedding_cache"] is False for r in rr), "p50_s": percentile((r["elapsed_s"] for r in rr), .5), "p95_s": percentile((r["elapsed_s"] for r in rr), .95), "num_ctx_counts": {str(v): sum(r["actual_sent_payload"]["options"]["num_ctx"] == v for r in rr) for v in set(r["actual_sent_payload"]["options"]["num_ctx"] for r in rr)}} for k, rr in grouped(rs, ["condition"]).items()}

# Configuration and small numeric audit artifacts are recorded separately from
# request/step logs; their status must not masquerade as new experiments.
audit["supporting_numeric_artifacts"] = {}
for rel in ["20261014_v2_lora/train_2b_1024_seed42/TRAINING_AUDIT.json", "20261014_v2_lora/train_2b_1024_seed42/steps.config.json", "20261014_v2_lora/prepared_2b_1024_v2/MASK_AND_SPLIT_AUDIT.json", "v3_20261015_014506/S5_MERGE_LOGIT_GATE.json", "v3_20261015_014506/S3_NATURAL_RERANK_D_confirm_ANALYSIS.json", "v3_20261015_014506/BERTSCORE_ALL_OLD_nli_base_cpu_smallbatch.json", "v3_finish_20261015_1415/DEPLOYMENT_ANSWERS_COMPLETION_RESIDENT8K.json"]:
    p = RUNS / rel
    identity(p)
    d = json.loads(p.read_text(encoding="utf-8"))
    audit["supporting_numeric_artifacts"][rel] = {k: v for k, v in d.items() if k in ["status", "n", "config", "steps", "actual_optimizer_steps", "train_examples", "epochs", "dev_losses", "all_restore_checks_exact", "all_losses_and_gradients_finite", "splits", "passed", "checks", "groups", "unique_pair_n", "truncated_pairs", "new_llm_requests", "reused_exact_requests", "new_calls", "failures"]}
p = RUNS / "20261014_v2_lora/train_2b_1024_seed42/steps.jsonl"
rs = list(rows(p))
audit["V2_LoRA_training_steps"] = {"n": len(rs), "step_ids_complete": sorted(r["step"] for r in rs) == list(range(1,123)), "all_losses_finite": all(math.isfinite(r["loss"]) for r in rs), "first_loss": rs[0]["loss"], "last_loss": rs[-1]["loss"]}

audit["source_files"] = sources
OUT.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({"output": str(OUT), "source_files": len(sources), "V2": audit["V2_generation_total"], "V3_scored_groups": {k: {"n": v["n"], "f1": v["token_f1"], "correct_abstain": v["correct_abstain_relevant"]} for k,v in audit["V3_finish_scored"].items()}}, ensure_ascii=False))
