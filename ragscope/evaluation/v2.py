"""Failure-aware generation scoring and paper-clustered summaries."""
from __future__ import annotations
import math
import random
from dataclasses import asdict, is_dataclass
from collections import defaultdict

from ragscope.evaluation.evidence import evidence_metrics
from ragscope.evaluation.generation_metrics import answer_reference_texts, exact_match, token_f1, rouge_l


def _dict(value):
    return asdict(value) if is_dataclass(value) else dict(value)


def score_generation(query, result, *, evidence_units=None):
    q, r = _dict(query), _dict(result)
    failed = bool(r.get("error"))
    unanswerable = q.get("answerability") in {"unanswerable", False}
    abstain = bool(r.get("abstain")) and not failed
    refs = answer_reference_texts(q.get("answer_refs", []))
    prediction = r.get("answer", "")
    bundle = r.get("context_bundle", {})
    citations = r.get("parsed_output", {}).get("citations", r.get("citations", []))
    pieces = bundle.get("pieces", [])
    visible = {p["chunk_id"] for p in pieces}
    citation_spans = [span for p in pieces if p["chunk_id"] in citations for span in p.get("unit_spans", [])]
    for span in citation_spans:
        # Modern chunks carry doc_id in each span; legacy no-span chunks yield N/A.
        if "doc_id" not in span:
            raise ValueError("Citation provenance requires doc_id")
    scores = {"exact_match": 0.0, "token_f1": 0.0, "rouge_l": 0.0}
    if unanswerable:
        scores = {key: None for key in scores}
    elif not failed and not abstain:
        scores = {"exact_match": exact_match(prediction, refs), "token_f1": token_f1(prediction, refs),
                  "rouge_l": rouge_l(prediction, refs)}
    evidence = evidence_metrics(q.get("evidence_reference_sets", []), bundle.get("unit_spans", []),
                                q.get("answerability", "answerable"), evidence_units)
    cited = evidence_metrics(q.get("evidence_reference_sets", []), citation_spans,
                             q.get("answerability", "answerable"), evidence_units)
    return {**scores, "failed": failed, "answerability": q.get("answerability", "answerable"),
            "gold_unanswerable": unanswerable, "predicted_abstain": abstain,
            "wrong_abstention": float(abstain and not unanswerable),
            "json_valid": float(not failed and bool(r.get("parsed_output"))),
            "citation_visible_rate": sum(c in visible for c in citations)/len(citations) if citations else None,
            "has_citation": float(bool(citations) and not failed),
            "context_evidence": evidence, "citation_evidence": cited,
            "annotation_disagreements": q.get("annotation_disagreements", [])}


def percentile(values, quantile):
    if not values:
        return None
    ordered = sorted(values)
    pos = (len(ordered)-1)*quantile
    lo, hi = math.floor(pos), math.ceil(pos)
    return ordered[lo] + (ordered[hi]-ordered[lo])*(pos-lo)


def summarize_records(records):
    """One record per frozen query. Failures remain in applicable denominators."""
    rows = [_dict(r) for r in records]
    metrics = [r.get("scores", r) for r in rows]
    failures = [bool(r.get("error") or m.get("failed")) for r, m in zip(rows, metrics)]
    summary = {"n_total": len(rows), "n_failed": sum(failures), "n_success": len(rows)-sum(failures)}
    for metric in ("exact_match", "token_f1", "rouge_l", "json_valid", "has_citation", "citation_visible_rate"):
        values = [(0.0 if failed else m.get(metric)) for m, failed in zip(metrics, failures)
                  if m.get(metric) is not None]
        success = [m[metric] for m, failed in zip(metrics, failures) if not failed and m.get(metric) is not None]
        summary[metric] = {"overall": sum(values)/len(values) if values else None, "n_applicable": len(values),
                           "success_conditional": sum(success)/len(success) if success else None,
                           "n_success_applicable": len(success)}
    tp = sum(m.get("gold_unanswerable", False) and m.get("predicted_abstain", False) and not f for m, f in zip(metrics, failures))
    fp = sum(not m.get("gold_unanswerable", False) and m.get("predicted_abstain", False) and not f for m, f in zip(metrics, failures))
    fn = sum(m.get("gold_unanswerable", False) and (not m.get("predicted_abstain", False) or f) for m, f in zip(metrics, failures))
    precision, recall = tp/(tp+fp) if tp+fp else 0.0, tp/(tp+fn) if tp+fn else 0.0
    summary["abstention"] = {"precision": precision, "recall": recall,
                             "f1": 2*precision*recall/(precision+recall) if precision+recall else 0.0,
                             "tp": tp, "fp": fp, "fn": fn}
    latency = [r.get("latency_ms", r.get("timings", {}).get("total_ms")) for r in rows]
    latency = [v for v in latency if v is not None]
    summary["latency_ms"] = {"p50": percentile(latency, .5), "p95": percentile(latency, .95),
                             "p99_exploratory": percentile(latency, .99), "n": len(latency)}
    return summary


def cluster_bootstrap(rows, metric, cluster_key="doc_id", n_resamples=2000, seed=42):
    """Percentile CI with entire papers resampled; row-weighted mean statistic."""
    if n_resamples <= 0:
        raise ValueError("n_resamples must be positive")
    clusters = defaultdict(list)
    for row in rows:
        value = row.get("scores", row).get(metric)
        if value is not None:
            if not math.isfinite(value):
                raise ValueError("Non-finite bootstrap observation")
            clusters[row[cluster_key]].append(float(value))
    if not clusters:
        return {"mean": None, "low": None, "high": None, "n_clusters": 0, "n_resamples": n_resamples}
    groups = list(clusters.values())
    rng, estimates = random.Random(seed), []
    for _ in range(n_resamples):
        sampled = [groups[rng.randrange(len(groups))] for _ in groups]
        estimates.append(sum(sum(g) for g in sampled)/sum(len(g) for g in sampled))
    flat = [v for g in groups for v in g]
    return {"mean": sum(flat)/len(flat), "low": percentile(estimates, .025), "high": percentile(estimates, .975),
            "n_clusters": len(groups), "n_rows": len(flat), "n_resamples": n_resamples, "seed": seed,
            "method": "paper_cluster_percentile", "confidence": .95}


def paired_cluster_bootstrap(left, right, metric, **kwargs):
    def index(rows):
        result = {r["query_id"]: r for r in rows}
        if len(result) != len(rows):
            raise ValueError("Duplicate paired query IDs")
        return result
    a, b = index(left), index(right)
    if set(a) != set(b):
        raise ValueError("Paired comparison requires identical frozen query IDs, including failures")
    differences = []
    for qid in sorted(a):
        if a[qid]["doc_id"] != b[qid]["doc_id"]:
            raise ValueError("Paired paper mismatch")
        av, bv = a[qid].get("scores", a[qid]).get(metric), b[qid].get("scores", b[qid]).get(metric)
        if (av is None) != (bv is None):
            raise ValueError("Paired metric applicability mismatch")
        differences.append({"doc_id": a[qid]["doc_id"], metric: av-bv if av is not None else None})
    return {**cluster_bootstrap(differences, metric, **kwargs), "contrast": "left_minus_right"}


def holm_adjust(p_values):
    """Monotone Holm step-down adjusted p-values in original order."""
    if any(not 0 <= p <= 1 for p in p_values):
        raise ValueError("p-values must lie in [0,1]")
    result, previous = [0.0]*len(p_values), 0.0
    for rank, index in enumerate(sorted(range(len(p_values)), key=lambda i: p_values[i])):
        previous = max(previous, min(1.0, (len(p_values)-rank)*p_values[index]))
        result[index] = previous
    return result
