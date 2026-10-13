"""Evidence coverage in canonical source coordinates, before/after packing."""
from __future__ import annotations
from dataclasses import asdict, is_dataclass


def _dict(value):
    return asdict(value) if is_dataclass(value) else dict(value)


def merge_intervals(intervals):
    merged = []
    for start, end in sorted(intervals):
        if end < start:
            raise ValueError("negative canonical interval")
        if end == start:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _coordinates(unit):
    return unit.get("start", unit.get("canonical_start")), unit.get("end", unit.get("canonical_end"))


def evidence_metrics(reference_sets, visible_spans, answerability="answerable", evidence_units=None):
    """Best legal reference set, lexicographic: full-set/full-unit/continuous.

    Continuous coverage averages unit-level coverage (not document hit). String
    unit IDs require evidence_units; unresolved references raise instead of
    treating an invalid annotation as a zero-scoring valid sample.
    """
    empty = {"continuous_coverage": None, "full_unit_recall": None,
             "full_reference_coverage": None, "best_reference_index": None, "reference_results": []}
    if answerability in {"unanswerable", False}:
        return {**empty, "applicability": "unanswerable"}
    if not reference_sets:
        return {**empty, "applicability": "no_reference_evidence"}
    if evidence_units is None:
        evidence_units = {}
    elif not isinstance(evidence_units, dict):
        evidence_units = {_dict(u)["unit_id"]: _dict(u) for u in evidence_units}
    available = {}
    for value in visible_spans:
        span = _dict(value)
        key = (span.get("doc_id"), span["unit_id"])
        available.setdefault(key, []).append(_coordinates(span))
    available = {key: merge_intervals(intervals) for key, intervals in available.items()}
    results = []
    for index, reference in enumerate(reference_sets):
        if not reference:
            continue  # An empty annotation is not evidence of complete support.
        unit_scores, seen = [], set()
        for entry in reference:
            unit = _dict(evidence_units[entry]) if isinstance(entry, str) else _dict(entry)
            uid, doc = unit["unit_id"], unit.get("doc_id")
            start, end = _coordinates(unit)
            if start is None or end is None or end <= start:
                raise ValueError(f"Invalid evidence span: {uid}")
            identity = (doc, uid, start, end)
            if identity in seen:
                continue
            seen.add(identity)
            intervals = available.get((doc, uid), [])
            # Missing doc is permitted for globally stable IDs; no cross-document fallback.
            covered = sum(max(0, min(end, b)-max(start, a)) for a, b in intervals)
            coverage = min(1.0, covered / (end-start))
            unit_scores.append({"unit_id": uid, "coverage": coverage,
                                "fully_covered": covered >= end-start})
        n = len(unit_scores)
        results.append({"reference_index": index, "continuous_coverage": sum(u["coverage"] for u in unit_scores)/n,
                        "full_unit_recall": sum(u["fully_covered"] for u in unit_scores)/n,
                        "full_reference_coverage": float(all(u["fully_covered"] for u in unit_scores)),
                        "units": unit_scores})
    if not results:
        return {**empty, "applicability": "no_reference_evidence"}
    best = max(results, key=lambda r: (r["full_reference_coverage"], r["full_unit_recall"], r["continuous_coverage"]))
    return {key: best[key] for key in ("continuous_coverage", "full_unit_recall", "full_reference_coverage")} | {
        "best_reference_index": best["reference_index"], "reference_results": results,
        "applicability": "answerable"}


def context_statistics(bundle):
    data = _dict(bundle)
    spans = data["unit_spans"]
    intervals, raw = {}, 0
    for span in spans:
        a, b = _coordinates(span)
        raw += b-a
        intervals.setdefault((span.get("doc_id"), span["unit_id"]), []).append((a, b))
    unique = sum(b-a for ranges in intervals.values() for a, b in merge_intervals(ranges))
    budget = data["budget"]
    return {"piece_count": len(data["pieces"]), "duplicate_ratio": (raw-unique)/raw if raw else 0.0,
            "budget_use": budget.get("reference_tokens", 0)/budget["max_reference_tokens"] if budget.get("max_reference_tokens") else 0.0,
            "reference_tokens": budget.get("reference_tokens", 0)}
