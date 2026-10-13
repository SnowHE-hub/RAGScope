"""Independently rebuild retrieval evidence means from immutable source slices.

This intentionally does not import the production evidence/scoring/summary code.
Only completed per-run summaries are audited; an active incomplete run is never
promoted to a completed result. All numbers are reconstructed from record rows.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


METRICS = ("full_reference_coverage", "full_unit_recall", "continuous_coverage")


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def rows(path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def merged_length(ranges):
    ordered = sorted((a, b) for a, b in ranges if b > a)
    total, end = 0, -1
    for start, stop in ordered:
        total += max(0, stop-max(start, end))
        end = max(end, stop)
    return total


def independently_score(query, spans, units, failed=False):
    if query["answerability"] == "unanswerable":
        return {key: None for key in METRICS}
    if failed:
        return {key: 0.0 for key in METRICS}
    alternatives = []
    for reference in query["evidence_reference_sets"]:
        fractions = []
        for unit_id in dict.fromkeys(reference):
            unit = units[unit_id]
            start, end = unit["canonical_start"], unit["canonical_end"]
            intersections = [(max(start, s["start"]), min(end, s["end"])) for s in spans
                             if s["unit_id"] == unit_id and s["doc_id"] == unit["doc_id"]]
            fractions.append(merged_length(intersections)/(end-start))
        if fractions:
            alternatives.append((float(all(v == 1 for v in fractions)),
                                 sum(v == 1 for v in fractions)/len(fractions),
                                 sum(fractions)/len(fractions)))
    assert alternatives, "Answerable frozen sample has no valid reference"
    return dict(zip(METRICS, max(alternatives)))


def quantile(values, fraction):
    values = sorted(values)
    if not values:
        return None
    position = (len(values)-1)*fraction
    lower, upper = math.floor(position), math.ceil(position)
    return values[lower]+(values[upper]-values[lower])*(position-lower)


def audit(home, campaign):
    corpus = home / "corpora" / campaign
    documents, queries = {}, {}
    for base in (corpus, corpus / "squad_e4"):
        if not (base / "documents.jsonl").exists():
            continue
        documents.update((d["doc_id"], d) for d in rows(base / "documents.jsonl"))
        for split in ("train", "dev", "test", "global_test"):
            path = base / ("queries_" + split + ".jsonl")
            if path.exists():
                queries.update((q["query_id"], q) for q in rows(path))
    units = {u["unit_id"]: u for document in documents.values() for u in document["evidence_units"]}
    run = home / "runs" / (campaign + "_retrieval")
    audited, aggregate_checks, latencies = [], [], {}
    total_rows = source_pieces = 0
    for summary_path in sorted((run / "summaries").glob("*.json")):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if "records" not in summary:
            continue
        record_path = Path(summary["records"])
        assert record_path.resolve().is_relative_to((run / "records").resolve())
        config = json.loads(record_path.with_suffix(".config.json").read_text(encoding="utf-8"))
        config_hash = signature(config)
        values, timing, ids, failures, cache_hits = defaultdict(list), defaultdict(list), set(), 0, []
        count = 0
        for row in rows(record_path):
            count += 1
            stored_hash = row.pop("record_hash")
            assert signature(row) == stored_hash, "Record hash mismatch"
            assert row["config_hash"] == config_hash and row["record_id"] not in ids
            ids.add(row["record_id"])
            query = queries[row["query_id"]]
            failed = bool(row.get("error"))
            failures += failed
            bundle = row.get("context_bundle", {})
            assert bundle.get("unit_spans", []) == [s for piece in bundle.get("pieces", []) for s in piece["unit_spans"]]
            assert bundle.get("budget", {}).get("reference_tokens", 0) <= 2048
            for piece in bundle.get("pieces", []):
                start, end = piece["canonical_start"], piece["canonical_end"]
                document = documents[piece["doc_id"]]
                assert document["text"][start:end] == piece["text"], "Context/source slice differs"
                for span in piece["unit_spans"]:
                    unit = units[span["unit_id"]]
                    assert span["doc_id"] == piece["doc_id"] == unit["doc_id"]
                    assert start <= span["start"] < span["end"] <= end
                    assert unit["canonical_start"] <= span["start"] < span["end"] <= unit["canonical_end"]
                    assert span["chunk_start"] == span["start"]-start and span["chunk_end"] == span["end"]-start
                source_pieces += 1
            expected = independently_score(query, bundle.get("unit_spans", []), units, failed)
            for key, value in expected.items():
                observed = row["scores"][key]
                assert observed is None if value is None else abs(observed-value) <= 1e-12
                if value is not None:
                    values[key].append(value)
            for key, value in row.get("timings", {}).items():
                if isinstance(value, (float, int)):
                    timing[key].append(value)
            cache_hits.extend(m["cache_hit"] for m in row.get("query_encoding_metrics", []) if "cache_hit" in m)
        assert count == summary["n_total"] and failures == summary["n_failed"]
        assert ids == set(config["query_ids"]), "Missing/duplicated frozen sample"
        for key in METRICS:
            mean = sum(values[key])/len(values[key]) if values[key] else None
            reported = summary[key]["overall"]
            assert reported is None if mean is None else abs(reported-mean) <= 1e-12
            assert len(values[key]) == summary[key]["n_applicable"]
            aggregate_checks.append({"run_id": summary["run_id"], "metric": key, "n": len(values[key]),
                                     "reconstructed": mean, "reported": reported, "match": True})
        latencies[summary["run_id"]] = {"n": count, "query_cache_observations": len(cache_hits),
            "query_cache_hits": sum(cache_hits), "query_cache_misses": len(cache_hits)-sum(cache_hits),
            "components_ms": {key: {"n": len(v), "p50": quantile(v, .5), "p95": quantile(v, .95)} for key, v in timing.items()},
            "note": "Absent components are unavailable; cached query encoding is not cold inference latency."}
        audited.append({"run_id": summary["run_id"], "records": count, "record_sha256": sha(record_path),
                        "summary_sha256": sha(summary_path), "config_hash": config_hash})
        total_rows += count
    randomizer = random.Random(42)
    result = {"status": "PASS", "as_of_utc": datetime.now(timezone.utc).isoformat(), "runs": audited,
              "records_checked": total_rows, "context_source_pieces_checked": source_pieces,
              "aggregate_numbers_checked": len(aggregate_checks), "aggregate_checks": aggregate_checks,
              "seed42_sample_10_numbers": randomizer.sample(aggregate_checks, min(10, len(aggregate_checks))),
              "independent_metric_implementation": True, "unfinished_runs_not_promoted": True}
    (run / "RETRIEVAL_INDEPENDENT_AUDIT.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (run / "RETRIEVAL_LATENCY_AUDIT.json").write_text(json.dumps(latencies, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "PASS", "completed_runs": len(audited), "records_checked": total_rows,
                      "context_source_pieces_checked": source_pieces, "aggregate_numbers_checked": len(aggregate_checks)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--home", type=Path, default=Path("D:/RAGScope_Runtime"))
    parser.add_argument("--campaign", default="20261014_v2")
    args = parser.parse_args()
    audit(args.home, args.campaign)
