"""Describe all paired dev expansion changes, without a new quality experiment."""
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT))
from ragscope.indexing.identity import artifact, file_hash, atomic_json, signature, verify_artifacts


def main():
    run = Path("D:/RAGScope_Runtime/runs/20261014_v2_retrieval")
    snapshot_path = run / "analysis_complete_dev/SOURCE_SNAPSHOT.json"
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    if not verify_artifacts(snapshot["sources"]):
        raise ValueError("Frozen dev evidence changed")
    evidence = {row["run_id"]: row for row in snapshot["runs"]}
    groups = []
    for group in snapshot["candidates"]["selected"].values():
        observations = {}
        for expansion in ("none", "neighbor"):
            source = evidence[f"E1_{group}_hybrid_{expansion}"]["record_artifact"]
            if file_hash(source["path"]) != source["sha256"]:
                raise ValueError("Pair source changed")
            rows = {}
            with Path(source["path"]).open(encoding="utf-8") as stream:
                for line in stream:
                    row = json.loads(line)
                    expected = row.pop("record_hash")
                    assert signature(row) == expected
                    rows[row["query_id"]] = row
            observations[expansion] = rows
        assert observations["none"].keys() == observations["neighbor"].keys()
        pairs = []
        for query_id in sorted(observations["none"]):
            baseline, neighbor = (observations[mode][query_id] for mode in ("none", "neighbor"))
            assert baseline["answerability"] == neighbor["answerability"]
            values = {}
            for metric in ("full_reference_coverage", "continuous_coverage"):
                left, right = baseline["scores"][metric], neighbor["scores"][metric]
                if left is None:
                    assert right is None
                values[metric] = {"none": left, "neighbor": right, "delta": None if left is None else right-left}
            pairs.append({"query_id": query_id, "doc_id": baseline["doc_id"], "answerability": baseline["answerability"],
                          "baseline_error": baseline["error"], "neighbor_error": neighbor["error"], "metrics": values})
        counts = {}
        for metric in ("full_reference_coverage", "continuous_coverage"):
            values = [p["metrics"][metric]["delta"] for p in pairs if p["metrics"][metric]["delta"] is not None]
            counts[metric] = {"n_applicable": len(values), "improved": sum(v > 1e-12 for v in values),
                              "declined": sum(v < -1e-12 for v in values), "unchanged": sum(abs(v) <= 1e-12 for v in values),
                              "mean_paired_change": sum(values)/len(values)}
            assert len(values) == 92 and sum(counts[metric][key] for key in ("improved", "declined", "unchanged")) == 92
        groups.append({"group": group, "pairs": pairs, "counts": counts})
    report = {"status": "PASS_DESCRIPTIVE_DEV_ONLY", "source_snapshot": artifact(snapshot_path),
              "groups": groups, "n_unique_questions": 100, "note": "Same paired100dev questions per candidate;8 evidence-N/A. Counts are descriptive, not significance tests or test generalization claims."}
    destination = run / "analysis_complete_dev/EXPANSION_PAIRED_CHANGES.json"
    if destination.exists():
        assert json.loads(destination.read_text(encoding="utf-8")) == report
    else:
        atomic_json(destination, report)
    print(json.dumps({"status": report["status"], "groups": [{"group": g["group"], "counts": g["counts"]} for g in groups]}))


if __name__ == "__main__":
    main()
