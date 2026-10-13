import argparse
import json
from ragscope.experiments.runtime import runtime_home
from ragscope.experiments.ledger import read_records
from ragscope.indexing.identity import verify_artifacts

def validate(path):
    state = json.loads((path / "state.json").read_text(encoding="utf-8"))
    errors = []
    for key, task in state["tasks"].items():
        if task["status"] in {"PASS", "PASS_FALLBACK"} and not verify_artifacts(task["outputs"]):
            errors.append(f"Artifact mismatch: {key}")
    files, rows = 0, 0
    for file in path.rglob("*.jsonl"):
        try:
            records = read_records(file)
            identities = [row["record_id"] for row in records if "record_id" in row]
            if len(identities) != len(set(identities)):
                errors.append(f"Duplicate identities: {file}")
            rows += len(records)
            files += 1
        except (ValueError, KeyError) as exc:
            errors.append(str(exc))
    return {"integrity_pass": not errors, "errors": errors, "files": files, "rows": rows,
            "completion_verified": False, "note": "Integrity checks do not imply task completion; see ACCEPTANCE.json."}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", default="20261014_v2")
    args = parser.parse_args()
    result = validate(runtime_home() / "runs" / args.campaign)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["integrity_pass"] else 1)

if __name__ == "__main__":
    main()
