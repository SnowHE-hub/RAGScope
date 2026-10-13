import argparse
import json
from ragscope.experiments.runtime import runtime_home

def main():
    parser = argparse.ArgumentParser(description="RAGScope 本地实验状态")
    parser.add_argument("--campaign", default="20261014_v2")
    args = parser.parse_args()
    path = runtime_home() / "runs" / args.campaign / "state.json"
    if not path.is_file():
        parser.error(f"Campaign not started: {path}")
    state = json.loads(path.read_text(encoding="utf-8"))
    print(json.dumps({"campaign_id": state["campaign_id"], "status": state["status"],
        "tasks": {key: {k: value.get(k) for k in ("status", "attempts", "validation")} for key, value in state["tasks"].items()}}, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
