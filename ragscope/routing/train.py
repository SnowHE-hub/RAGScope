"""Train a bounded router from frozen real-generation proxy-label JSONL."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ragscope.indexing.identity import artifact, atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kind", choices=["linear", "mlp", "lstm"], required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.labels.read_text(encoding="utf-8").splitlines() if line.strip()]
    train, dev = [row for row in rows if row["split"] == "train"], [row for row in rows if row["split"] == "dev"]
    if args.kind == "lstm":
        from .lstm import LSTMRouter
        router = LSTMRouter()
    else:
        from .learned import TfidfRouter
        router = TfidfRouter(args.kind)
    router.fit(train, dev)
    router.save(args.output)
    atomic_json(args.output / "training_report.json", {**router.report, "input": artifact(args.labels)})
    print(json.dumps({"status": router.report["status"], "output": str(args.output), "kind": args.kind}))


if __name__ == "__main__":
    main()
