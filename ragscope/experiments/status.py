"""Read all 25 plan tasks, including pending work, without launching experiments."""
import argparse
import json
from types import SimpleNamespace

from ragscope.experiments.runtime import runtime_home
from ragscope.experiments.task_catalog import load_catalog, reconcile, summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", default="20261014_v2")
    parser.add_argument("--details", action="store_true", help="Include all source hashes and observed child metrics")
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args(argv)
    # Reuse the safe ID validator without entering or creating a Campaign.
    if not args.campaign or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in args.campaign):
        parser.error("Unsafe campaign ID")
    path = runtime_home() / "runs" / args.campaign
    state_path = path / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {"tasks": {}}
    campaign = SimpleNamespace(path=path, campaign_id=args.campaign, state=state)
    report = reconcile(campaign, load_catalog(), persist=False)
    report["state_file_exists"] = state_path.exists()
    print(json.dumps(report if args.details else summary(report), ensure_ascii=False, indent=2))
    return 2 if args.require_complete and report["status"] != "COMPLETE" else 0


if __name__ == "__main__":
    raise SystemExit(main())
