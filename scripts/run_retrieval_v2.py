"""Run the bounded v2 CPU matrix; model service must be explicitly supplied."""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from ragscope.experiments.retrieval_campaign import run_campaign, run_test_contexts, run_final_contexts, freeze_context_choices


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--home", type=Path, default=Path(os.environ.get("RAGSCOPE_HOME", "D:/RAGScope_Runtime")))
    parser.add_argument("--campaign", default="20261014_v2")
    parser.add_argument("--stage", choices=("bm25", "dense", "all"), default="bm25")
    parser.add_argument("--embedding-url")
    parser.add_argument("--embedding-identity")
    parser.add_argument("--groups", nargs="*")
    parser.add_argument("--expansion-comparison", action="store_true")
    parser.add_argument("--test-contexts", action="store_true")
    parser.add_argument("--final-contexts", action="store_true")
    parser.add_argument("--selected-expansions", action="store_true")
    parser.add_argument("--dataset", choices=("qasper", "squad"), default="qasper")
    parser.add_argument("--methods", choices=("bm25", "dense", "hybrid"), nargs="+")
    parser.add_argument("--family", choices=("simple", "structured"),
                        help="Schedule one already selected family; does not change per-query experiment identity")
    args = parser.parse_args()
    if args.selected_expansions:
        selected = json.loads((args.home / "runs" / (args.campaign + "_retrieval") / "CANDIDATE_SELECTION.json").read_text(encoding="utf-8"))
        groups = list(selected["selected"].values())
        # One prespecified parent-family diagnostic keeps none/neighbor/parent
        # comparison available when the selected structured family is recursive.
        if not any(group.startswith("P256_") for group in groups):
            groups.append("P256_1024")
        run_campaign(args.home, args.campaign, "dense", args.embedding_url, args.embedding_identity, groups, True)
        freeze_context_choices(args.home, args.campaign)
    elif args.final_contexts:
        result = run_final_contexts(args.home, args.campaign, args.dataset, args.embedding_url,
                                    args.embedding_identity, args.methods, args.family)
        if result["status"] == "BLOCKED_RESOURCE":
            sys.exit(2)
    elif args.test_contexts:
        result = run_test_contexts(args.home, args.campaign)
        if result["status"] == "BLOCKED_RESOURCE":
            sys.exit(2)
    else:
        run_campaign(args.home, args.campaign, args.stage, args.embedding_url, args.embedding_identity,
                     args.groups, args.expansion_comparison)
