"""Materialize actual train200/dev60 routing retrieval inputs after index readiness."""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from ragscope.experiments.retrieval_campaign import run_routing_contexts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--home", type=Path, default=Path("D:/RAGScope_Runtime"))
    parser.add_argument("--campaign", default="20261014_v2")
    parser.add_argument("--embedding-url", required=True)
    parser.add_argument("--embedding-identity")
    parser.add_argument("--routing-manifest-hash", default="f1b6ae1ef21a44b720b53e1a6fffe6600a12d16d51e4b88aefe26a69c0ff952b")
    args = parser.parse_args()
    run_routing_contexts(args.home, args.campaign, args.embedding_url,
                         args.embedding_identity, args.routing_manifest_hash)
