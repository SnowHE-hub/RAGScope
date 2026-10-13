"""CPU-only validation of the frozen scoped simple comparator; no neural calls."""
from pathlib import Path
import argparse
import json
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ragscope.experiments.simple_scope import prepare_simple_scope

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--home", type=Path, default=Path("D:/RAGScope_Runtime"))
    parser.add_argument("--campaign", default="20261014_v2")
    args = parser.parse_args()
    result = prepare_simple_scope(args.home, args.campaign)
    print(json.dumps({k: result[k] for k in ("status", "scoped_documents", "full_reference_chunks",
        "scoped_chunks", "neural_vectors_avoided", "neural_body_tokens_avoided", "neural_calls")}))
