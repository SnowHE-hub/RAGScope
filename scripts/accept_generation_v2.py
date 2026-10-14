"""Validate fixed production populations and retained real rows; no model calls."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import json

from ragscope.experiments.production_acceptance import write_acceptance
from ragscope.experiments.runtime import runtime_home

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign-dir', type=Path, required=True)
    parser.add_argument('--corpus', type=Path, default=runtime_home()/'corpora/20261014_v2')
    parser.add_argument('--scope', choices=['main', 'e2', 'e3'], default='main')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    output = args.output or args.campaign_dir/f'PRODUCTION_ACCEPTANCE_{args.scope.upper()}.json'
    result = write_acceptance(args.campaign_dir, args.corpus, output, scope=args.scope)
    print(json.dumps({key: result.get(key) for key in ('scope', 'status', 'integrity_valid', 'n_expected', 'n_observed', 'n_failed', 'n_missing', 'n_blocked', 'validation_error')}))
    raise SystemExit(0 if result['status'] == 'COMPLETE' else 2)
