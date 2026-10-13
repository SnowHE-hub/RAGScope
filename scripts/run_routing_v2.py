"""Freeze, prepare actual labels, and evaluate immutable router decisions locally."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import json

from ragscope.experiments.runtime import configure
from ragscope.indexing.identity import atomic_json
from ragscope.routing.campaign import freeze_routing, prepare_labels, freeze_decisions, evaluate_decisions
from ragscope.routing.core import FixedRouter, RuleRouter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--mode', choices=['freeze', 'labels', 'replay'], required=True)
    parser.add_argument('--router', choices=['fixed_bm25', 'fixed_hybrid', 'fixed_hybrid_rerank', 'rules', 'linear', 'mlp', 'lstm', 'laya'])
    parser.add_argument('--router-artifact', type=Path)
    args = parser.parse_args()
    configure(cpu_only=True)
    if not (args.campaign/'EXPERIMENT_MANIFEST.json').exists():
        atomic_json(args.output/'READINESS.json', {'status': 'WAITING_FOR_FROZEN_GENERATION', 'campaign': str(args.campaign)})
        return 2
    protocol = freeze_routing(args.campaign, args.output)
    if args.mode == 'labels':
        prepare_labels(args.campaign, args.output)
    if args.mode == 'replay':
        name = args.router
        if not name:
            parser.error('--router is required for replay')
        if name.startswith('fixed_'):
            router = FixedRouter(name.removeprefix('fixed_'))
        elif name == 'rules':
            router = RuleRouter()
        elif name in ('linear', 'mlp'):
            from ragscope.routing.learned import TfidfRouter
            router = TfidfRouter.load(args.router_artifact or args.output/name)
        elif name == 'lstm':
            from ragscope.routing.lstm import LSTMRouter
            router = LSTMRouter.load(args.router_artifact or args.output/name)
        else:
            parser.error('Laya requires its separately audited local SDK command; no implicit download/load')
        # This write happens before generation scores are loaded for evaluation.
        decisions = freeze_decisions(router, protocol, args.output, name)
        result = evaluate_decisions(args.campaign, protocol, decisions, args.output)
        print(json.dumps({split: group['summary'] for split, group in result['groups'].items()}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
