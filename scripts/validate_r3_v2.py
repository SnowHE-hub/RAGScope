"""Validate all five real R3 groups after the model worker exits."""
from pathlib import Path
import argparse
import json
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ragscope.experiments.rerank_validation import validate_campaign

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--home', default='D:/RAGScope_Runtime')
    parser.add_argument('--campaign', default='20261014_v2')
    parser.add_argument('--prefix', help='Accept one complete immutable group while other groups still run; never marks all660 complete')
    parser.add_argument('--gpu', action='store_true', help='Validate explicit GPU global100+ROUTE200/60 migration, preserving CPU evidence')
    args = parser.parse_args()
    print(json.dumps(validate_campaign(args.home, args.campaign, prefix=args.prefix, gpu=args.gpu), ensure_ascii=False))
