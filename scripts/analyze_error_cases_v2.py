"""Export actual generation error cases, preserving uncertain interpretations."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
from freeze_generation_v2 import verified_corpus
from ragscope.experiments.error_cases import build_error_cases
from ragscope.experiments.generation_campaign import load_campaign


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    manifest, _ = load_campaign(args.campaign)
    corpus = Path(manifest['provenance']['spec']['corpus_dir'])
    _, units, _ = verified_corpus(corpus)
    _, squad, _ = verified_corpus(corpus/'squad_e4'); units.update(squad)
    result = build_error_cases(args.campaign, args.output, units)
    print(result['status'], result['n_observed'], '/', result['n_expected'])
    return 0 if result['status'] == 'COMPLETE' else 2


if __name__ == '__main__':
    raise SystemExit(main())
