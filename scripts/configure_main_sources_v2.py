"""Write an explicit main spec only after accepted GPU ROUTE source handoff."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.freeze_generation_v2 import default_spec
from ragscope.experiments.rerank_sources import accepted_gpu_sources
from ragscope.indexing.identity import atomic_json


def configure(home, campaign, output):
    accepted = accepted_gpu_sources(home, campaign)
    if accepted['n_failed']:
        raise ValueError('Failed GPU context rows need explicit blocking, not silent replacement')
    spec = default_spec(home, campaign)
    spec['context_sources'].update({
        'ROUTE_TRAIN:hybrid_rerank': str(accepted['paths']['ROUTE_CONTEXT_TRAIN']),
        'ROUTE_DEV:hybrid_rerank': str(accepted['paths']['ROUTE_CONTEXT_DEV'])})
    spec['gpu_r3_source_acceptance'] = {'artifacts': accepted['evidence'],
                                       'comparison_limit': accepted['comparison_limit']}
    if Path(output).exists():
        raise FileExistsError('Preserve an existing explicit main spec')
    atomic_json(Path(output), spec)
    return spec


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--home', type=Path, default=Path('D:/RAGScope_Runtime'))
    parser.add_argument('--campaign', default='20261014_v2')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    configure(args.home, args.campaign, args.output)
    print('EXPLICIT_MAIN_SPEC_WRITTEN_NO_MODEL_CALLS')
