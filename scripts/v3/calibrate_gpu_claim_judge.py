"""Calibrate the actual GPU NLI stack before all-old scoring; no generation."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import sys
import time

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT)); sys.path.insert(0, str(PROJECT/'scripts/v3'))
from calibrate_claim_completion import fixtures
from filelock import FileLock
from ragscope.experiments.runtime import configure, require_ram
from ragscope.experiments.ledger import RecordStore, read_records
from ragscope.indexing.identity import artifact, atomic_json
from ragscope.v3.claim_metrics import proposition
from ragscope.v3.gpu import GpuSession
from ragscope.v3.nli import WindowedNLI


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    runtime = Path(configure(cpu_only=True)['home']).resolve()
    run = Path(args.campaign).resolve()
    if run.parent != runtime/'runs' or not run.name.startswith('v3_'):
        raise ValueError('Wrong campaign')
    require_ram(2*1024**3)
    with FileLock(str(run/'writer.lock'), timeout=0):
        manifest = json.loads((run/'METRIC_MODEL_MANIFEST.json').read_text('utf-8'))
        chosen = next(m for m in manifest['models'] if m['model'].startswith('MoritzLaurer/'))
        tasks = []
        for group in read_records(run/'JUDGE_CALIBRATION_CASES_V2.jsonl'):
            for direction in ('positive', 'negative'):
                tasks.append({'record_id': group['group_id']+':'+direction, 'family': 'declarative',
                    'premise': group['premise'], 'hypothesis': group[direction],
                    'expected': group[direction+'_relation']})
            if group.get('citation_negative_text'):
                tasks.append({'record_id': group['group_id']+':citation', 'family': 'declarative',
                    'premise': group['citation_negative_text'], 'hypothesis': group['positive'], 'expected': 'neutral'})
        fixed_grammar = []
        for i, original in enumerate(fixtures()):
            case = dict(original)
            # Construction-only repair: plural subject agreement; no selection by judge outcome.
            case['question'] = case['question'].replace('Is they ', 'Are they ').replace('Does they ', 'Do they ')
            case['premise'] = case['premise'].replace('They is ', 'They are ').replace('They does ', 'They do ')
            fixed_grammar.append(case)
            for direction in ('positive', 'negative'):
                auxiliary = proposition(case['question'], case[direction])
                if not auxiliary['eligible']:
                    raise ValueError('Constructed completion not supported by frozen rules')
                tasks.append({'record_id': 'qa:'+str(i)+':'+direction, 'family': 'completion',
                    'premise': case['premise'], 'hypothesis': auxiliary['text'],
                    'question': case['question'], 'raw_answer': case[direction],
                    'expected': 'entailment' if direction == 'positive' else 'contradiction'})
        config = {'model': chosen, 'runner': artifact(__file__), 'nli': artifact(PROJECT/'ragscope/v3/nli.py'),
            'completion': artifact(PROJECT/'ragscope/v3/claim_metrics.py'),
            'declarative_cases': artifact(run/'JUDGE_CALIBRATION_CASES_V2.jsonl'),
            'fixed_grammar_completion_cases': fixed_grammar, 'maximum': 512, 'threshold': .8,
            'device': 'cuda:0', 'dtype': 'float32', 'batch_size': 4,
            'torch': importlib.metadata.version('torch'), 'transformers': importlib.metadata.version('transformers'),
            'truth_source': 'RULE_CONSTRUCTION_NOT_HUMAN', 'new_llm_requests': 0}
        store = RecordStore(run/'NLI_GPU_SCORING_CALIBRATION.jsonl', config, max_records=len(tasks))
        pending = [t for t in tasks if t['record_id'] not in store.ids]
        started = time.monotonic()
        if pending:
            with GpuSession(run) as gpu:
                nli = WindowedNLI(chosen['directory'], device='cuda:0')
                try:
                    last_gpu_check = 0.0
                    for task in pending:
                        require_ram(2*1024**3)
                        if time.monotonic()-last_gpu_check >= 10:
                            gpu.check()
                            last_gpu_check = time.monotonic()
                        result = nli.classify(task['premise'], task['hypothesis'])
                        prediction = result['windows'][0]['argmax'] if len(result['windows']) == 1 else 'UNKNOWN'
                        store.add(task['record_id'], {**task, 'result': result,
                            'correct': prediction == task['expected'], 'human_review_status': 'NOT_REVIEWED'})
                finally:
                    nli.close()
        summaries = {}
        for family in ('declarative', 'completion'):
            rows = [r for r in store.records if r['family'] == family]
            expected = sum(t['family'] == family for t in tasks)
            summaries[family] = {'n': len(rows), 'expected': expected, 'correct': sum(r['correct'] for r in rows),
                                'accuracy': sum(r['correct'] for r in rows)/expected}
        complete = len(store.ids) == len(tasks)
        output = {'status': 'PASS_ENGINEERING_CALIBRATION' if complete and all(s['accuracy'] >= .9 for s in summaries.values()) else 'FAIL_ENGINEERING_CALIBRATION',
            'families': summaries, 'declarative_eligible': complete and summaries['declarative']['accuracy'] >= .9,
            'completion_eligible': complete and summaries['completion']['accuracy'] >= .9,
            'encoder_tokens': sum(r['result']['encoder_tokens'] for r in store.records),
            'elapsed_seconds': time.monotonic()-started, 'records': artifact(store.path), 'config': config,
            'human_review_status': 'NOT_REVIEWED', 'new_llm_requests': 0,
            'natural_language_accuracy': 'NOT_ESTABLISHED'}
        atomic_json(run/'NLI_GPU_SCORING_CALIBRATION.json', output)
        print(json.dumps({k: v for k, v in output.items() if k not in ('config', 'records')}))
        return 0 if complete else 2


if __name__ == '__main__':
    raise SystemExit(main())
