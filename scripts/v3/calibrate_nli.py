"""Actually run the frozen NLI calibration; no human review or text generation."""
from __future__ import annotations
import argparse
from collections import Counter
from contextlib import nullcontext
import importlib.metadata
import json
from pathlib import Path
import sys
import time

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
from filelock import FileLock
from ragscope.experiments.runtime import configure, require_ram
from ragscope.experiments.ledger import RecordStore
from ragscope.indexing.identity import artifact, atomic_json
from ragscope.v3.gpu import GpuSession
from ragscope.v3.nli import WindowedNLI


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    args = parser.parse_args()
    runtime = Path(configure(cpu_only=True)['home']).resolve()
    run = Path(args.campaign).resolve()
    if not run.is_relative_to(runtime / 'runs') or not run.name.startswith('v3_'):
        raise ValueError('Wrong campaign')
    require_ram(2 * 1024**3)
    with FileLock(str(run / 'writer.lock'), timeout=0):
        models = json.loads((run / 'METRIC_MODEL_MANIFEST.json').read_text('utf-8'))
        chosen = next(m for m in models['models'] if m['model'].startswith('MoritzLaurer/'))
        cases_path = run / 'JUDGE_CALIBRATION_CASES_V2.jsonl'
        with cases_path.open(encoding='utf-8') as handle:
            groups = [json.loads(line) for line in handle]
        tests = []
        for group in groups:
            for tag in ('positive', 'negative'):
                tests.append({'record_id': group['group_id'] + ':' + tag, 'group_id': group['group_id'],
                    'category': group['category'], 'premise': group['premise'], 'hypothesis': group[tag],
                    'expected_relation': group[tag + '_relation']})
            if group.get('citation_negative_text'):
                tests.append({'record_id': group['group_id'] + ':wrong_citation', 'group_id': group['group_id'],
                    'category': 'citation_mismatch', 'premise': group['citation_negative_text'],
                    'hypothesis': group['positive'], 'expected_relation': 'neutral'})
        config = {'cases': artifact(cases_path), 'model': chosen, 'device': args.device,
            'maximum': 512, 'stride': 64, 'threshold': 0.8, 'dtype': 'float32', 'batch_size': 4,
            'implementation': artifact(PROJECT / 'ragscope/v3/nli.py'), 'runner': artifact(__file__),
            'transformers': importlib.metadata.version('transformers'), 'torch': importlib.metadata.version('torch'),
            'human_review_status': 'NOT_REVIEWED', 'new_llm_requests': 0}
        store = RecordStore(run / ('NLI_CALIBRATION_' + args.device + '.jsonl'), config, max_records=len(tests))
        pending = [test for test in tests if test['record_id'] not in store.ids]
        started = time.monotonic()
        if pending:
            with GpuSession(run) if args.device == 'cuda' else nullcontext() as gpu:
                nli = WindowedNLI(chosen['directory'], device='cuda:0' if gpu else 'cpu')
                try:
                    for index, test in enumerate(pending):
                        require_ram(2 * 1024**3)
                        if gpu:
                            gpu.check()
                        result = nli.classify(test['premise'], test['hypothesis'])
                        prediction = result['windows'][0]['argmax'] if len(result['windows']) == 1 else 'UNKNOWN'
                        store.add(test['record_id'], {**test, 'prediction': prediction,
                            'correct': prediction == test['expected_relation'], 'nli_result': result,
                            'truth_source': 'RULE_CONSTRUCTION_NOT_HUMAN', 'human_review_status': 'NOT_REVIEWED'})
                        if (index+1) % 10 == 0:
                            print(json.dumps({'event': 'NLI_CALIBRATION_PROGRESS', 'completed': len(store.ids),
                                              'expected': len(tests)}), flush=True)
                finally:
                    nli.close()
        passed = sum(row['correct'] for row in store.records)
        per_group = {}
        for row in store.records:
            per_group.setdefault(row['group_id'], []).append(row['correct'])
        group_passed = sum(all(values) for values in per_group.values())
        complete = len(store.records) == len(tests)
        gate = complete and group_passed / len(groups) >= 0.90
        output = {'status': 'PASS' if gate else 'FAIL_CALIBRATION' if complete else 'INCOMPLETE',
            'completed_tests': len(store.records), 'expected_tests': len(tests), 'correct_tests': passed,
            'pair_accuracy': passed/len(store.records) if store.records else None,
            'group_n': len(groups), 'fully_correct_group_n': group_passed,
            'group_accuracy': group_passed/len(groups), 'engineering_gate_90_percent_passed': gate,
            'by_category': {c: {'n': sum(r['category']==c for r in store.records),
                'correct': sum(r['correct'] for r in store.records if r['category']==c)} for c in sorted({r['category'] for r in store.records})},
            'encoder_tokens': sum(r['nli_result']['encoder_tokens'] for r in store.records),
            'elapsed_seconds_this_invocation': time.monotonic()-started, 'records': artifact(store.path),
            'config': config, 'calibration_type': 'engineering_constructed_not_natural_accuracy',
            'human_review_status': 'NOT_REVIEWED', 'new_llm_requests': 0,
            'automatic_promotion_eligible_from_this_judge': gate}
        atomic_json(run / ('JUDGE_CALIBRATION_NLI_' + args.device + '.json'), output)
        print(json.dumps({k: v for k, v in output.items() if k not in ('config','records','by_category')}))
        return 0 if complete else 2


if __name__ == '__main__':
    raise SystemExit(main())
