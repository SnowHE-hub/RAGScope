"""Real NLI checks for the narrow, reference-blind QA completion rules."""
from __future__ import annotations
import argparse
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
from ragscope.v3.claim_metrics import proposition
from ragscope.v3.nli import WindowedNLI


def fixtures():
    cases = []
    for subject in ('they', 'the model', 'the method', 'the experiment', 'the system'):
        for auxiliary, remainder in (('did', 'use dataset Delta'), ('can', 'process English text'),
                                     ('is', 'available to researchers'), ('does', 'include a tokenizer')):
            question = f'{auxiliary.title()} {subject} {remainder}?'
            statement = f'{subject.title()} {auxiliary} not {remainder}.'
            cases.append({'question': question, 'premise': statement, 'positive': 'No', 'negative': 'Yes'})
        for number in range(128, 132):
            cases.append({'question': f'How many samples did {subject} use?',
                          'premise': f'{subject.title()} did use exactly {number} samples.',
                          'positive': str(number), 'negative': str(number+1)})
    return cases


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
    job = run/'nli_completion_calibration'
    job.mkdir(exist_ok=True)
    # This isolated CPU job never writes atomization, parent writer state, or GPU leases.
    with FileLock(str(job/'writer.lock'), timeout=0):
        manifest = json.loads((run/'METRIC_MODEL_MANIFEST.json').read_text('utf-8'))
        chosen = next(m for m in manifest['models'] if m['model'].startswith('MoritzLaurer/'))
        config = {'model': chosen, 'runner': artifact(__file__),
            'completion': artifact(PROJECT/'ragscope/v3/claim_metrics.py'),
            'nli': artifact(PROJECT/'ragscope/v3/nli.py'), 'device': 'cpu', 'threshold': .8,
            'source_truth': 'SYNTHETIC_RULE_CONSTRUCTION_NOT_HUMAN', 'fixture_n': 40,
            'fixtures': fixtures(), 'reference_or_context_used_in_completion': False}
        store = RecordStore(job/'records.jsonl', config, max_records=80)
        tasks = [(str(i)+':'+direction, case, direction) for i, case in enumerate(fixtures())
                 for direction in ('positive', 'negative')]
        pending = [t for t in tasks if t[0] not in store.ids]
        started = time.monotonic()
        if pending:
            nli = WindowedNLI(chosen['directory'])
            try:
                for rid, case, direction in pending:
                    require_ram(2*1024**3)
                    auxiliary = proposition(case['question'], case[direction])
                    expected = 'entailment' if direction == 'positive' else 'contradiction'
                    result = nli.classify(case['premise'], auxiliary['text']) if auxiliary['eligible'] else {'windows': [], 'encoder_tokens': 0, 'verdict': 'UNKNOWN'}
                    prediction = result['windows'][0]['argmax'] if len(result['windows']) == 1 else 'UNKNOWN'
                    store.add(rid, {'question': case['question'], 'raw_answer': case[direction],
                        'auxiliary': auxiliary, 'result': result, 'expected': expected, 'prediction': prediction,
                        'correct': prediction == expected, 'human_review_status': 'NOT_REVIEWED'})
            finally:
                nli.close()
        correct = sum(r['correct'] for r in store.records)
        complete = len(store.records) == len(tasks)
        gate = complete and correct/len(tasks) >= .90
        output = {'status': 'PASS_ENGINEERING_CALIBRATION' if gate else 'FAIL_ENGINEERING_CALIBRATION',
            'completed': len(store.records), 'expected': len(tasks), 'correct': correct,
            'accuracy': correct/len(tasks), 'completion_eligible': gate,
            'confident_n': sum(r['result']['verdict'] != 'UNKNOWN' for r in store.records),
            'encoder_tokens': sum(r['result']['encoder_tokens'] for r in store.records),
            'elapsed_seconds': time.monotonic()-started, 'records': artifact(store.path),
            'human_review_status': 'NOT_REVIEWED', 'new_llm_requests': 0,
            'limitation': 'Narrow constructed English templates; not natural QA or human accuracy'}
        atomic_json(job/'SUMMARY.json', output)
        print(json.dumps(output))
        return 0 if complete else 2


if __name__ == '__main__':
    raise SystemExit(main())
