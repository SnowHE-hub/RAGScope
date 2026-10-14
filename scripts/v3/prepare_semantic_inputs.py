"""Prepare a corrected calibration version and full blinded context companion."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sqlite3
import sys

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
from filelock import FileLock
from ragscope.experiments.runtime import configure, require_ram
from ragscope.indexing.identity import artifact, atomic_json, canonical_json
from ragscope.v3.semantic_inputs import blinded_context_pieces, validated_calibration_groups


def read_lines(path):
    with path.open(encoding='utf-8') as handle:
        return [json.loads(line) for line in handle]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    runtime = Path(configure(cpu_only=True)['home']).resolve()
    run = Path(args.campaign).resolve()
    if not run.is_relative_to(runtime / 'runs') or not run.name.startswith('v3_'):
        raise ValueError('Wrong campaign')
    require_ram(2 * 1024**3)
    with FileLock(str(run / 'writer.lock'), timeout=0):
        receipt = run / 'S1_SEMANTIC_INPUTS_RECEIPT.json'
        if receipt.exists():
            saved = json.loads(receipt.read_text('utf-8'))
            if not args.resume or saved['script'] != artifact(__file__):
                raise ValueError('Semantic preparation identity differs')
            for ref in saved['outputs']:
                if artifact(ref['path']) != ref:
                    raise ValueError('Blinded/calibration artifact changed')
            print(json.dumps({'status': 'REUSED_SEMANTIC_INPUTS'}))
            return 0
        cases = validated_calibration_groups(read_lines(run / 'JUDGE_CALIBRATION_CASES.jsonl'))
        if len(cases) != 60:
            raise ValueError('Calibration population differs')
        db = sqlite3.connect(run / 'S0_EVIDENCE.sqlite')
        context_to_input = {}
        for (payload,) in db.execute('SELECT payload FROM records'):
            record = json.loads(payload)
            context_to_input[record['context_hash']] = record['input_id']
        companion = []
        for blind in read_lines(run / 'BLIND_REVIEW_PACKAGE.jsonl'):
            iid = context_to_input[blind['context_hash']]
            payload = db.execute('SELECT payload FROM inputs WHERE input_id=?', (iid,)).fetchone()[0]
            bundle = json.loads(payload)['context_bundle']
            companion.append({'blind_id': blind['blind_id'], 'context_hash': blind['context_hash'],
                              'context_pieces': blinded_context_pieces(bundle)})
        db.close()
        outputs = []
        for name, rows in [('JUDGE_CALIBRATION_CASES_V2.jsonl', cases), ('BLIND_REVIEW_CONTEXTS.jsonl', companion)]:
            path = run / name
            with path.open('x', encoding='utf-8', newline='\n') as handle:
                for row in rows:
                    handle.write(canonical_json(row) + '\n')
            outputs.append(artifact(path))
        atomic_json(receipt, {'status': 'PASS_INPUT_PREPARATION_ONLY', 'script': artifact(__file__),
            'implementation': artifact(PROJECT / 'ragscope/v3/semantic_inputs.py'), 'outputs': outputs,
            'calibration_groups': len(cases), 'blind_context_n': len(companion),
            'old_v1_cases_preserved': True, 'judge_run': False, 'new_llm_requests': 0,
            'human_review_status': 'NOT_REVIEWED', 'actual_exit_code': 0})
        print(json.dumps({'status': 'SEMANTIC_INPUTS_PREPARED', 'calibration_groups': len(cases),
                          'blind_context_n': len(companion), 'new_llm_requests': 0}))
        return 0


if __name__ == '__main__':
    raise SystemExit(main())
