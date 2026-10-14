"""Freeze question/context-only sufficiency inputs for blind cases and LoRA refusals."""
from __future__ import annotations
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import sqlite3
import sys

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
from filelock import FileLock
from ragscope.experiments.runtime import configure, require_ram
from ragscope.indexing.identity import artifact, atomic_json, canonical_json, signature
from ragscope.v3.semantic_inputs import blinded_context_pieces
from ragscope.v3.splits import assert_inference_safe


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
    with FileLock(str(run/'context_input_preparation.lock'), timeout=0):
        receipt = run/'CONTEXT_JUDGEMENT_INPUTS.json'
        if receipt.exists():
            saved = json.loads(receipt.read_text('utf-8'))
            if not args.resume or saved['runner'] != artifact(__file__) or any(artifact(o['path']) != o for o in saved['outputs']):
                raise ValueError('Context input identity changed')
            print(json.dumps({'status': 'REUSED_CONTEXT_ONLY_INPUTS'})); return 0
        blind = [json.loads(l) for l in (run/'BLIND_REVIEW_PACKAGE.jsonl').read_text('utf-8').splitlines()]
        companion = {r['blind_id']: r for r in (json.loads(l) for l in (run/'BLIND_REVIEW_CONTEXTS.jsonl').read_text('utf-8').splitlines())}
        pairs = [json.loads(l) for l in (run/'LORA_100_PAIRS.jsonl').read_text('utf-8').splitlines()]
        lora_path = runtime/'runs/20261014_v2_lora/train_2b_1024_seed42/paired_test100/FROZEN_EVALUATION_INPUTS.json'
        lora_inputs = {r['query']['query_id']: r for r in json.loads(lora_path.read_text('utf-8'))['inputs']}
        groups, joins = {}, []
        def add(question, pieces, source):
            row = {'question': question, 'context_pieces': pieces}
            assert_inference_safe(row)
            rid = 'SUFF_'+signature(row)[:24]
            row.update(record_id=rid, input_hash=signature(row))
            groups[rid] = row
            joins.append({**source, 'context_judge_id': rid})
        for row in blind:
            pieces = companion[row['blind_id']]['context_pieces']
            add(row['question'], pieces, {'population': 'BLIND120', 'blind_id': row['blind_id']})
        refusals = [p for p in pairs if p['group'] == 'NEW_ABSTENTION']
        if len(refusals) != 23:
            raise ValueError('Expected 23 retained new LoRA refusals')
        for pair in refusals:
            original = lora_inputs[pair['query_id']]
            if original['context_bundle']['context_hash'] != pair['adapted']['context_hash']:
                raise ValueError('LoRA actual context identity differs')
            add(pair['adapted']['question'], blinded_context_pieces(original['context_bundle']),
                {'population': 'LORA_NEW_REFUSAL23', 'query_id': pair['query_id']})
        outputs = []
        for filename, rows in (('CONTEXT_ONLY_JUDGEMENT_INPUTS.jsonl', sorted(groups.values(), key=lambda r: r['record_id'])),
                               ('CONTEXT_JUDGEMENT_PRIVATE_JOIN.jsonl', joins)):
            path = run/filename
            with path.open('x', encoding='utf-8', newline='\n') as handle:
                for row in rows:
                    handle.write(canonical_json(row)+'\n')
                handle.flush(); os.fsync(handle.fileno())
            outputs.append(artifact(path))
        result = {'status': 'PASS_CONTEXT_ONLY_INPUTS_FROZEN', 'runner': artifact(__file__),
            'sources': [artifact(run/'BLIND_REVIEW_PACKAGE.jsonl'), artifact(run/'BLIND_REVIEW_CONTEXTS.jsonl'),
                        artifact(run/'LORA_100_PAIRS.jsonl'), artifact(lora_path)],
            'blind_n': 120, 'lora_new_refusal_n': 23, 'unique_question_context_n': len(groups),
            'outputs': outputs, 'answers_references_answerability_and_model_names_in_judge_input': False,
            'human_review_status': 'NOT_REVIEWED', 'new_llm_requests': 0}
        atomic_json(receipt, result)
        print(json.dumps({k: v for k, v in result.items() if k not in ('runner', 'sources', 'outputs')}))
        return 0


if __name__ == '__main__':
    raise SystemExit(main())
