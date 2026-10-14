"""Freeze old-answer-only grouping before any paid-in-budget local LM call."""
from __future__ import annotations
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sqlite3
import sys

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
from filelock import FileLock
from ragscope.experiments.runtime import configure, require_ram
from ragscope.indexing.identity import artifact, atomic_json, canonical_json, signature
from ragscope.v3.atomization import ATOMIZATION_SYSTEM
from ragscope.v3.splits import assert_inference_safe


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
        receipt = run / 'ATOMIZATION_INPUT_MANIFEST.json'
        if receipt.exists():
            saved = json.loads(receipt.read_text('utf-8'))
            if not args.resume or saved['preparer'] != artifact(__file__):
                raise ValueError('Frozen grouping identity differs')
            for ref in saved['outputs']:
                if artifact(ref['path']) != ref:
                    raise ValueError('Frozen answer grouping differs')
            print(json.dumps({'status': 'REUSED_ATOMIZATION_INPUTS'}))
            return 0
        db = sqlite3.connect(run / 'S0_EVIDENCE.sqlite')
        rows = [json.loads(payload) for (payload,) in db.execute('SELECT payload FROM records')]
        db.close()
        with (run / 'LORA_100_PAIRS.jsonl').open(encoding='utf-8') as handle:
            for line in handle:
                pair = json.loads(line)
                for condition in ('baseline', 'adapted'):
                    row = pair[condition]
                    row['record_id'] = 'LORA:' + condition + ':' + row['query_id']
                    rows.append(row)
        questions, groups, private = {}, defaultdict(dict), []
        for row in rows:
            qid, text = row['query_id'], row['answer']
            if qid in questions and questions[qid] != row['question']:
                raise ValueError('Same question ID has different text')
            questions[qid] = row['question']
            if row['failed'] or row['model_abstained'] or not text.strip():
                private.append({'record_id': row['record_id'], 'query_id': qid, 'source_id': None,
                    'status': 'FAILED_OR_ABSTAINED_OR_EMPTY_RETAINED_IN_METRIC_DENOMINATOR'})
                continue
            source_id = 'TEXT_' + signature([qid, text])[:20]
            groups[qid][source_id] = text
            private.append({'record_id': row['record_id'], 'query_id': qid, 'source_id': source_id,
                            'status': 'PENDING_SEGMENTATION', 'source_record_hash': row['source_record_hash']})
        inputs = []
        for qid in sorted(groups):
            value = {'group_id': 'ATOM_' + signature([qid])[:20], 'query_id': qid,
                'question': questions[qid], 'texts': [{'source_id': key, 'text': text} for key, text in sorted(groups[qid].items())]}
            assert_inference_safe(value)
            value['input_hash'] = signature(value)
            inputs.append(value)
        paths = []
        for name, values in [('ATOMIZATION_INPUTS.jsonl', inputs), ('ATOMIZATION_PRIVATE_JOIN.jsonl', private)]:
            path = run / name
            with path.open('x', encoding='utf-8', newline='\n') as handle:
                for row in values:
                    handle.write(canonical_json(row) + '\n')
            paths.append(artifact(path))
        manifest = {'status': 'FROZEN_INPUTS_MODEL_CALLS_NOT_RUN', 'preparer': artifact(__file__),
            'validator': artifact(PROJECT / 'ragscope/v3/atomization.py'), 'system_prompt': ATOMIZATION_SYSTEM,
            'old_record_n': len(rows), 'question_groups': len(inputs),
            'unique_answer_texts': sum(len(g) for g in groups.values()), 'outputs': paths,
            'per_record_provenance_preserved': True, 'gold_or_reference_supplied_to_segmenter': False,
            'actual_new_llm_requests': 0, 'needs_actual_template_budget_before_call': True,
            'generation_stage': 'S0_S1_evidence_metrics', 'human_review_status': 'NOT_REVIEWED',
            'note': 'Segmentation returns offsets only. No rewritten old answer is used for word or semantic score.'}
        atomic_json(receipt, manifest)
        print(json.dumps({k: manifest[k] for k in ['status','old_record_n','question_groups','unique_answer_texts','actual_new_llm_requests']}))
        return 0


if __name__ == '__main__':
    raise SystemExit(main())
