"""Join independent sufficiency proxies to retained LoRA and blinded populations."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
from ragscope.experiments.ledger import read_records
from ragscope.indexing.identity import artifact, atomic_json, canonical_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', required=True)
    args = parser.parse_args()
    run = Path(args.campaign).resolve()
    if run.parent != Path('D:/RAGScope_Runtime/runs').resolve() or not run.name.startswith('v3_'):
        raise ValueError('Wrong campaign')
    verdicts = {r['record_id']: r for r in read_records(run/'CONTEXT_SUFFICIENCY_PROXY.jsonl')}
    joins = read_records(run/'CONTEXT_JUDGEMENT_PRIVATE_JOIN.jsonl')
    if len(verdicts) != 117 or any(r['answers_or_references_supplied'] is not False for r in verdicts.values()):
        raise ValueError('Independent context-only judgment coverage or boundary differs')
    pairs = {r['query_id']: r for r in read_records(run/'LORA_100_PAIRS.jsonl')}
    details = []
    for row in joins:
        if row['population'] != 'LORA_NEW_REFUSAL23':
            continue
        pair, verdict = pairs[row['query_id']], verdicts[row['context_judge_id']]
        if pair['group'] != 'NEW_ABSTENTION' or not pair['adapted']['document_answerable']:
            raise ValueError('Wrong LoRA subgroup')
        label = verdict['visible_context_sufficient']
        details.append({'query_id': row['query_id'], 'doc_id': pair['doc_id'],
            'document_answerable': True, 'model_abstained': True, 'visible_context_sufficient': label,
            'refusal_reason_proxy': 'OVER_ABSTENTION_PROXY' if label == 'YES' else 'REASONABLE_GIVEN_INSUFFICIENT_CONTEXT_PROXY' if label == 'NO' else 'UNKNOWN',
            'context_judge_id': row['context_judge_id'], 'context_judge_record_hash': verdict['record_hash'],
            'baseline_answer': pair['baseline']['answer'], 'adapted_answer': pair['adapted']['answer'],
            'delta_legacy_token_f1': pair['delta_legacy_token_f1'], 'human_review_status': 'NOT_REVIEWED'})
    if len(details) != 23:
        raise ValueError('New-refusal denominator differs')
    path = run/'LORA_23_REFUSAL_CONTEXT_PROXY.jsonl'
    if not path.exists():
        with path.open('x', encoding='utf-8', newline='\n') as handle:
            for row in details: handle.write(canonical_json(row)+'\n')
    elif read_records(path) != details:
        raise ValueError('Frozen refusal assessment differs')
    blind = {r['blind_id']: r for r in read_records(run/'BLIND_REVIEW_PACKAGE.jsonl')}
    blind_rows = []
    for row in joins:
        if row['population'] == 'BLIND120':
            verdict = verdicts[row['context_judge_id']]
            blind_rows.append({'blind_id': row['blind_id'], 'visible_context_sufficient': verdict['visible_context_sufficient'],
                'sample_weight': blind[row['blind_id']]['sample_weight'], 'sampling_probability': blind[row['blind_id']]['sampling_probability'],
                'judge_record_hash': verdict['record_hash'], 'human_review_status': 'NOT_REVIEWED'})
    path_blind = run/'BLIND_REVIEW_CONTEXT_PROXY.jsonl'
    if not path_blind.exists():
        with path_blind.open('x', encoding='utf-8', newline='\n') as handle:
            for row in blind_rows: handle.write(canonical_json(row)+'\n')
    elif read_records(path_blind) != blind_rows:
        raise ValueError('Frozen blinded assessment differs')
    counts = Counter(r['visible_context_sufficient'] for r in details)
    result = {'status': 'PASS_INDEPENDENT_CONTEXT_REFUSAL_DIAGNOSIS', 'new_refusals': 23,
        'new_refusal_context_counts': dict(counts), 'blind_context_rows': len(blind_rows),
        'unique_context_count': 117, 'judge_calibration': artifact(run/'LOCAL_JUDGE_CALIBRATION.json'),
        'source': artifact(run/'CONTEXT_SUFFICIENCY_PROXY.jsonl'), 'refusal_details': artifact(path),
        'blind_context_details': artifact(path_blind), 'runner': artifact(__file__),
        'human_review_status': 'NOT_REVIEWED', 'new_llm_requests': 0,
        'interpretation': 'AI sufficiency proxy, not human correctness; material insufficiency is not a dataset-unanswerable label'}
    atomic_json(run/'CONTEXT_DIAGNOSIS.json', result)
    print(json.dumps({k: v for k, v in result.items() if k not in ('source', 'refusal_details', 'blind_context_details', 'runner', 'judge_calibration')}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
