"""One deterministic source-format repair, preserving teacher failures and words."""
import argparse
import copy
import json
from pathlib import Path
import re
import sys

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
from ragscope.experiments.ledger import read_records
from ragscope.indexing.identity import artifact, atomic_json, canonical_json
from ragscope.v3.task_references import validate_reference


def source_span(source, quote):
    if quote in source:
        return quote
    normalized, offsets = [], []
    for match in re.finditer(r'\s+|\S', source):
        normalized.append(' ' if match.group().isspace() else match.group())
        offsets.append((match.start(), match.end()))
    target = re.sub(r'\s+', ' ', quote)
    start = ''.join(normalized).find(target)
    if start < 0:
        raise ValueError('No whitespace-only source match; do not change words or punctuation')
    return source[offsets[start][0]:offsets[start+len(target)-1][1]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', required=True)
    args = parser.parse_args()
    run = Path(args.campaign).resolve()
    if run.parent != Path('D:/RAGScope_Runtime/runs').resolve() or not run.name.startswith('v3_'):
        raise ValueError('Wrong campaign')
    receipt = run/'H_TASKS_SOURCE_REPAIR.json'
    if receipt.exists():
        saved = json.loads(receipt.read_text('utf-8'))
        if saved['runner'] != artifact(__file__) or any(artifact(o['path']) != o for o in saved['outputs']):
            raise ValueError('Frozen repair identity differs')
        print(json.dumps({'status': 'REUSED_WHITESPACE_SOURCE_REPAIR'})); return 0
    papers = {p['source_id']: p for p in read_records(run/'H_tasks.source_documents.jsonl')}
    rows = read_records(run/'splits/evaluator_only/H_tasks_teacher.jsonl')
    successes, failures = [], []
    for row in rows:
        response = row.get('raw_response') or {}
        try:
            parsed = json.loads(response['content'])
        except Exception as exc:
            failures.append({'source_id': row['source_id'], 'kind': 'raw_response', 'error': str(exc)})
            continue
        for field, kind in (('summary', 'summarize'), ('comparison', 'compare')):
            if parsed.get(field) is None:
                continue
            task = copy.deepcopy(parsed[field])
            try:
                original_quotes = [p['quote'] for p in task['keypoints']]
                for point in task['keypoints']:
                    point['quote'] = source_span(papers[row['source_id']]['text'], point['quote'])
                public, gold = validate_reference(papers[row['source_id']], task, kind)
                gold['source_format_repair'] = 'whitespace collapse only; words and punctuation unchanged'
                gold['original_teacher_quotes'] = original_quotes
                successes.append({'public': public, 'gold': gold, 'teacher_record_hash': row['record_hash']})
            except Exception as exc:
                failures.append({'source_id': row['source_id'], 'kind': kind, 'error': str(exc),
                                 'teacher_record_hash': row['record_hash']})
    outputs = []
    for path, values in ((run/'splits/H_tasks.questions_v2.jsonl', [t['public'] for t in successes]),
        (run/'splits/evaluator_only/H_tasks.gold_v2.jsonl', [t['gold'] for t in successes]),
        (run/'H_tasks_source_validation_v2.jsonl', failures)):
        with path.open('x', encoding='utf-8', newline='\n') as handle:
            for value in values: handle.write(canonical_json(value)+'\n')
        outputs.append(artifact(path))
    result = {'status': 'PASS_BOUNDED_SOURCE_FORMAT_REPAIR', 'runner': artifact(__file__),
        'teacher': artifact(run/'splits/evaluator_only/H_tasks_teacher.jsonl'), 'outputs': outputs,
        'valid_task_n': len(successes), 'failed_task_n': len(failures), 'original_teacher_attempts': 30,
        'old_task_population_retained': True, 'word_or_punctuation_change_forbidden': True,
        'human_review_status': 'NOT_REVIEWED', 'operator_blindness': 'NOT_OPERATOR_BLIND', 'new_llm_requests': 0}
    atomic_json(receipt, result)
    print(json.dumps({k: v for k, v in result.items() if k not in ('runner', 'teacher', 'outputs')})); return 0


if __name__ == '__main__':
    raise SystemExit(main())
