"""Separate answer polarity from verbosity; never replace official scores."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
import sqlite3
import sys

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
from ragscope.indexing.identity import artifact, atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', required=True)
    args = parser.parse_args()
    run = Path(args.campaign).resolve()
    if run.parent != Path('D:/RAGScope_Runtime/runs').resolve() or not run.name.startswith('v3_'):
        raise ValueError('Wrong campaign')
    recount = {r['record_id']: r for r in (json.loads(l) for l in (run/'RECOUNT_ALL_OLD.jsonl').read_text('utf-8').splitlines())}
    rows = []
    db = sqlite3.connect('file:'+str(run/'S0_EVIDENCE.sqlite')+'?mode=ro', uri=True)
    for (payload,) in db.execute('SELECT payload FROM records'):
        row = json.loads(payload)
        if (row['experiment_id'], row['method']) not in {('E2', 'R2'), ('E3', 'oracle')} or row['model_id'] not in {'qwen_2b', 'qwen_9b'}:
            continue
        metric = recount[row['record_id']]
        if metric['answer_type'] != 'yes_no' or not row['document_answerable']:
            continue
        references = {r.strip().lower().rstrip('.') for r in row['answer_refs']}
        if not references or not references <= {'yes', 'no'}:
            raise ValueError('Pure yes/no references are not pure polarity')
        match = re.match(r'^\s*(yes|no)\b', row['answer'], re.I)
        prefix = match.group(1).lower() if match else None
        rows.append({'record_id': row['record_id'], 'query_id': row['query_id'], 'experiment_id': row['experiment_id'],
            'method': row['method'], 'model_id': row['model_id'], 'answer': row['answer'], 'references': sorted(references),
            'prefix': prefix, 'prefix_reference_agreement': prefix in references,
            'legacy_token_f1': metric['legacy_token_f1'], 'failed': row['failed'], 'abstained': row['model_abstained']})
    db.close()
    groups = defaultdict(list)
    for row in rows:
        groups[(row['experiment_id'], row['method'], row['model_id'])].append(row)
    summaries = [{'experiment_id': key[0], 'method': key[1], 'model_id': key[2], 'n': len(group),
        'prefix_agreement_n': sum(r['prefix_reference_agreement'] for r in group),
        'prefix_agreement_rate': sum(r['prefix_reference_agreement'] for r in group)/len(group),
        'missing_prefix_n': sum(r['prefix'] is None for r in group),
        'ambiguous_reference_n': sum(len(r['references']) > 1 for r in group),
        'mean_legacy_word_f1': sum(r['legacy_token_f1'] for r in group)/len(group)} for key, group in sorted(groups.items())]
    output = {'status': 'PASS_DIAGNOSTIC_ONLY', 'groups': summaries, 'rows': rows,
        'sources': [artifact(run/'S0_EVIDENCE.sqlite'), artifact(run/'RECOUNT_ALL_OLD.jsonl'), artifact(__file__)],
        'metric_limit': 'Prefix agreement is not factual accuracy; following explanation may contradict it. No old metric or answer was changed.',
        'human_review_status': 'NOT_REVIEWED', 'new_llm_requests': 0}
    destination = run/'POLAR_PREFIX_DIAGNOSTIC.json'
    if destination.exists():
        saved = json.loads(destination.read_text('utf-8'))
        if saved != output:
            raise ValueError('Immutable diagnostic identity differs')
    else:
        atomic_json(destination, output)
    print(json.dumps({'status': output['status'], 'groups': summaries, 'new_llm_requests': 0}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
