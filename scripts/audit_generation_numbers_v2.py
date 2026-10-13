"""Independent reverse audit: recompute reported numbers from raw outputs.

No production generation/evaluation/analysis functions are imported. Overlap is
recomputed from raw parsed text and every legitimate reference; evidence uses
the separate source-coordinate auditor, not saved metric values.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import re
import string
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.audit_retrieval_v2 import independently_score, quantile, rows, sha, signature


def normalized_words(value):
    chars = ''.join(c.lower() for c in value if c not in string.punctuation)
    return re.sub(r'\b(a|an|the)\b', ' ', chars).split()


def references(query):
    result = []
    for ref in query.get('answer_refs', []):
        if isinstance(ref, str): result.append(ref)
        elif not ref.get('unanswerable'):
            if ref.get('yes_no') is not None: result.append('yes' if ref['yes_no'] else 'no')
            elif ref.get('free_form_answer'): result.append(ref['free_form_answer'])
            elif ref.get('extractive_spans'): result.append(', '.join(ref['extractive_spans']))
            elif isinstance(ref.get('text'), str): result.append(ref['text'])
            elif isinstance(ref.get('answer'), str): result.append(ref['answer'])
    return result


def overlap(row, query):
    if query['answerability'] == 'unanswerable': return None, None
    if row.get('error') or row.get('abstain'): return 0., 0.
    pred = normalized_words(row.get('answer', ''))
    f1, exact = [], []
    for ref in references(query):
        gold = normalized_words(ref)
        exact.append(float(pred == gold))
        if not pred or not gold: f1.append(float(pred == gold))
        else:
            gold_counts = Counter(gold)
            common = 0
            for word in pred:
                if gold_counts[word] > 0:
                    common += 1; gold_counts[word] -= 1
            f1.append(2*common/(len(pred)+len(gold)))
    return max(f1, default=0.), max(exact, default=0.)


def independently_load(directory):
    manifest = json.loads((directory/'EXPERIMENT_MANIFEST.json').read_text(encoding='utf-8'))
    body = {k: v for k, v in manifest.items() if k != 'manifest_hash'}
    if signature(body) != manifest['manifest_hash']:
        raise ValueError('Independent manifest hash mismatch')
    inputs = json.loads((directory/'FROZEN_INPUTS.json').read_text(encoding='utf-8'))
    if set(inputs) != set(manifest['input_hashes']):
        raise ValueError('Independent input membership mismatch')
    for key, value in inputs.items():
        body = {k: v for k, v in value.items() if k != 'input_hash'}
        if signature(body) != value['input_hash'] or value['input_hash'] != manifest['input_hashes'][key]:
            raise ValueError('Independent frozen input hash mismatch')
    return manifest, inputs


def independently_owned_sources(directory, manifest, inputs):
    """Separate implementation of provenance/ownership checks, not production API."""
    directory = directory.resolve()
    tasks = {row['task_id']: row for row in manifest['tasks']}
    if len(tasks) != len(manifest['tasks']):
        raise ValueError('Duplicate frozen task')
    owned, visited, sources = set(), {directory}, []
    for external in manifest.get('provenance', {}).get('external_campaigns', []):
        item = external['manifest']; path = Path(item['path']).resolve()
        if path.name != 'EXPERIMENT_MANIFEST.json' or path.parent in visited or sha(path) != item['sha256']:
            raise ValueError('Independent external artifact/ownership mismatch')
        visited.add(path.parent)
        other, other_inputs = independently_load(path.parent)
        if other.get('provenance', {}).get('external_campaigns'):
            raise ValueError('Nested external source rejected')
        for field in ('generation_config', 'metric_version', 'seed', 'reference_set_rule', 'parse_repair_attempts'):
            if other.get(field) != manifest.get(field):
                raise ValueError('External protocol mismatch')
        for field in ('prompt_version', 'prompt_hash', 'output_constraint_version', 'backend_runtime_config'):
            if other.get('provenance', {}).get(field) != manifest.get('provenance', {}).get(field):
                raise ValueError('External prompt/backend protocol mismatch')
        task_ids = set()
        for row in other['tasks']:
            key = row['task_id']
            if key in owned or key in task_ids or row != tasks.get(key):
                raise ValueError('External task duplicated or changed')
            if other_inputs[row['input_id']] != inputs[row['input_id']]:
                raise ValueError('External input changed')
            if other['models'][row['model_id']] != manifest['models'][row['model_id']]:
                raise ValueError('External model changed')
            task_ids.add(key)
        if not task_ids:
            raise ValueError('Empty external source')
        owned.update(task_ids)
        sources.append((path.parent, other, other_inputs, task_ids))
    return [(directory, manifest, inputs, set(tasks)-owned), *sources]


def audit(campaign, analysis_path, corpus, output):
    report = json.loads(analysis_path.read_text(encoding='utf-8'))
    manifest, inputs = independently_load(campaign)
    if report['manifest_hash'] != manifest['manifest_hash']:
        raise ValueError('Analysis/manifest identity mismatch')
    units, documents = {}, {}
    for base in (corpus, corpus/'squad_e4'):
        docs_path = base/'documents.jsonl'
        if docs_path.exists():
            split_manifest = json.loads((base/'SPLIT_MANIFEST.json').read_text(encoding='utf-8'))
            if sha(docs_path) != split_manifest['files']['documents.jsonl']['sha256']:
                raise ValueError('Canonical source documents changed')
            for doc in rows(docs_path):
                documents[doc['doc_id']] = doc
                units.update((u['unit_id'], u) for u in doc['evidence_units'])
    grouped, files, seen, tickets = defaultdict(list), [], set(), set()
    stores = []
    for directory, source_manifest, source_inputs, owned in independently_owned_sources(campaign, manifest, inputs):
        for path in sorted((directory/'records').glob('*.jsonl')):
            stores.append((path, source_manifest, source_inputs, owned))
    tasks = {row['task_id']: row for row in manifest['tasks']}
    for path, source_manifest, source_inputs, owned in stores:
        if path.name.startswith('warmup_'): continue
        files.append({'path': str(path), 'sha256': sha(path)})
        config = json.loads(path.with_suffix('.config.json').read_text(encoding='utf-8'))
        if config['manifest_hash'] != source_manifest['manifest_hash']:
            raise ValueError('Independent record-store manifest mismatch')
        for row in rows(path):
            expected = row.pop('record_hash')
            if signature(row) != expected: raise ValueError('Raw record hash mismatch')
            task_id = row['task_id']; ticket = row.get('call_ticket', {}).get('call_number')
            if task_id not in owned or task_id in seen or row['record_id'] != task_id:
                raise ValueError('Independent duplicate/wrong-owner record')
            if any(row.get(key) != value for key, value in tasks[task_id].items()):
                raise ValueError('Independent record task differs')
            if (row.get('real_model') is not True or row['model_identity'] != manifest['models'][row['model_id']]
                    or row['config_hash'] != signature(config) or ticket is None or ticket in tickets):
                raise ValueError('Independent model/config/request identity mismatch')
            seen.add(task_id); tickets.add(ticket)
            source = inputs[row['input_id']]
            if source['input_hash'] != manifest['input_hashes'][row['input_id']]:
                raise ValueError('Frozen input identity mismatch')
            source_body = {k:v for k,v in source.items() if k != 'input_hash'}
            if signature(source_body) != source['input_hash']: raise ValueError('Frozen input hash mismatch')
            if not row.get('error'):
                parsed = json.loads(row['raw_output'])
                if parsed['answer'] != row['answer'] or parsed['abstain'] != row['abstain']:
                    raise ValueError('Saved prediction differs from independently parsed raw output')
            query = source['query']; f1, em = overlap(row, query)
            bundle = row['context_bundle']
            if bundle != source['context_bundle']:
                raise ValueError('Actual prompt context differs from the frozen input')
            if bundle['unit_spans'] != [s for p in bundle['pieces'] for s in p['unit_spans']]:
                raise ValueError('Evidence provenance aggregation mismatch')
            for piece in bundle['pieces']:
                a, b = piece['canonical_start'], piece['canonical_end']
                if documents[piece['doc_id']]['text'][a:b] != piece['text']:
                    raise ValueError('Prompt text differs from canonical source slice')
            evidence = independently_score(query, row['context_bundle']['unit_spans'], units, failed=False)
            grouped[(row['experiment_id'], row['model_id'], row['method'])].append({
                'query_id': row['query_id'], 'f1': f1, 'em': em, 'full': evidence['full_reference_coverage'],
                'failed': bool(row.get('error')), 'latency_ms': row['latency_ms'], 'record_hash': expected})
    checks = []
    for group in report['groups']:
        if group['status'] != 'COMPLETE': continue
        key = (group['experiment_id'], group['model_id'], group['method'])
        observed = grouped[key]
        if len(observed) != group['n_expected'] or len({r['query_id'] for r in observed}) != len(observed):
            raise ValueError('Incomplete/duplicated raw audit population')
        def mean(field):
            values = [r[field] for r in observed if r[field] is not None]
            return sum(values)/len(values) if values else None
        values = {'token_f1': (mean('f1'), group['estimates']['token_f1']['mean']),
            'exact_match': (mean('em'), group['observed_summary']['exact_match']['overall']),
            'context_full_reference_coverage': (mean('full'), group['estimates']['context_full_reference_coverage']['mean']),
            'failed_n': (sum(r['failed'] for r in observed), group['observed_summary']['n_failed']),
            'generation_p50_ms': (quantile([r['latency_ms'] for r in observed], .5), group['observed_summary']['latency_ms']['p50'])}
        for name, (actual, claimed) in values.items():
            if actual is None and claimed is None: continue
            checks.append({'group': key, 'metric': name, 'n_rows': len(observed), 'recomputed': actual,
                'reported': claimed, 'match': actual is not None and claimed is not None and abs(actual-claimed) < 1e-9,
                'raw_record_hashes': [r['record_hash'] for r in observed]})
    if len(checks) < 10: raise ValueError('Fewer than10 reported numbers available; audit remains pending')
    selected = random.Random(42).sample(checks, 10)
    result = {'status': 'PASS' if all(c['match'] for c in checks) else 'FAILED_VALIDATION',
        'seed': 42, 'independently_recomputed_numbers': len(checks), 'random_report_numbers': selected,
        'all_checks': checks, 'analysis_source': {'path': str(analysis_path), 'sha256': sha(analysis_path)},
        'record_sources': files, 'scope': 'Raw answer overlap, source-coordinate full evidence, failure counts and median latency; not a second bootstrap validation'}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(result['status'], len(checks), 'numbers independently recomputed; random10 retained')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', required=True, type=Path)
    parser.add_argument('--analysis', required=True, type=Path)
    parser.add_argument('--corpus', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    return 0 if audit(args.campaign, args.analysis, args.corpus, args.output)['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
