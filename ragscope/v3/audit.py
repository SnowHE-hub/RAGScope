"""Independent, streaming reconstruction of the preserved v2 answer journals.

This module deliberately does not call the v2 answer scorer. Frozen source and
record hashes are verified before independent overlap arithmetic is performed.
Large inputs are stored once in SQLite, rather than repeated for every model.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import gc
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import statistics
import string
import tarfile
import time

from ragscope.indexing.identity import artifact, atomic_json, canonical_json, signature


def normalized_words(text):
    text = str(text).lower()
    text = ''.join(c for c in text if c not in string.punctuation)
    return re.sub(r'\b(a|an|the)\b', ' ', text).split()


def reference_texts(query):
    """Reconstruct the declared v2 reference convention independently."""
    texts = []
    for ref in query.get('answer_refs', []):
        if isinstance(ref, str):
            texts.append(ref)
        elif isinstance(ref, dict) and not ref.get('unanswerable'):
            if ref.get('yes_no') is not None:
                texts.append('yes' if ref['yes_no'] else 'no')
            elif ref.get('free_form_answer'):
                texts.append(ref['free_form_answer'])
            elif ref.get('extractive_spans'):
                texts.append(', '.join(ref['extractive_spans']))
            else:
                text = ref.get('text', ref.get('answer'))
                if isinstance(text, str):
                    texts.append(text)
    return list(dict.fromkeys(texts))


def overlap(prediction, references):
    pred = normalized_words(prediction)
    results = []
    for i, ref in enumerate(references):
        gold = normalized_words(ref)
        same = sum((Counter(pred) & Counter(gold)).values())
        precision = same / len(pred) if pred else 0.0
        recall = same / len(gold) if gold else 0.0
        # The official QASPER implementation returns zero for empty overlap.
        official = 2 * same / (len(pred) + len(gold)) if same else 0.0
        legacy = official if pred or gold else 1.0
        results.append({'reference_index': i, 'precision': precision, 'recall': recall,
                        'legacy_token_f1': legacy, 'qasper_compatible_answer_f1': official,
                        'exact_match': float(pred == gold), 'reference_words': len(gold)})
    best = max(results, key=lambda r: r['legacy_token_f1'], default=None)
    return {'predicted_words': len(pred), 'reference_words': [len(normalized_words(r)) for r in references],
            'best_reference_index': best['reference_index'] if best else None,
            'precision': best['precision'] if best else 0.0,
            'recall': best['recall'] if best else 0.0,
            'legacy_token_f1': best['legacy_token_f1'] if best else 0.0,
            'qasper_compatible_answer_f1': max((r['qasper_compatible_answer_f1'] for r in results), default=0.0),
            'exact_match': max((r['exact_match'] for r in results), default=0.0)}


def journal_rows(path, *, require_hash=True):
    """Never repair, truncate, or write an old source journal."""
    with Path(path).open('rb') as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.endswith(b'\n'):
                raise ValueError(f'Incomplete immutable source: {path}:{line_number}')
            row = json.loads(line)
            expected = row.get('record_hash')
            if (require_hash and not expected) or (expected and
                    signature({k: v for k, v in row.items() if k != 'record_hash'}) != expected):
                raise ValueError(f'Corrupt source record: {path}:{line_number}')
            yield line_number, row


def load_manifest(directory):
    manifest = json.loads((Path(directory) / 'EXPERIMENT_MANIFEST.json').read_text('utf-8'))
    if signature({k: v for k, v in manifest.items() if k != 'manifest_hash'}) != manifest['manifest_hash']:
        raise ValueError('Manifest content hash mismatch')
    return manifest


def answer_type(query):
    if query['answerability'] == 'unanswerable':
        return 'unanswerable'
    annotations = query.get('metadata', {}).get('annotations', [])
    types = sorted({a.get('answer_type', 'unknown') for a in annotations})
    return '+'.join(types) if types else 'extractive_or_unspecified'


def query_document_id(query):
    """Some frozen Query dataclasses retain document identity only in scope."""
    if query.get('doc_id'):
        return query['doc_id']
    ids = query.get('scope', {}).get('doc_ids', [])
    if len(ids) == 1:
        return ids[0]
    return query.get('metadata', {}).get('gold_doc_id_evaluation_only')


def projection(row, query, source):
    refs = reference_texts(query)
    pred = row.get('answer') or ''
    result = overlap(pred, refs)
    answerable = query['answerability'] != 'unanswerable'
    failed = bool(row.get('error'))
    abstain = row.get('abstain') is True
    for key in ('legacy_token_f1', 'qasper_compatible_answer_f1', 'exact_match'):
        if not answerable:
            result[key] = None
        elif failed or abstain:
            result[key] = 0.0
    observed = row['scores']['token_f1']
    if observed != result['legacy_token_f1'] and not (observed is not None and
            result['legacy_token_f1'] is not None and abs(observed - result['legacy_token_f1']) <= 1e-12):
        raise ValueError('Independent F1 disagrees with raw journal: ' + row['record_id'])
    if row['scores'].get('exact_match') != result['exact_match']:
        raise ValueError('Independent exact match disagrees: ' + row['record_id'])
    backend = row.get('backend_metadata', {})
    tokens = backend.get('eval_count', row.get('completion_tokens'))
    decode_ns = backend.get('eval_duration')
    output = {
        'record_id': row['record_id'], 'query_id': query['query_id'], 'doc_id': query_document_id(query) or row.get('doc_id'),
        'exposure_status': 'EXPOSED_DIAGNOSTIC', 'source': source,
        'input_id': row.get('input_id'), 'input_hash': row.get('input_hash'),
        'experiment_id': row.get('experiment_id', source.get('condition')),
        'model_id': row.get('model_id', source.get('condition')), 'method': row.get('method', 'paired_lora'),
        'question': query['question'], 'answer_refs': refs, 'answer_type': answer_type(query),
        'document_answerable': answerable, 'visible_context_sufficient': 'UNKNOWN',
        'context_sufficiency_source': 'NOT_ASSESSED; evidence coverage is not sufficiency',
        'model_abstained': abstain, 'answer': pred, 'raw_output': row.get('raw_output'),
        'parsed_output': row.get('parsed_output'), 'error': row.get('error'), 'failed': failed,
        'citations': row.get('citations', []), 'scope': query.get('scope'),
        'context_hash': row['context_bundle']['context_hash'],
        'context_budget': row['context_bundle']['budget'],
        'context_evidence_legacy': row['scores'].get('context_evidence'),
        'citation_evidence_legacy': row['scores'].get('citation_evidence'),
        'actual_input_tokens': backend.get('actual_input_tokens', backend.get('prompt_eval_count', row.get('prompt_tokens'))),
        'actual_output_tokens': tokens, 'output_limit_reached_proxy': tokens >= 512 if isinstance(tokens, int) else None,
        'finish_reason': backend.get('done_reason', backend.get('finish_reason')),
        'finish_reason_status': 'OBSERVED' if backend.get('done_reason', backend.get('finish_reason')) else 'NOT_RECORDED',
        'first_token_ms': row.get('timings', {}).get('first_token_ms'),
        'decode_tokens_per_second': tokens / (decode_ns / 1e9) if isinstance(tokens, int) and decode_ns else None,
        'timings': row.get('timings', {}), 'backend_metadata': {k: v for k, v in backend.items() if k != 'resources'},
        'latency_ms': row.get('timings', {}).get('generation_ms', row.get('latency_ms')),
        'config_hash': row['config_hash'], 'source_record_hash': row['record_hash'], **result}
    return output


def add_group(groups, record):
    key = (record['experiment_id'], record['model_id'], record['method'])
    group = groups.setdefault(key, {'n': 0, 'failures': 0, 'answerable_n': 0,
        'unanswerable_n': 0, 'answerable_abstentions': 0, 'unanswerable_abstentions': 0,
        'legacy_f1_sum': 0.0, 'compatible_f1_sum': 0.0, 'em_sum': 0.0,
        'latencies_ms': [], 'answer_type_counts': Counter()})
    group['n'] += 1
    group['failures'] += int(record['failed'])
    group['answer_type_counts'][record['answer_type']] += 1
    if record['document_answerable']:
        group['answerable_n'] += 1
        group['answerable_abstentions'] += int(record['model_abstained'])
        group['legacy_f1_sum'] += record['legacy_token_f1']
        group['compatible_f1_sum'] += record['qasper_compatible_answer_f1']
        group['em_sum'] += record['exact_match']
    else:
        group['unanswerable_n'] += 1
        group['unanswerable_abstentions'] += int(record['model_abstained'])
    if record['latency_ms'] is not None:
        group['latencies_ms'].append(record['latency_ms'])


def group_summaries(groups, expected):
    output = []
    for key in sorted(set(groups) | set(expected)):
        group = groups.get(key, {})
        n = group.get('answerable_n', 0)
        output.append({'experiment_id': key[0], 'model_id': key[1], 'method': key[2],
            'expected_n': expected[key], 'status': 'PASS' if group.get('n', 0) == expected[key] else 'INCOMPLETE',
            **{k: v for k, v in group.items() if not k.endswith('_sum') and k != 'latencies_ms'},
            'legacy_token_f1': group.get('legacy_f1_sum', 0) / n if n else None,
            'qasper_compatible_answer_f1': group.get('compatible_f1_sum', 0) / n if n else None,
            'exact_match': group.get('em_sum', 0) / n if n else None,
            'generation_p50_ms': statistics.median(group['latencies_ms']) if group.get('latencies_ms') else None})
    return output


def add_inventory(inventory, path, role, expected=None):
    path = Path(path)
    row = {'path': str(path.resolve()), 'role': role, 'status': 'MISSING'}
    if path.is_file():
        row.update(artifact(path), status='FOUND')
        if expected and any(row.get(k) != expected.get(k) for k in ('bytes', 'sha256')):
            row['status'] = 'HASH_MISMATCH'
    inventory[str(path.resolve())] = row
    return row


def ingest_inputs(db, directory, manifest, main_hashes, inventory):
    from ragscope.generation.context import validate_bundle
    path = Path(directory) / 'FROZEN_INPUTS.json'
    add_inventory(inventory, path, 'immutable_frozen_inputs')
    inputs = json.loads(path.read_text('utf-8'))
    if set(inputs) != set(manifest['input_hashes']):
        raise ValueError('Frozen input membership mismatch')
    for input_id, value in inputs.items():
        body = {k: v for k, v in value.items() if k != 'input_hash'}
        digest = signature(body)
        if digest != value['input_hash'] or digest != manifest['input_hashes'][input_id] or digest != main_hashes[input_id]:
            raise ValueError('Frozen input content differs: ' + input_id)
        validate_bundle(value['context_bundle'], scope=value['query'].get('scope'))
        previous = db.execute('SELECT input_hash FROM inputs WHERE input_id=?', (input_id,)).fetchone()
        if previous and previous[0] != digest:
            raise ValueError('Cross-source input mismatch')
        if not previous:
            db.execute('INSERT INTO inputs VALUES(?,?,?)', (input_id, digest, canonical_json(value)))
            query = value['query']
            prior = db.execute('SELECT payload FROM queries WHERE query_id=?', (query['query_id'],)).fetchone()
            if prior:
                old = json.loads(prior[0])
                for key in ('question', 'doc_id', 'answer_refs', 'answerability'):
                    if old.get(key) != query.get(key):
                        raise ValueError('Same question has inconsistent gold references')
            else:
                db.execute('INSERT INTO queries VALUES(?,?)', (query['query_id'], canonical_json(query)))
    db.commit()
    count = len(inputs)
    del inputs
    gc.collect()
    return count


def audit_generation(run, runtime, inventory):
    old = runtime / 'runs/20261014_v2_generation'
    manifest = load_manifest(old)
    add_inventory(inventory, old / 'EXPERIMENT_MANIFEST.json', 'main_generation_manifest')
    tasks = {t['task_id']: t for t in manifest['tasks']}
    expected = Counter((t['experiment_id'], t['model_id'], t['method']) for t in tasks.values())
    sources, owned = [], set()
    for ref in manifest['provenance'].get('external_campaigns', []):
        receipt = ref['manifest']
        path = Path(receipt['path'])
        if not path.resolve().is_relative_to(runtime.resolve()):
            raise ValueError('External source outside approved runtime')
        entry = add_inventory(inventory, path, 'external_generation_manifest', receipt)
        if entry['status'] != 'FOUND':
            raise ValueError('External source missing or changed')
        source = load_manifest(path.parent)
        if source.get('provenance', {}).get('external_campaigns'):
            raise ValueError('Nested source ownership unsupported')
        for key in ('generation_config', 'metric_version', 'reference_set_rule', 'parse_repair_attempts', 'seed'):
            if source.get(key) != manifest.get(key):
                raise ValueError('External protocol differs: ' + key)
        for key in ('prompt_version', 'prompt_hash', 'output_constraint_version', 'backend_runtime_config'):
            if source['provenance'].get(key) != manifest['provenance'].get(key):
                raise ValueError('External prompt/runtime differs: ' + key)
        ids = {t['task_id'] for t in source['tasks']}
        if owned & ids or any(t != tasks.get(t['task_id']) for t in source['tasks']):
            raise ValueError('Duplicate or changed external ownership')
        owned.update(ids)
        sources.append((path.parent, source, ids))
    sources.insert(0, (old, manifest, set(tasks) - owned))
    db_path = run / 'S0_EVIDENCE.sqlite'
    db = sqlite3.connect(db_path)
    db.execute('PRAGMA journal_mode=DELETE')
    db.executescript('CREATE TABLE IF NOT EXISTS inputs(input_id TEXT PRIMARY KEY,input_hash TEXT,payload TEXT);'
                     'CREATE TABLE IF NOT EXISTS queries(query_id TEXT PRIMARY KEY,payload TEXT);'
                     'CREATE TABLE IF NOT EXISTS records(record_id TEXT PRIMARY KEY,query_id TEXT,payload TEXT);')
    seen, tickets, groups, config_ids = set(), set(), {}, {}
    for directory, source_manifest, owned_ids in sources:
        ingest_inputs(db, directory, source_manifest, manifest['input_hashes'], inventory)
        files = sorted({f"{tasks[tid]['experiment_id']}_{tasks[tid]['model_id']}.jsonl" for tid in owned_ids})
        for filename in files:
            path = directory / 'records' / filename
            entry = add_inventory(inventory, path, 'raw_generation_journal')
            if entry['status'] == 'MISSING':
                continue
            cfg_path = path.with_suffix('.config.json')
            add_inventory(inventory, cfg_path, 'immutable_record_config')
            config = json.loads(cfg_path.read_text('utf-8'))
            if config['manifest_hash'] != source_manifest['manifest_hash']:
                raise ValueError('Journal config source differs')
            config_hash = signature(config)
            config_ids[config_hash] = config
            for number, row in journal_rows(path):
                tid = row['task_id']
                if tid not in owned_ids or tid in seen or row['record_id'] != tid:
                    raise ValueError('Wrong-owner/duplicate/unknown journal task')
                if row.get('config_hash') != config_hash or any(row.get(k) != v for k, v in tasks[tid].items()):
                    raise ValueError('Task/config differs from immutable manifest')
                if row.get('real_model') is not True or row['model_identity'] != manifest['models'][row['model_id']]:
                    raise ValueError('Mock or different model')
                ticket = row.get('call_ticket', {}).get('call_number')
                if not isinstance(ticket, int) or ticket in tickets:
                    raise ValueError('Missing/duplicate request reservation')
                input_blob = db.execute('SELECT payload FROM inputs WHERE input_id=?', (row['input_id'],)).fetchone()
                source_input = json.loads(input_blob[0])
                if row['context_bundle'] != source_input['context_bundle'] or row['question'] != source_input['query']['question']:
                    raise ValueError('Actual sent question/context differs')
                locator = {'path': str(path), 'line': number, 'manifest_hash': source_manifest['manifest_hash'],
                           'config_hash': config_hash, 'record_hash': row['record_hash']}
                projected = projection(row, source_input['query'], locator)
                encoded = canonical_json(projected)
                previous = db.execute('SELECT payload FROM records WHERE record_id=?', (tid,)).fetchone()
                if previous and previous[0] != encoded:
                    raise ValueError('Existing v3 projection differs; create a new identity')
                if not previous:
                    db.execute('INSERT INTO records VALUES(?,?,?)', (tid, row['query_id'], encoded))
                seen.add(tid)
                tickets.add(ticket)
                add_group(groups, projected)
            db.commit()
            print(json.dumps({'event': 'JOURNAL_AUDITED', 'source': filename, 'cumulative_records': len(seen)}), flush=True)
    summaries = group_summaries(groups, expected)
    out = {'status': 'PASS' if len(seen) == len(tasks) else 'INCOMPLETE',
        'observed_n': len(seen), 'expected_n': len(tasks), 'failures': sum(g['failures'] for g in summaries),
        'group_n': len(summaries), 'groups': summaries, 'missing_task_ids': sorted(set(tasks) - seen),
        'source_manifest_hash': manifest['manifest_hash'], 'model_identities': manifest['models'],
        'config_identities': config_ids, 'new_llm_requests': 0,
        'method_mapping': {'R0': 'BM25', 'R1': 'Dense', 'R2': 'Hybrid', 'R3': 'Hybrid plus reranker'},
        'method_mapping_status': 'CHECK_SOURCE_SELECTION_BEFORE_FINAL_VERDICT',
        'historical_report_expected': {'records': 5520, 'groups': 39, 'failures': 32},
        'report_count_agrees': len(seen) == 5520 and len(summaries) == 39 and sum(g['failures'] for g in summaries) == 32,
        'answer_field_scored': 'parsed answer only; never raw JSON or citation fields',
        'independent_arithmetic_matches_all_raw_f1_and_em': True,
        'unanswerable_f1': None, 'failure_policy': 'Zero within answerable denominator; retain all records',
        'context_sufficiency': 'UNKNOWN pending question/context assessment; not copied from coverage',
        'human_review_status': 'NOT_REVIEWED'}
    atomic_json(run / 'BASELINE_RECOUNT.json', out)
    db.close()
    return out


def audit_lora(run, runtime, inventory):
    path = runtime / 'runs/20261014_v2_lora/train_2b_1024_seed42/paired_test100'
    db = sqlite3.connect(run / 'S0_EVIDENCE.sqlite')
    pairs, conditions = {}, {}
    for condition in ('baseline', 'adapted'):
        journal = path / (condition + '.jsonl')
        config_path = journal.with_suffix('.config.json')
        add_inventory(inventory, journal, 'lora_paired_raw_output')
        add_inventory(inventory, config_path, 'lora_paired_config')
        config = json.loads(config_path.read_text('utf-8'))
        indexed, groups = {}, {}
        for number, row in journal_rows(journal):
            qid = row['query_id']
            if qid in indexed or row['record_id'] != qid or row['config_hash'] != signature(config) or row.get('real_model') is not True:
                raise ValueError('LoRA paired identity/config mismatch')
            query = db.execute('SELECT payload FROM queries WHERE query_id=?', (qid,)).fetchone()
            if not query:
                raise ValueError('LoRA question lacks original gold source')
            query = json.loads(query[0])
            if query['question'] != row['question'] or query_document_id(query) != row['doc_id']:
                raise ValueError('LoRA question differs from frozen source')
            p = projection(row, query, {'path': str(journal), 'line': number,
                'condition': condition, 'record_hash': row['record_hash'], 'config_hash': row['config_hash']})
            indexed[qid] = (row, p)
            add_group(groups, p)
        conditions[condition] = indexed
    if set(conditions['baseline']) != set(conditions['adapted']):
        raise ValueError('LoRA paired populations differ')
    counts, contributions = Counter(), defaultdict(float)
    answerable_n = 0
    for qid in sorted(conditions['baseline']):
        before, bp = conditions['baseline'][qid]
        after, ap = conditions['adapted'][qid]
        if before['context_bundle'] != after['context_bundle'] or before['prompt'] != after['prompt']:
            raise ValueError('LoRA paired prompts/contexts differ')
        delta = None
        if bp['document_answerable']:
            answerable_n += 1
            delta = ap['legacy_token_f1'] - bp['legacy_token_f1']
            if bp['failed'] or ap['failed']:
                group = 'FORMAT_OR_REQUEST_FAILURE'
            elif not bp['model_abstained'] and ap['model_abstained']:
                group = 'NEW_ABSTENTION'
            elif delta > 1e-12:
                group = 'IMPROVED'
            elif delta < -1e-12:
                group = 'WORSE_STILL_ANSWERING'
            else:
                group = 'UNCHANGED'
            contributions[group] += delta
        else:
            group = 'UNANSWERABLE_SEPARATE'
        counts[group] += 1
        pairs[qid] = {'query_id': qid, 'doc_id': bp['doc_id'], 'group': group,
            'delta_legacy_token_f1': delta, 'baseline': bp, 'adapted': ap,
            'refusal_reason_verdict': 'UNKNOWN pending evidence-only assessment' if group == 'NEW_ABSTENTION' else None}
    outfile = run / 'LORA_100_PAIRS.jsonl'
    with outfile.open('w', encoding='utf-8', newline='\n') as handle:
        for pair in pairs.values():
            handle.write(canonical_json(pair) + '\n')
        handle.flush()
        os.fsync(handle.fileno())
    summaries = {}
    for condition, indexed in conditions.items():
        rows = [p for _, p in indexed.values()]
        can = [r for r in rows if r['document_answerable']]
        summaries[condition] = {'n': len(rows), 'answerable_n': len(can),
            'f1': sum(r['legacy_token_f1'] for r in can) / len(can),
            'em': sum(r['exact_match'] for r in can) / len(can),
            'answerable_abstentions': sum(r['model_abstained'] for r in can),
            'failures': sum(r['failed'] for r in rows),
            'generation_p50_ms': statistics.median(r['latency_ms'] for r in rows),
            'output_tokens_p50': statistics.median(r['actual_output_tokens'] for r in rows if r['actual_output_tokens'] is not None),
            'at_output_limit_n': sum(r['output_limit_reached_proxy'] is True for r in rows),
            'finish_reason_not_recorded_n': sum(r['finish_reason_status'] == 'NOT_RECORDED' for r in rows)}
    delta = summaries['adapted']['f1'] - summaries['baseline']['f1']
    contribution = {k: v / answerable_n for k, v in contributions.items()}
    if not math.isclose(sum(contribution.values()), delta, abs_tol=1e-12):
        raise ValueError('LoRA mutually exclusive decomposition does not reconstruct delta')
    result = {'status': 'PASS', 'conditions': summaries, 'group_counts': counts,
        'answerable_n': answerable_n, 'delta_f1': delta, 'group_contributions': contribution,
        'contribution_sum_matches_delta': True, 'new_llm_requests': 0,
        'human_review_status': 'NOT_REVIEWED', 'causal_verdict': 'Not inferred from correlation'}
    atomic_json(run / 'LORA_OLD_RECOUNT.json', result)
    db.close()
    return result


def preserve_official_evaluator(run, runtime, inventory):
    archive = runtime / 'corpora/20261014_v2/raw/qasper-test-and-evaluator-v0.3.tgz'
    add_inventory(inventory, archive, 'official_qasper_release_archive')
    with tarfile.open(archive) as bundle:
        contents = bundle.extractfile('qasper_evaluator.py').read()
    path = run / 'source_package/qasper_evaluator_v0.3.py'
    if path.exists() and path.read_bytes() != contents:
        raise ValueError('Pinned official evaluator changed')
    path.write_bytes(contents)
    add_inventory(inventory, path, 'pinned_official_evaluator')
    return path
