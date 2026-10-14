"""Read-only R3 acceptance after the owned worker exits; never imports a model."""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random

from ragscope.experiments.ledger import read_records
from ragscope.experiments.rerank_campaign import frozen_jobs, verified_source, find_index_material, restore_candidates
from ragscope.indexing.identity import artifact, atomic_json, signature


def validate_windows(row, source, chunks):
    """Check every scored candidate, coverage and max aggregation against the actual source."""
    candidates, eligible = restore_candidates(source, chunks)
    if (row['raw_retrieval'] != source['raw_retrieval'] or row['source_record_hash'] != source['record_hash']
            or row['source_record_id'] != source['record_id'] or row['candidate_count'] != len(candidates)
            or row['scoped_chunk_count'] != eligible):
        raise ValueError('R3 source/candidate identity differs')
    if row.get('error'):
        return {'failed': True, 'windows': len(row['rerank_windows']), 'max_pair_tokens': None}
    grouped = defaultdict(list)
    for window in row['rerank_windows']:
        if not 1 <= window['pair_tokens'] <= 512 or not math.isfinite(window['score']):
            raise ValueError('Invalid query-inclusive window budget/score')
        grouped[window['chunk_id']].append(window)
    if set(grouped) != {item['chunk_id'] for item in candidates}:
        raise ValueError('Window population does not cover every original candidate')
    scores = {}
    lookup = {item['chunk_id']: item for item in candidates}
    for cid, windows in grouped.items():
        length = len(lookup[cid]['text'])
        covered = 0
        for window in sorted(windows, key=lambda value: (value['start'], value['end'])):
            if not 0 <= window['start'] < window['end'] <= length or window['start'] > covered:
                raise ValueError('Window coordinates leave a gap or exceed original text')
            covered = max(covered, window['end'])
        if covered != length:
            raise ValueError('Window coverage silently drops the end of original text')
        scores[cid] = max(window['score'] for window in windows)
    selected = row['selected_retrieval']
    expected = sorted(scores, key=lambda cid: (-scores[cid], cid))[:10]
    if [item['chunk_id'] for item in selected] != expected:
        raise ValueError('Selected order differs from all-candidate windowed_max ranking')
    for rank, item in enumerate(selected, 1):
        original = lookup[item['chunk_id']]
        if (item['rank'] != rank or item['rerank_score'] != scores[item['chunk_id']]
                or item['rerank_windows'] != grouped[item['chunk_id']]
                or any(item[key] != original[key] for key in
                       ('doc_id', 'score', 'canonical_start', 'canonical_end', 'unit_spans', 'hybrid_rank'))):
            raise ValueError('Selected candidate score/coordinates/windows differ')
    return {'failed': False, 'windows': len(row['rerank_windows']),
            'max_pair_tokens': max(window['pair_tokens'] for window in row['rerank_windows'])}


def validate_campaign(home, campaign, *, prefix=None, gpu=False):
    import psutil
    import tiktoken
    from ragscope.generation.context import validate_bundle, render_context
    from ragscope.experiments.retrieval_campaign import score_retrieval
    from ragscope.retrieval.verified_reranker import verified_checkpoint, REVISION
    home = Path(home)
    run = home / 'runs' / (campaign + '_retrieval')
    device = 'cuda' if gpu else 'cpu'
    worker = json.loads((run / ('R3_GPU_RUNTIME' if gpu else 'R3_RUNTIME') / 'worker.json').read_text('utf-8'))
    worker_alive = False
    try:
        process = psutil.Process(worker['pid'])
        if abs(process.create_time() - worker['creation_time']) < .001:
            worker_alive = True
            if prefix is None:
                raise RuntimeError('R3 worker still exists; final acceptance requires process exit')
    except psutil.NoSuchProcess:
        pass
    _, checkpoint = verified_checkpoint(home)
    with (home / 'corpora' / campaign / 'documents.jsonl').open(encoding='utf-8') as reader:
        units = {unit['unit_id']: unit for line in reader for unit in json.loads(line)['evidence_units']}
    migration = None
    if gpu:
        from ragscope.experiments.rerank_gpu import gpu_jobs, verify_migration, migration_path
        migration = verify_migration(home, campaign)
        jobs = gpu_jobs(home, campaign)
    else:
        jobs = frozen_jobs(home, campaign)
    if prefix is not None:
        jobs = [job for job in jobs if job['prefix'] == prefix]
        if len(jobs) != 1:
            raise ValueError('A partial acceptance must name exactly one existing frozen group')
    groups = []
    for job in jobs:
        config, sources = verified_source(job)
        source_by_id = {row['query_id']: row for row in sources}
        queries = {row['query_id']: row for row in job['queries']}
        path = job['destination']
        rows = read_records(path)
        result_config = json.loads(path.with_suffix('.config.json').read_text('utf-8'))
        summary_path = run / 'summaries' / (path.stem + '.json')
        summary = json.loads(summary_path.read_text('utf-8'))
        if len(rows) != len(queries) or {row['query_id'] for row in rows} != set(queries):
            raise ValueError('Missing/duplicate/extra R3 population')
        if result_config['source_config'] != config or result_config['query_ids'] != list(queries):
            raise ValueError('R3 configuration differs from the frozen source population')
        identity = result_config['reranker_identity']
        expected_identity = {'checkpoint_manifest': checkpoint, 'revision': REVISION, 'device': device,
                             'cpu_threads': 4, 'dtype': 'torch.float32', 'max_length': 512,
                             'window_overlap': 32, 'batch_size': 2, 'aggregation': 'windowed_max'}
        if gpu:
            expected_identity.update(tf32=False, autocast=False, activation='sigmoid',
                                     attention_implementation='eager', float32_matmul_precision='highest',
                                     backend='transformers_sequence_classifier',
                                     torch_version=migration['backend_versions']['torch'],
                                     transformers_version=migration['backend_versions']['transformers'])
            expected_artifacts = [artifact(job['source']), artifact(job['source'].with_suffix('.config.json'))]
            if (result_config['source_artifacts'] != expected_artifacts
                    or result_config['selection_provenance'] != job['provenance'] + [artifact(migration_path(home, campaign))]):
                raise ValueError('GPU results are not bound to the immutable migration/source hashes')
        if any(identity.get(key) != value for key, value in expected_identity.items()):
            raise ValueError('R3 model identity differs from the pinned explicit device configuration')
        warm_path = run / 'warmups' / path.name
        warmups = read_records(warm_path)
        warm_config = json.loads(warm_path.with_suffix('.config.json').read_text('utf-8'))
        warm_ids = sorted(queries); random.Random(42).shuffle(warm_ids); warm_ids = warm_ids[:3]
        if (warm_config['query_ids'] != warm_ids or warm_config['formal_config_hash'] != signature(result_config)
                or len(warmups) != 3 or {row['record_id'] for row in warmups} != set(warm_ids)
                or any(row.get('real_model') is not True or row.get('error') or row['model_identity'] != identity
                       or row['config_hash'] != signature(warm_config) for row in warmups)):
            raise ValueError('The three fixed real excluded warmups are missing/failed/mismatched')
        chunks, _, _ = find_index_material(home, campaign, config['index_identity'])
        windows = 0; max_tokens = 0; failures = 0
        for row in rows:
            query = queries[row['query_id']]
            if (row.get('real_model') is not True or row['device'] != device
                    or row['model_identity'] != result_config['reranker_identity']
                    or row['config_hash'] != signature(result_config) or not row.get('record_hash')
                    or row['seed'] != 42 or row['record_id'] != row['query_id']
                    or any(row[key] != query[key] for key in ('question', 'scope', 'split', 'answerability'))):
                raise ValueError('R3 row is not bound to the frozen real explicit device experiment')
            if gpu:
                validate_gpu_memory(row.get('gpu_memory'))
            checked = validate_windows(row, source_by_id[row['query_id']], chunks)
            windows += checked['windows']; failures += checked['failed']
            max_tokens = max(max_tokens, checked['max_pair_tokens'] or 0)
            if not checked['failed']:
                bundle = validate_bundle(row['context_bundle'], query['scope'])
                budget = bundle['budget']
                counted = len(tiktoken.get_encoding(budget['reference_tokenizer']).encode(render_context(bundle), disallowed_special=()))
                if (budget['max_reference_tokens'] != 2048 or budget['expansion'] != job['expansion']
                        or budget['reference_tokens'] != counted or counted > 2048):
                    raise ValueError('Final packed context budget differs')
            gold = query.get('doc_id', query.get('metadata', {}).get('gold_doc_id_evaluation_only'))
            expected = score_retrieval({**query, 'doc_id': gold}, row['selected_retrieval'],
                                       row['context_bundle'], units, failed=checked['failed'])
            if expected != row['scores']:
                raise ValueError('Stored retrieval metrics do not reproduce from selected/context evidence')
        if summary['n_total'] != len(rows) or summary['n_failed'] != failures:
            raise ValueError('Summary silently changes the actual success/failure denominator')
        groups.append({'run_id': path.stem, 'n_total': len(rows), 'n_failed': failures,
                       'window_count': windows, 'max_actual_pair_tokens': max_tokens,
                       'candidate_min': min(row['candidate_count'] for row in rows),
                       'candidate_max': max(row['candidate_count'] for row in rows),
                       'n_excluded_warmups': len(warmups),
                       'artifacts': [artifact(path), artifact(path.with_suffix('.config.json')), artifact(summary_path),
                                     artifact(warm_path), artifact(warm_path.with_suffix('.config.json'))]})
    result = {'status': 'PASS', 'checked_at': datetime.now(timezone.utc).isoformat(), 'n_total': sum(g['n_total'] for g in groups),
              'n_failed': sum(g['n_failed'] for g in groups), 'groups': groups, 'worker_exited': not worker_alive,
              'worker_identity': worker, 'scope': ('all_gpu_three_groups' if gpu else 'all_five_groups') if prefix is None else prefix,
              'whole_campaign_complete': prefix is None and not gpu,
              'validation_source': artifact(__file__), 'interpretation': 'Acceptance of actual denominators/provenance/window arithmetic, not a claim that reranking improved quality'}
    if gpu:
        result.update(device='cuda', gpu_migration_complete=prefix is None,
                      migration=artifact(migration_path(home, campaign)), historical_partial_calls_preserved=29)
    expected_n = (360 if gpu else 660) if prefix is None else len(jobs[0]['queries'])
    if result['n_total'] != expected_n:
        raise ValueError('The required frozen-group denominator is incomplete')
    marker = 'R3_GPU' if gpu else 'R3'
    destination = f'{marker}_COMPLETION_VALIDATION.json' if prefix is None else f'{marker}_GROUP_{prefix}_VALIDATION.json'
    atomic_json(run / destination, result)
    return result


def validate_gpu_memory(memory):
    if not isinstance(memory, dict) or memory.get('device') != 'cuda:0':
        raise ValueError('GPU row lacks actual memory observations')
    keys = ('allocated_bytes', 'reserved_bytes', 'peak_allocated_bytes', 'peak_reserved_bytes', 'free_bytes', 'total_bytes')
    if any(isinstance(memory.get(k), bool) or not isinstance(memory.get(k), int) or memory[k] < 0 for k in keys):
        raise ValueError('GPU memory observations must be nonnegative integers')
    if (not 0 < memory['allocated_bytes'] <= memory['reserved_bytes'] <= memory['total_bytes']
            or not memory['allocated_bytes'] <= memory['peak_allocated_bytes'] <= memory['peak_reserved_bytes'] <= memory['total_bytes']
            or memory['free_bytes'] > memory['total_bytes']):
        raise ValueError('GPU memory observations are inconsistent')
