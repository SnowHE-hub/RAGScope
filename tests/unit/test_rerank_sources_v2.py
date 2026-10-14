"""Source handoff safety fixtures; these are not real model experiments."""
import json

import pytest

from ragscope.experiments.rerank_sources import accepted_gpu_sources
from ragscope.indexing.identity import artifact, atomic_json, signature


def fixture(tmp_path):
    run = tmp_path / 'runs' / 'fixture_retrieval'
    records = run / 'records'
    records.mkdir(parents=True)
    protected = run / 'cpu_original.json'
    protected.write_text('original CPU evidence', encoding='utf-8')
    code = run / 'fixture_source.py'
    code.write_text('# model-free fixture', encoding='utf-8')
    atomic_json(run / 'CANDIDATE_SELECTION.json', {'selected': {'structured': 'R512'}})
    atomic_json(run / 'CONTEXT_SELECTION.json', {'selected': {'R512': 'neighbor'}})
    groups, accepted = [], []
    for prefix, n in [('E8', 100), ('ROUTE_CONTEXT_TRAIN', 200), ('ROUTE_CONTEXT_DEV', 60)]:
        run_id = f'{prefix}_R512_hybrid_rerank_neighbor_gpu'
        path = records / (run_id + '.jsonl')
        path.write_text('fixture bytes only', encoding='utf-8')
        groups.append({'prefix': prefix, 'run_id': run_id, 'destination': str(path.resolve()),
            'n_expected': n, 'query_ids': [f'{prefix}:{i}' for i in range(n)],
            'source_artifacts': [artifact(protected)], 'selection_provenance': []})
        accepted.append({'run_id': run_id, 'n_total': n, 'artifacts': [artifact(path)]})
    migration = {'expected_gpu_rows': 360, 'device': 'cuda', 'historical_extra_partial_calls': 29,
        'protected_cpu_artifacts': [artifact(protected)], 'source_code_artifacts': [artifact(code)],
        'checkpoint_manifest': artifact(protected), 'groups': groups, 'comparison_limit': 'fixture only'}
    migration['migration_hash'] = signature(migration)
    atomic_json(run / 'R3_GPU_MIGRATION.json', migration)
    validation = {'status': 'PASS', 'device': 'cuda', 'scope': 'all_gpu_three_groups',
        'gpu_migration_complete': True, 'worker_exited': True, 'n_total': 360, 'n_failed': 0,
        'migration': artifact(run / 'R3_GPU_MIGRATION.json'),
        'validation_source': artifact(code), 'groups': accepted}
    atomic_json(run / 'R3_GPU_COMPLETION_VALIDATION.json', validation)
    return run, protected, validation


def test_complete_bound_sources_resolve_without_backend(tmp_path):
    run, _, _ = fixture(tmp_path)
    result = accepted_gpu_sources(tmp_path, 'fixture')
    assert set(result['paths']) == {'E8', 'ROUTE_CONTEXT_TRAIN', 'ROUTE_CONTEXT_DEV'}
    assert result['paths']['E8'].parent == run / 'records'


@pytest.mark.parametrize('field,value', [('n_total', 359), ('worker_exited', False),
    ('gpu_migration_complete', False), ('scope', 'E8')])
def test_partial_or_live_worker_never_becomes_main_source(tmp_path, field, value):
    run, _, validation = fixture(tmp_path)
    validation[field] = value
    atomic_json(run / 'R3_GPU_COMPLETION_VALIDATION.json', validation)
    with pytest.raises(ValueError):
        accepted_gpu_sources(tmp_path, 'fixture')


def test_changed_original_cpu_evidence_rejects_gpu_handoff(tmp_path):
    _, protected, _ = fixture(tmp_path)
    protected.write_text('changed', encoding='utf-8')
    with pytest.raises(ValueError, match='Protected CPU'):
        accepted_gpu_sources(tmp_path, 'fixture')


def test_changed_gpu_record_rejects_accepted_receipt(tmp_path):
    run, _, _ = fixture(tmp_path)
    next((run / 'records').glob('E8_*.jsonl')).write_text('changed', encoding='utf-8')
    with pytest.raises(ValueError, match='records or population'):
        accepted_gpu_sources(tmp_path, 'fixture')


def test_missing_route_group_rejects_complete_claim(tmp_path):
    run, _, validation = fixture(tmp_path)
    validation['groups'].pop()
    atomic_json(run / 'R3_GPU_COMPLETION_VALIDATION.json', validation)
    with pytest.raises(ValueError, match='three exact'):
        accepted_gpu_sources(tmp_path, 'fixture')
