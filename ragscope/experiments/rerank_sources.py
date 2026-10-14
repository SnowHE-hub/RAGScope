"""Resolve explicitly accepted GPU R3 sources without importing a model backend."""
import json
from pathlib import Path

from ragscope.indexing.identity import artifact, signature, verify_artifacts


def accepted_gpu_sources(home, campaign):
    run = Path(home) / 'runs' / (campaign + '_retrieval')
    validation_path = run / 'R3_GPU_COMPLETION_VALIDATION.json'
    migration_path = run / 'R3_GPU_MIGRATION.json'
    validation = json.loads(validation_path.read_text(encoding='utf-8'))
    if (validation.get('status') != 'PASS' or validation.get('device') != 'cuda'
            or validation.get('scope') != 'all_gpu_three_groups'
            or validation.get('gpu_migration_complete') is not True
            or validation.get('worker_exited') is not True or validation.get('n_total') != 360
            or validation.get('migration') != artifact(migration_path)):
        raise ValueError('GPU R3 needs complete, exited, source-bound acceptance')
    migration = json.loads(migration_path.read_text(encoding='utf-8'))
    if (migration.get('migration_hash') != signature({k: v for k, v in migration.items() if k != 'migration_hash'})
            or migration.get('expected_gpu_rows') != 360 or migration.get('device') != 'cuda'
            or migration.get('historical_extra_partial_calls') != 29):
        raise ValueError('GPU migration identity or preserved history differs')
    protected = (migration['protected_cpu_artifacts'] + migration['source_code_artifacts']
                 + [migration['checkpoint_manifest'], validation['validation_source']])
    if not verify_artifacts(protected):
        raise ValueError('Protected CPU evidence, weights, or acceptance source changed')
    selection = json.loads((run / 'CANDIDATE_SELECTION.json').read_text(encoding='utf-8'))
    contexts = json.loads((run / 'CONTEXT_SELECTION.json').read_text(encoding='utf-8'))
    group = selection['selected']['structured']
    expansion = contexts['selected'][group]
    expected = {f'{prefix}_{group}_hybrid_rerank_{expansion}_gpu': n for prefix, n in (
        ('E8', 100), ('ROUTE_CONTEXT_TRAIN', 200), ('ROUTE_CONTEXT_DEV', 60))}
    accepted = {g['run_id']: g for g in validation['groups']}
    migrated = {g['run_id']: g for g in migration['groups']}
    if len(accepted) != 3 or len(migrated) != 3 or set(accepted) != set(expected) or set(migrated) != set(expected):
        raise ValueError('GPU acceptance does not contain the three exact frozen groups')
    paths = {}
    evidence = [artifact(validation_path), artifact(migration_path)]
    for run_id, n in expected.items():
        checked, frozen = accepted[run_id], migrated[run_id]
        destination = run / 'records' / (run_id + '.jsonl')
        if (checked['n_total'] != n or frozen['n_expected'] != n
                or Path(frozen['destination']).resolve() != destination.resolve()
                or len(frozen['query_ids']) != n or len(set(frozen['query_ids'])) != n
                or artifact(destination) not in checked['artifacts']
                or not verify_artifacts(checked['artifacts'] + frozen['source_artifacts'] + frozen['selection_provenance'])):
            raise ValueError('Accepted GPU records or population changed')
        paths[frozen['prefix']] = destination
        evidence.extend(checked['artifacts'])
    return {'paths': paths, 'evidence': evidence, 'comparison_limit': migration['comparison_limit'],
            'n_failed': validation['n_failed']}
