"""Audit preserved v2 journals on CPU without making model requests."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))

from filelock import FileLock
from ragscope.experiments.runtime import configure, require_disk, require_ram
from ragscope.indexing.identity import artifact, atomic_json, signature
from ragscope.v3.audit import add_inventory, audit_generation, audit_lora, preserve_official_evaluator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    env = configure(cpu_only=True)
    runtime = Path(env['home']).resolve()
    run = Path(args.campaign).resolve()
    if not run.is_relative_to(runtime / 'runs') or not run.name.startswith('v3_'):
        raise ValueError('Audit must target its identified v3 campaign')
    identity = json.loads((run / 'CAMPAIGN_IDENTITY.json').read_text('utf-8'))
    if Path(identity['project']).resolve() != PROJECT or identity['campaign_id'] != run.name:
        raise ValueError('Campaign/repository identity mismatch')
    for ref in (identity['plan'], identity['budget_spec'], identity['diagnosis']):
        if artifact(ref['path']) != ref:
            raise ValueError('Approved plan/budget/diagnosis changed')
    config = {'campaign_id': run.name, 'auditor': artifact(__file__),
              'audit_implementation': artifact(PROJECT / 'ragscope/v3/audit.py'), 'source_head': identity['source_head']}
    receipt = run / 'S0_AUDIT_RECEIPT.json'
    with FileLock(str(run / 'writer.lock'), timeout=0):
        if args.resume and receipt.exists():
            saved = json.loads(receipt.read_text('utf-8'))
            if saved['config_hash'] != signature(config):
                raise ValueError('Auditor identity changed; retain old receipt and create a new experiment')
            for ref in saved['outputs']:
                if artifact(ref['path']) != ref:
                    raise ValueError('Completed audit artifact changed')
            print(json.dumps({'status': 'REUSED_VALIDATED_AUDIT', 'new_llm_requests': 0}))
            return 0
        memory = require_ram(2 * 1024**3)
        disk = require_disk()
        started = time.time()
        inventory = {}
        baseline = json.loads((run / 'BASELINE_PRESERVATION.json').read_text('utf-8'))
        for ref in baseline['tracked_sources']:
            entry = add_inventory(inventory, ref['path'], 'preserved_v2_tracked_source', ref)
            if entry['status'] != 'FOUND':
                raise ValueError('Preserved v2 source changed: ' + ref['path'])
        for relative in ['reports/v2/ANALYSIS.json', 'reports/v2/ERROR_CASES.json',
                'docs/execution/DECISIONS.md', 'docs/execution/RECOVERY.md', 'LOCAL_RUN_V2.md']:
            add_inventory(inventory, PROJECT / relative, 'v2_diagnostic_source')
        for filename in ['all_documents.jsonl', 'all_queries_audit.jsonl', 'documents.jsonl',
                'queries_train.jsonl', 'queries_dev.jsonl', 'queries_test.jsonl', 'queries_global.jsonl',
                'source_manifest.json', 'SPLIT_MANIFEST.json', 'FROZEN_CONFIG.json']:
            add_inventory(inventory, runtime / 'corpora/20261014_v2' / filename, 'v2_corpus_source')
        for directory in sorted((runtime / 'envs').iterdir()):
            if directory.is_dir() and not directory.name.startswith('v3_'):
                add_inventory(inventory, directory / 'pyvenv.cfg', 'existing_environment_identity')
        lora = runtime / 'runs/20261014_v2_lora'
        for sub in ['prepared_2b_1024', 'prepared_2b_1024_v2', 'train_2b_1024_seed42']:
            for path in sorted((lora / sub).glob('*.json*')):
                add_inventory(inventory, path, 'lora_supervision_or_training_source')
        official = preserve_official_evaluator(run, runtime, inventory)
        recount = audit_generation(run, runtime, inventory)
        lora_result = audit_lora(run, runtime, inventory)
        atomic_json(run / 'SOURCE_INVENTORY.json', {'entries': list(inventory.values()),
            'status_counts': dict(__import__('collections').Counter(r['status'] for r in inventory.values())),
            'note': 'Environment and weight metadata still require dedicated identity audit; no model imports.',
            'official_evaluator': artifact(official)})
        outputs = [run / name for name in ['SOURCE_INVENTORY.json', 'BASELINE_RECOUNT.json',
            'LORA_OLD_RECOUNT.json', 'LORA_100_PAIRS.jsonl', 'S0_EVIDENCE.sqlite']]
        result = {'status': 'PASS' if recount['status'] == 'PASS' else 'INCOMPLETE',
            'config_hash': signature(config), 'config': config, 'outputs': [artifact(p) for p in outputs],
            'elapsed_seconds': time.time() - started, 'cpu_only': True, 'threads': env['threads'],
            'starting_memory': memory, 'starting_disk': disk, 'new_llm_requests': 0,
            'observed_records': recount['observed_n'], 'failures': recount['failures'],
            'lora_paired_n': lora_result['conditions']['baseline']['n'],
            'exit_code': 0 if recount['status'] == 'PASS' else 2}
        atomic_json(receipt, result)
        state = json.loads((run / 'STATUS.json').read_text('utf-8'))
        state.update(updated_utc=datetime.now(timezone.utc).isoformat(),
            source_audit_receipt=artifact(receipt), next_command_status='IMPLEMENT_NEXT',
            next_command='Implement CPU-only scripts/v3/seal_splits.py; seal paper-disjoint D_confirm/H_QA before candidate selection')
        atomic_json(run / 'STATUS.json', state)
        atomic_json(PROJECT / 'docs/execution/v3/STATUS.json', state)
        print(json.dumps({k: v for k, v in result.items() if k not in ('outputs', 'config')}, ensure_ascii=False))
        return result['exit_code']


if __name__ == '__main__':
    raise SystemExit(main())
