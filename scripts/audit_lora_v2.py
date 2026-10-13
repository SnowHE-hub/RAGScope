"""Audit completed real LoRA records/checkpoints without loading Torch or weights."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import json
import math
import statistics

from ragscope.experiments.ledger import read_records
from ragscope.indexing.identity import artifact, atomic_json
from ragscope.training.checkpoint import CheckpointStore


def audit(run):
    run = Path(run)
    read = lambda name: json.loads((run/name).read_text(encoding='utf-8'))
    result, config = read('training_result.json'), read('steps.config.json')
    if result['status'] != 'TRAIN_LOOP_COMPLETE' or result['epoch'] != 1:
        raise ValueError('Bounded full epoch is not complete')
    rows = read_records(run/'steps.jsonl')
    expected = math.ceil(len(config['query_ids'])/config['config']['gradient_accumulation'])
    if [row['step'] for row in rows] != list(range(1, expected+1)) or result['actual_optimizer_steps'] != expected:
        raise ValueError('Step sequence missing, duplicated or beyond one epoch')
    if result['data_position'] != len(config['query_ids']) or expected > 200:
        raise ValueError('Training data progress/bounds differ')
    if any(not math.isfinite(row[key]) for row in rows for key in ('loss', 'gradient_norm')):
        raise ValueError('Nonfinite actual loss/gradient')
    initialization = read_records(run/'initialization_receipts.jsonl')
    if len(initialization) < 3 or len({row['initial_trainable_state_sha256'] for row in initialization}) != 1:
        raise ValueError('Independent seed-fixed adapter initialization evidence differs')
    resumes = [read(f'resume_from_step_{step:06d}.json') for step in (5, 10)]
    if any(row['status'] != 'RESTORED_VERIFIED' or not all(row['verification'].values()) for row in resumes):
        raise ValueError('Actual complete-state restoration failed')
    store = CheckpointStore(run/'checkpoints', config)
    checkpoints = sorted(path for path in store.root.glob('step-*') if path.is_dir() and '.partial' not in path.name)
    if len(checkpoints) > 3 or not checkpoints:
        raise ValueError('Owned checkpoint retention bound differs')
    manifests = [store.verify(path) for path in checkpoints]
    best = store.root/read('checkpoints/best.json')['checkpoint']
    store.verify(best)
    final = store.latest()
    summary = {'status': 'PASS_REAL_TRAINING_AUDIT', 'run': str(run),
        'actual_optimizer_steps': expected, 'train_examples': len(config['query_ids']), 'epochs': 1,
        'all_losses_and_gradients_finite': True, 'initialization_processes': len(initialization),
        'initial_adapter_sha256': initialization[0]['initial_trainable_state_sha256'],
        'restoration_steps': [5, 10], 'all_restore_checks_exact': True,
        'best_checkpoint': str(best), 'latest_checkpoint': str(final), 'retained_checkpoints': len(checkpoints),
        'dev_losses': [{key: row[key] for key in ('step', 'dev_loss')} for row in rows if 'dev_loss' in row],
        'first_step_loss': rows[0]['loss'], 'last_step_loss': rows[-1]['loss'],
        'optimizer_step_seconds_sum': sum(row['seconds'] for row in rows),
        'optimizer_step_seconds_p50': statistics.median(row['seconds'] for row in rows),
        'timing_note': 'Step timing excludes checkpoint I/O, dev evaluation and model loading; not total wall time.',
        'max_torch_allocated_bytes': max(row['peak_allocated_bytes'] for row in rows),
        'max_torch_reserved_bytes': max(row['peak_reserved_bytes'] for row in rows),
        'max_sampled_process_rss_bytes': max(row['peak_sampled_process_rss_bytes'] for row in rows),
        'min_sampled_host_available_bytes': min(row['minimum_sampled_host_available_bytes'] for row in rows),
        'gpu_metric_note': 'Torch allocator counters are not physical device VRAM; Windows shared/paging may participate.',
        'same_precision_test_comparison': 'PENDING_FROZEN_E2_R2_CONTEXTS',
        'artifacts': [artifact(run/name) for name in ('steps.jsonl', 'steps.config.json', 'training_result.json',
            'initialization_receipts.jsonl', 'resume_from_step_000005.json', 'resume_from_step_000010.json')]
            +[artifact(path/'manifest.json') for path in checkpoints]+[artifact(best/'adapter/adapter_model.safetensors')]}
    atomic_json(run/'TRAINING_AUDIT.json', summary)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    print(json.dumps(audit(parser.parse_args().run), ensure_ascii=False))
