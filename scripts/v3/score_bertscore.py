"""Run actual official BERTScore on every applicable preserved old answer."""
from __future__ import annotations
import argparse
from collections import defaultdict
from contextlib import nullcontext
import gc
import importlib.metadata
import json
import os
from pathlib import Path
import sqlite3
import sys
import time

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
from filelock import FileLock
from ragscope.experiments.runtime import configure, require_ram
from ragscope.experiments.ledger import RecordStore
from ragscope.indexing.identity import artifact, atomic_json, canonical_json, signature
from ragscope.v3.gpu import GpuSession


def old_records(run):
    db = sqlite3.connect(run / 'S0_EVIDENCE.sqlite')
    records = [json.loads(payload) for (payload,) in db.execute('SELECT payload FROM records ORDER BY record_id')]
    db.close()
    with (run / 'LORA_100_PAIRS.jsonl').open(encoding='utf-8') as handle:
        for line in handle:
            pair = json.loads(line)
            for condition in ('baseline', 'adapted'):
                row = pair[condition]
                row['record_id'] = 'LORA:' + condition + ':' + row['query_id']
                records.append(row)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    args = parser.parse_args()
    runtime = Path(configure(cpu_only=True)['home']).resolve()
    run = Path(args.campaign).resolve()
    if not run.is_relative_to(runtime / 'runs') or not run.name.startswith('v3_'):
        raise ValueError('Wrong campaign')
    require_ram(2 * 1024**3)
    manifest = json.loads((run / 'METRIC_MODEL_MANIFEST.json').read_text('utf-8'))
    chosen = next(m for m in manifest['models'] if m['model'] == 'FacebookAI/roberta-large')
    package_version = importlib.metadata.version('bert-score')
    if package_version != '0.3.13':
        raise ValueError('Metric implementation version differs')
    config = {'implementation_version': package_version, 'model': chosen['model'], 'revision': chosen['revision'],
        'model_artifacts': chosen['artifacts'], 'num_layers': 17, 'idf': False, 'rescale_with_baseline': False,
        'use_fast_tokenizer': False, 'dtype': 'float32', 'device': 'cuda:0' if args.device == 'cuda' else 'cpu', 'batch_size': 4,
        'multi_reference_rule': 'single reference maximizing F1; precision/recall from that same reference; first tie',
        'source_db': artifact(run / 'S0_EVIDENCE.sqlite'), 'lora_projection': artifact(run / 'LORA_100_PAIRS.jsonl'),
        'transformers': importlib.metadata.version('transformers'), 'torch': importlib.metadata.version('torch'),
        'scorer': artifact(__file__), 'failure_policy': 'answerable failures/abstentions zero; raw semantic score also retained',
        'human_review_status': 'NOT_REVIEWED', 'new_llm_requests': 0}
    with FileLock(str(run / 'writer.lock'), timeout=0):
        suffix = '_cpu' if args.device == 'cpu' else ''
        receipt = run / ('S1_BERTSCORE_RECEIPT' + suffix + '.json')
        if receipt.exists():
            saved = json.loads(receipt.read_text('utf-8'))
            if not args.resume or saved['config_hash'] != signature(config):
                raise ValueError('Metric config changed; create a separate experiment')
            for ref in saved['outputs']:
                if artifact(ref['path']) != ref:
                    raise ValueError('Completed BERTScore artifact changed')
            print(json.dumps({'status': 'REUSED_BERTSCORE'}))
            return 0
        records = old_records(run)
        pairs, memberships = {}, {}
        for row in records:
            ids = []
            if row['document_answerable'] and not row['failed']:
                for ref in row['answer_refs']:
                    key = signature([row['answer'], ref])
                    pairs[key] = (row['answer'], ref)
                    ids.append(key)
            memberships[row['record_id']] = ids
        store = RecordStore(run / ('BERTSCORE_PAIRS' + suffix + '.jsonl'), config, max_records=len(pairs))
        if any(r['record_id'] not in pairs for r in store.records):
            raise ValueError('Cached metric pair outside frozen source population')
        pending = sorted(set(pairs) - store.ids)
        started = time.monotonic()
        encoded_tokens, truncated_pairs = 0, 0
        if pending:
            with GpuSession(run) if args.device == 'cuda' else nullcontext() as gpu:
                import torch
                from bert_score import BERTScorer
                torch.set_num_threads(4)
                if gpu:
                    torch.backends.cuda.matmul.allow_tf32 = False
                    torch.backends.cudnn.allow_tf32 = False
                scorer = BERTScorer(model_type=chosen['directory'], num_layers=17, batch_size=4,
                    nthreads=4, lang='en', device=config['device'], idf=False,
                    rescale_with_baseline=False, use_fast_tokenizer=False)
                maximum = scorer._tokenizer.model_max_length
                for start in range(0, len(pending), 64):
                    require_ram(2 * 1024**3)
                    if gpu:
                        gpu.check()
                    keys = pending[start:start+64]
                    candidate = [pairs[k][0] for k in keys]
                    refs = [pairs[k][1] for k in keys]
                    token_counts = [(len(scorer._tokenizer.encode(c, truncation=False)),
                                     len(scorer._tokenizer.encode(r, truncation=False))) for c, r in zip(candidate, refs)]
                    p, r, f = scorer.score(candidate, refs, batch_size=4)
                    for i, key in enumerate(keys):
                        c_len, r_len = token_counts[i]
                        clipped = max(c_len, r_len) > maximum
                        store.add(key, {'precision': float(p[i]), 'recall': float(r[i]), 'f1': float(f[i]),
                            'candidate_tokens': c_len, 'reference_tokens': r_len, 'maximum_tokens': maximum,
                            'truncated_by_official_implementation': clipped, 'real_encoder': True})
                        encoded_tokens += min(c_len, maximum) + min(r_len, maximum)
                        truncated_pairs += int(clipped)
                    print(json.dumps({'event': 'BERTSCORE_PROGRESS', 'completed_pairs': len(store.ids),
                        'expected_pairs': len(pairs), 'elapsed_seconds': time.monotonic()-started}), flush=True)
                del scorer
                gc.collect()
                if gpu:
                    torch.cuda.empty_cache()
        cache = {r['record_id']: r for r in store.records}
        groups, outputs = defaultdict(list), []
        for row in records:
            values = [cache[key] for key in memberships[row['record_id']]]
            best = max(values, key=lambda v: v['f1'], default=None)
            f1 = None if not row['document_answerable'] else 0.0 if row['failed'] or row['model_abstained'] else best['f1'] if best else 0.0
            out = {'record_id': row['record_id'], 'query_id': row['query_id'], 'doc_id': row['doc_id'],
                'experiment_id': row['experiment_id'], 'model_id': row['model_id'], 'method': row['method'],
                'source_record_hash': row['source_record_hash'], 'bert_score_f1_proxy': f1,
                'raw_reference_semantic_score': {k: best[k] for k in ('precision','recall','f1')} if best else None,
                'failed': row['failed'], 'model_abstained': row['model_abstained'],
                'document_answerable': row['document_answerable'],
                'any_reference_pair_truncated': any(v['truncated_by_official_implementation'] for v in values),
                'config_hash': signature(config), 'human_review_status': 'NOT_REVIEWED'}
            outputs.append(out)
            groups[(row['experiment_id'], row['model_id'], row['method'])].append(out)
        path = run / ('BERTSCORE_ALL_OLD' + suffix + '.jsonl')
        with path.open('w', encoding='utf-8', newline='\n') as handle:
            for row in outputs:
                handle.write(canonical_json(row)+'\n')
            handle.flush()
            os.fsync(handle.fileno())
        summary = []
        for key, rows in sorted(groups.items()):
            values = [r['bert_score_f1_proxy'] for r in rows if r['bert_score_f1_proxy'] is not None]
            summary.append({'experiment_id': key[0], 'model_id': key[1], 'method': key[2], 'n': len(rows),
                'applicable_n': len(values), 'bert_score_f1_proxy': sum(values)/len(values) if values else None})
        summary_path = run / ('BERTSCORE_ALL_OLD' + suffix + '.json')
        atomic_json(summary_path, {'status': 'PASS', 'n': len(outputs), 'unique_pair_n': len(pairs),
            'groups': summary, 'truncated_pairs': sum(r['truncated_by_official_implementation'] for r in cache.values()),
            'encoder_tokens_this_invocation': encoded_tokens, 'elapsed_seconds': time.monotonic()-started,
            'config': config, 'new_llm_requests': 0, 'proxy_not_fact_truth': True})
        atomic_json(receipt, {'status': 'PASS', 'config_hash': signature(config),
            'outputs': [artifact(path), artifact(summary_path), artifact(store.path), artifact(store.path.with_suffix('.config.json'))],
            'actual_exit_code': 0, 'new_llm_requests': 0})
        print(json.dumps({'status': 'BERTSCORE_COMPLETE', 'n': len(outputs), 'unique_pairs': len(pairs),
                          'new_llm_requests': 0}))
        return 0


if __name__ == '__main__':
    raise SystemExit(main())
