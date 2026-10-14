"""Score every old answer with pinned NLI, full windows and explicit bounds."""
from __future__ import annotations
import argparse
from collections import defaultdict
from contextlib import nullcontext
from functools import lru_cache
import importlib.metadata
import json
from pathlib import Path
import sqlite3
import sys
import time

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
from filelock import FileLock
from ragscope.experiments.runtime import configure, require_ram
from ragscope.experiments.ledger import RecordStore, read_records
from ragscope.indexing.identity import artifact, atomic_json, signature
from ragscope.v3.claim_metrics import bounds, score_claims, retained_failure_metrics
from ragscope.v3.gpu import GpuSession
from ragscope.v3.nli import WindowedNLI


class CachedJudge:
    def __init__(self, nli, store, *, completion_gate, gpu=None):
        self.nli, self.store, self.completion_gate, self.gpu = nli, store, completion_gate, gpu
        self.index = {row['record_id']: row['result'] for row in store.records}
        self.last_gpu_check = 0.0

    def __call__(self, premise, hypothesis):
        if not premise['eligible'] or not hypothesis['eligible']:
            return {'verdict': 'UNKNOWN', 'reason': 'fragment_or_composite_not_calibrated'}
        if not self.completion_gate and any(p.get('reason', '').startswith('mechanical_') for p in (premise, hypothesis)):
            return {'verdict': 'UNKNOWN', 'reason': 'qa_completion_calibration_failed'}
        rid = signature([premise['text'], hypothesis['text']])
        if rid not in self.index:
            require_ram(2*1024**3)
            if self.gpu and time.monotonic()-self.last_gpu_check >= 10:
                self.gpu.check()
                self.last_gpu_check = time.monotonic()
            result = self.nli.classify(premise['text'], hypothesis['text'])
            self.store.add(rid, {'premise_hash': signature(premise['text']),
                'hypothesis_hash': signature(hypothesis['text']), 'premise_chars': len(premise['text']),
                'hypothesis_chars': len(hypothesis['text']), 'result': result,
                'human_review_status': 'NOT_REVIEWED'})
            self.index[rid] = result
        return self.index[rid]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--max-records', type=int, help='Bounded smoke; same frozen scoring identity can resume')
    args = parser.parse_args()
    runtime = Path(configure(cpu_only=True)['home']).resolve()
    run = Path(args.campaign).resolve()
    if run.parent != runtime/'runs' or not run.name.startswith('v3_'):
        raise ValueError('Wrong campaign')
    require_ram(2*1024**3)
    with FileLock(str(run/'writer.lock'), timeout=0):
        inputs_path, quotes_path = run/'ATOMIZATION_INPUTS.jsonl', run/'ATOMIZATION_QUOTES_V4.jsonl'
        groups = [json.loads(line) for line in inputs_path.read_text('utf-8').splitlines()]
        quotes = read_records(quotes_path)
        if {r['record_id'] for r in quotes} != {g['group_id'] for g in groups}:
            raise ValueError('Segmentation not complete; no partial identity adopted')
        quote_by_id = {r['record_id']: r for r in quotes}
        units_by_source = {}
        segmentation_failures = set()
        for group in groups:
            quote = quote_by_id[group['group_id']]
            if quote['input_hash'] != group['input_hash']:
                raise ValueError('Quote group input identity differs')
            if quote['error'] is not None:
                for source in group['texts']:
                    units_by_source[source['source_id']] = [{'start': 0, 'end': len(source['text']),
                        'kind': 'unclear', 'source_quote': source['text']}]
                    segmentation_failures.add(source['source_id'])
            else:
                units_by_source.update({s['source_id']: s['units'] for s in quote['segments']})
        joins = {r['record_id']: r for r in read_records(run/'ATOMIZATION_PRIVATE_JOIN.jsonl')}
        models = json.loads((run/'METRIC_MODEL_MANIFEST.json').read_text('utf-8'))
        chosen = next(m for m in models['models'] if m['model'].startswith('MoritzLaurer/'))
        gpu_suite_path = run/'NLI_GPU_SCORING_CALIBRATION.json'
        completion_path = gpu_suite_path if args.device == 'cuda' else run/'nli_completion_calibration/SUMMARY.json'
        completion = json.loads(completion_path.read_text('utf-8'))
        declarative = json.loads((run/'JUDGE_CALIBRATION_NLI_cpu.json').read_text('utf-8'))
        if not declarative['engineering_gate_90_percent_passed']:
            raise ValueError('Declarative engineering gate failed')
        if args.device == 'cuda' and not completion['declarative_eligible']:
            raise ValueError('Actual GPU declarative calibration failed')
        if args.device == 'cuda' and any(completion['config'][k] != importlib.metadata.version(k) for k in ('torch', 'transformers')):
            raise ValueError('GPU judge dependency stack differs from calibration')
        lora_path = runtime/'runs/20261014_v2_lora/train_2b_1024_seed42/paired_test100/FROZEN_EVALUATION_INPUTS.json'
        config = {'database': artifact(run/'S0_EVIDENCE.sqlite'), 'lora_pairs': artifact(run/'LORA_100_PAIRS.jsonl'),
            'lora_frozen_inputs': artifact(lora_path),
            'segmentation': artifact(quotes_path), 'segmentation_config': artifact(quotes_path.with_suffix('.config.json')),
            'source_join': artifact(run/'ATOMIZATION_PRIVATE_JOIN.jsonl'), 'model': chosen,
            'completion_calibration': artifact(completion_path), 'declarative_calibration': artifact(run/'JUDGE_CALIBRATION_NLI_cpu.json'),
            'implementation': artifact(PROJECT/'ragscope/v3/claim_metrics.py'), 'nli': artifact(PROJECT/'ragscope/v3/nli.py'),
            'runner': artifact(__file__), 'device': args.device, 'batch_size': 4, 'maximum': 512,
            'stride': 64, 'threshold': .8, 'dtype': 'float32', 'expected_old_answers': 5720,
            'reference_selection': 'best lower F1; upper F1 tie break; then first legal reference',
            'fragment_rule': 'UNKNOWN except separately calibrated reference-blind polar/numeric templates',
            'unknowns': 'point estimate null; complete lower/upper retained; failed applicable QA zero',
            'torch': importlib.metadata.version('torch'), 'transformers': importlib.metadata.version('transformers'),
            'human_review_status': 'NOT_REVIEWED', 'new_llm_requests': 0}
        parent_folder = run/('claim_scores_'+args.device+'_v2')
        if (parent_folder/'answers.jsonl').exists():
            parent_config = json.loads((parent_folder/'answers.config.json').read_text('utf-8'))
            if any(parent_config.get(k) != v for k, v in config.items() if k != 'runner'):
                raise ValueError('Parent scoring semantics differ; no partial-result adoption')
            config['reused_parent'] = {'answers': artifact(parent_folder/'answers.jsonl'),
                'config': artifact(parent_folder/'answers.config.json'),
                'relationships': artifact(parent_folder/'relationships.jsonl') if (parent_folder/'relationships.jsonl').exists() else None,
                'repair': 'stream old rows instead of retaining all decoded payloads; same float32 model and scoring semantics'}
        folder = run/('claim_scores_'+args.device+'_v3')
        folder.mkdir(exist_ok=True)
        store = RecordStore(folder/'answers.jsonl', config, max_records=5720)
        cache = RecordStore(folder/'relationships.jsonl', {**config, 'purpose': 'input-hash keyed NLI relationship cache'}, max_records=100000)
        # An all-UNKNOWN smoke can legitimately execute zero relationships.
        cache.path.touch(exist_ok=True)
        if 'reused_parent' in config:
            for number, row in enumerate(read_records(parent_folder/'answers.jsonl'), 1):
                if row['config_hash'] != signature(parent_config):
                    raise ValueError('Parent answer record configuration differs')
                body = {k: v for k, v in row.items() if k not in ('record_id', 'record_hash', 'config_hash', 'recorded_at')}
                body['inherited_source'] = {'path': str(parent_folder/'answers.jsonl'), 'line': number, 'record_hash': row['record_hash']}
                store.add(row['record_id'], body)
            if (parent_folder/'relationships.jsonl').exists():
                expected_cache_config = {**parent_config, 'purpose': 'input-hash keyed NLI relationship cache'}
                if json.loads((parent_folder/'relationships.config.json').read_text('utf-8')) != expected_cache_config:
                    raise ValueError('Parent relationship cache configuration differs')
                for number, row in enumerate(read_records(parent_folder/'relationships.jsonl'), 1):
                    if row['config_hash'] != signature(expected_cache_config):
                        raise ValueError('Parent relationship record configuration differs')
                    body = {k: v for k, v in row.items() if k not in ('record_id', 'record_hash', 'config_hash', 'recorded_at')}
                    body['inherited_source'] = {'path': str(parent_folder/'relationships.jsonl'), 'line': number, 'record_hash': row['record_hash']}
                    cache.add(row['record_id'], body)
        db = sqlite3.connect('file:'+str(run/'S0_EVIDENCE.sqlite')+'?mode=ro', uri=True)
        lora_inputs = {x['query']['query_id']: x['context_bundle'] for x in json.loads(lora_path.read_text('utf-8'))['inputs']}
        @lru_cache(maxsize=32)
        def input_pieces(iid):
            return json.loads(db.execute('SELECT payload FROM inputs WHERE input_id=?', (iid,)).fetchone()[0])['context_bundle']['pieces']
        def old_rows():
            for (payload,) in db.execute('SELECT payload FROM records'):
                yield json.loads(payload)
            for pair in read_records(run/'LORA_100_PAIRS.jsonl'):
                for condition in ('baseline', 'adapted'):
                    row = dict(pair[condition], record_id='LORA:'+condition+':'+pair['query_id'])
                    yield row
        target_count = 5720 if args.max_records is None else args.max_records
        if target_count < 1 or target_count > 5720:
            raise ValueError('Invalid bounded record count')
        started = time.monotonic()
        if len(store.ids) < target_count:
            with GpuSession(run) if args.device == 'cuda' else nullcontext() as gpu:
                nli = WindowedNLI(chosen['directory'], device='cuda:0' if gpu else 'cpu')
                judge = CachedJudge(nli, cache, completion_gate=completion['completion_eligible'], gpu=gpu)
                try:
                    for row in old_rows():
                        rid = row['record_id']
                        if rid in store.ids:
                            continue
                        if len(store.ids) >= target_count:
                            break
                        require_ram(2*1024**3)
                        if row['failed'] or row['model_abstained'] or not row['answer'].strip():
                            metrics = retained_failure_metrics('FAILED_OR_ABSTAINED_OR_EMPTY')
                        else:
                            source_id = joins[rid]['source_id']
                            units = units_by_source[source_id]
                            pieces = input_pieces(row['input_id']) if row['input_id'] is not None else lora_inputs[row['query_id']]['pieces']
                            metrics = score_claims(row['question'], row['answer'], units, row['answer_refs'],
                                                   pieces, row['citations'], judge)
                            metrics['segmentation_failed_retained_as_unclear'] = source_id in segmentation_failures
                        if not row['document_answerable']:
                            metrics['claim_reference_precision_proxy'] = bounds([])
                            metrics['key_fact_recall_proxy'] = bounds([])
                            for key in ('claim_f1_proxy', 'claim_f1_proxy_lower', 'claim_f1_proxy_upper'):
                                metrics[key] = None
                        body = {k: row[k] for k in ('query_id', 'doc_id', 'experiment_id', 'method', 'model_id',
                            'question', 'answer', 'answer_refs', 'document_answerable', 'model_abstained',
                            'failed', 'context_hash', 'source_record_hash')}
                        store.add(rid, {**body, **metrics, 'visible_context_sufficient': 'UNKNOWN',
                            'context_sufficiency_source': 'not inferred from answer agreement or evidence coverage',
                            'human_review_status': 'NOT_REVIEWED', 'new_llm_requests': 0})
                        if len(store.ids) % 25 == 0:
                            print(json.dumps({'event': 'CLAIM_METRICS_PROGRESS', 'answers': len(store.ids),
                                              'relationships': len(cache.ids)}), flush=True)
                finally:
                    nli.close()
        db.close()
        summary_groups = defaultdict(list)
        for row in store.records:
            summary_groups[(row['experiment_id'], row['model_id'], row['method'])].append(row)
        summaries = []
        for (exp, model, method), rows in sorted(summary_groups.items()):
            applicable = [r for r in rows if r['document_answerable']]
            n = len(applicable)
            summaries.append({'experiment_id': exp, 'model_id': model, 'method': method,
                'n': len(rows), 'applicable_n': n,
                'claim_f1_lower': sum(r['claim_f1_proxy_lower'] or 0 for r in applicable)/n if n else None,
                'claim_f1_upper': sum(r['claim_f1_proxy_upper'] or 0 for r in applicable)/n if n else None,
                'point_estimate_unknown_n': sum(r['claim_f1_proxy'] is None for r in applicable)})
        output = {'status': 'PASS_FULL_SCORING_WITH_UNCERTAINTY' if len(store.ids) == 5720 else 'PARTIAL_SMOKE',
            'answers': len(store.ids), 'expected': 5720, 'relationships': len(cache.ids), 'groups': summaries,
            'encoder_tokens': sum(r['result']['encoder_tokens'] for r in cache.records),
            'elapsed_seconds_this_invocation': time.monotonic()-started,
            'outputs': [artifact(store.path), artifact(cache.path)], 'config': config,
            'human_review_status': 'NOT_REVIEWED', 'new_llm_requests': 0}
        atomic_json(folder/'SUMMARY.json', output)
        print(json.dumps({k: v for k, v in output.items() if k not in ('groups', 'outputs', 'config')}))
        return 0


if __name__ == '__main__':
    raise SystemExit(main())
