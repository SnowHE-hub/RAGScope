"""Actual paired old-output decomposition; descriptive, never causal proof."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sqlite3
import statistics
import sys

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
from filelock import FileLock
from ragscope.experiments.runtime import configure, require_ram
from ragscope.indexing.identity import artifact, atomic_json, canonical_json


def median(values):
    values = [v for v in values if isinstance(v, (int, float))]
    return statistics.median(values) if values else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    runtime = Path(configure(cpu_only=True)['home']).resolve()
    run = Path(args.campaign).resolve()
    if run.parent != runtime/'runs' or not run.name.startswith('v3_'):
        raise ValueError('Wrong campaign')
    require_ram(2*1024**3)
    with FileLock(str(run/'old_attribution.lock'), timeout=0):
        receipt = run/'OLD_OUTPUT_ATTRIBUTION.json'
        if receipt.exists():
            saved = json.loads(receipt.read_text('utf-8'))
            if not args.resume or saved['runner'] != artifact(__file__) or any(artifact(x['path']) != x for x in saved['sources']):
                raise ValueError('Attribution identity changed')
            print(json.dumps({'status': 'REUSED_OLD_OUTPUT_ATTRIBUTION'})); return 0
        metrics = {r['record_id']: r for r in (json.loads(l) for l in (run/'RECOUNT_ALL_OLD.jsonl').read_text('utf-8').splitlines())}
        semantic = {r['record_id']: r for r in (json.loads(l) for l in (run/'BERTSCORE_ALL_OLD_nli_base_cpu_smallbatch.jsonl').read_text('utf-8').splitlines())}
        db = sqlite3.connect('file:'+str(run/'S0_EVIDENCE.sqlite')+'?mode=ro', uri=True)
        paired = {}
        for (payload,) in db.execute('SELECT payload FROM records'):
            row = json.loads(payload)
            if (row['experiment_id'], row['method']) not in {('E2', 'R2'), ('E3', 'oracle')} or row['model_id'] not in {'qwen_2b', 'qwen_9b'}:
                continue
            small = {k: row[k] for k in ('query_id', 'doc_id', 'question', 'answer', 'document_answerable', 'model_abstained', 'failed',
                                         'model_id', 'experiment_id', 'method', 'latency_ms', 'context_hash')}
            small.update(metrics[row['record_id']])
            small['bert_score_f1_proxy'] = semantic[row['record_id']]['bert_score_f1_proxy']
            paired[(row['experiment_id'], row['method'], row['query_id'], row['model_id'])] = small
        db.close()
        rows, summaries = [], []
        for exp, method in (('E2', 'R2'), ('E3', 'oracle')):
            qids = sorted({k[2] for k in paired if k[:2] == (exp, method)})
            local = []
            for qid in qids:
                base = paired[(exp, method, qid, 'qwen_2b')]
                large = paired[(exp, method, qid, 'qwen_9b')]
                if base['context_hash'] != large['context_hash']:
                    raise ValueError('Scale pair context differs')
                value = {'query_id': qid, 'experiment_id': exp, 'method': method, 'base': base, 'large': large,
                    'delta_word_f1': large['legacy_token_f1']-base['legacy_token_f1'] if base['document_answerable'] else None,
                    'delta_semantic_proxy': large['bert_score_f1_proxy']-base['bert_score_f1_proxy'] if base['document_answerable'] else None}
                local.append(value); rows.append(value)
            applicable = [r for r in local if r['base']['document_answerable']]
            buckets = defaultdict(list)
            for row in applicable:
                buckets[row['base']['answer_type']].append(row)
            summaries.append({'experiment_id': exp, 'method': method, 'n': len(local), 'applicable_n': len(applicable),
                'delta_word_f1': sum(r['delta_word_f1'] for r in applicable)/len(applicable),
                'delta_semantic_proxy': sum(r['delta_semantic_proxy'] for r in applicable)/len(applicable),
                'large_answerable_abstained_n': sum(r['large']['model_abstained'] for r in applicable),
                'abstention_perfect_repair_f1_upper_gain': sum(r['large']['model_abstained'] for r in applicable)/len(applicable),
                'by_answer_type': {key: {'n': len(group), 'delta_word_f1': sum(r['delta_word_f1'] for r in group)/len(group),
                    'delta_semantic_proxy': sum(r['delta_semantic_proxy'] for r in group)/len(group),
                    'base_output_tokens_p50': median(r['base']['actual_output_tokens'] for r in group),
                    'large_output_tokens_p50': median(r['large']['actual_output_tokens'] for r in group),
                    'base_predicted_words_p50': median(r['base']['predicted_words'] for r in group),
                    'large_predicted_words_p50': median(r['large']['predicted_words'] for r in group)} for key, group in buckets.items()}})
        pairs = [json.loads(l) for l in (run/'LORA_100_PAIRS.jsonl').read_text('utf-8').splitlines()]
        lora_summary = {}
        for condition in ('baseline', 'adapted'):
            values = [metrics['LORA:'+condition+':'+p['query_id']] for p in pairs]
            lora_summary[condition] = {'n': len(values), 'actual_output_tokens_p50': median(v['actual_output_tokens'] for v in values),
                'actual_output_tokens_mean': statistics.mean(v['actual_output_tokens'] for v in values if v['actual_output_tokens'] is not None),
                'at_output_cap_n': sum(v['at_output_limit'] is True for v in values),
                'repeated_4gram_fraction_mean': statistics.mean(v['repeated_4gram_fraction'] for v in values),
                'finish_reason_observed_n': sum(v['finish_reason'] is not None for v in values),
                'latency_ms_p50': median(p[condition]['latency_ms'] for p in pairs),
                'decode_tokens_per_second_p50': median(p[condition]['decode_tokens_per_second'] for p in pairs),
                'first_token_ms_observed_n': sum(p[condition]['first_token_ms'] is not None for p in pairs)}
        output_path = run/'MODEL_SCALE_400_PAIRS.jsonl'
        with output_path.open('x', encoding='utf-8', newline='\n') as handle:
            for row in rows:
                handle.write(canonical_json(row)+'\n')
        output = {'status': 'PASS_DESCRIPTIVE_DECOMPOSITION', 'runner': artifact(__file__), 'scale': summaries,
            'lora_length_latency': lora_summary, 'paired_rows': artifact(output_path),
            'sources': [artifact(run/name) for name in ('S0_EVIDENCE.sqlite', 'RECOUNT_ALL_OLD.jsonl',
                'BERTSCORE_ALL_OLD_nli_base_cpu_smallbatch.jsonl', 'LORA_100_PAIRS.jsonl')],
            'human_review_status': 'NOT_REVIEWED', 'new_llm_requests': 0,
            'causal_limit': 'Semantic proxy ordering does not prove factual ordering; longer output alone does not explain total latency. Missing EOS/TTFT stays missing.'}
        atomic_json(receipt, output)
        print(json.dumps({k: v for k, v in output.items() if k not in ('runner', 'sources', 'paired_rows')}, ensure_ascii=False))
        return 0


if __name__ == '__main__':
    raise SystemExit(main())
