"""Rescore all old answers, freeze blinded sampling, and audit EOS supervision.

No inference, model imports, holdout gold reads, or edits to v2 artifacts.
Semantic model scores are explicitly pending until actual models are run.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import importlib.util
import json
import math
import os
from pathlib import Path
import sqlite3
import statistics
import sys

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
from filelock import FileLock
from ragscope.experiments.runtime import configure, require_ram
from ragscope.indexing.identity import artifact, atomic_json, canonical_json, signature
from ragscope.v3.audit import journal_rows, normalized_words, overlap


def write_lines(path, rows):
    with path.open('x', encoding='utf-8', newline='\n') as handle:
        for row in rows:
            handle.write(canonical_json(row) + '\n')
        handle.flush()
        os.fsync(handle.fileno())


def observed_output_tokens(record):
    backend = record['backend_metadata']
    if isinstance(backend.get('actual_output_tokens'), int):
        return backend['actual_output_tokens'], 'backend.actual_output_tokens'
    if isinstance(backend.get('eval_count'), int):
        return backend['eval_count'], 'backend.eval_count'
    return None, 'NOT_RECORDED; reference tokenizer count is not actual generation count'


def repeating_ngram_fraction(text, n=4):
    words = normalized_words(text)
    counts = Counter(tuple(words[i:i+n]) for i in range(max(0, len(words)-n+1)))
    total = sum(counts.values())
    return sum(c-1 for c in counts.values()) / total if total else 0.0


def supervision_audit(runtime):
    model = runtime / 'models/Qwen3.5-2B/15852e8c16360a2fea060d615a32b45270f8a8fc'
    tc = json.loads((model / 'tokenizer_config.json').read_text('utf-8'))
    mc = json.loads((model / 'config.json').read_text('utf-8'))
    token_name = tc['eos_token']
    ids = [int(k) for k, v in tc['added_tokens_decoder'].items() if v['content'] == token_name]
    if len(ids) != 1:
        raise ValueError('Tokenizer EOS identity ambiguous')
    eos = ids[0]
    model_eos = mc.get('text_config', mc).get('eos_token_id')
    prepared = runtime / 'runs/20261014_v2_lora/prepared_2b_1024_v2'
    examples, counts, lengths = [], Counter(), []
    with (prepared / 'train_examples.jsonl').open(encoding='utf-8') as handle:
        rows = [json.loads(line) for line in handle]
    selected = sorted(rows, key=lambda r: signature([42, 'supervision_audit_20', r['query_id']]))[:20]
    for row in rows:
        counts[row['answerability']] += 1
        lengths.append(row['total_tokens'])
    for row in selected:
        ids, labels, attention = row['input_ids'], row['labels'], row['attention_mask']
        boundary = row['prompt_tokens']
        supervised = [i for i, v in enumerate(labels) if v != -100]
        checks = {'all_lengths_equal': len(ids) == len(labels) == len(attention) == row['total_tokens'],
            'prompt_masked': all(v == -100 for v in labels[:boundary]),
            'target_matches_input': all(labels[i] == ids[i] for i in supervised),
            'padding_not_supervised': all(labels[i] == -100 for i, mask in enumerate(attention) if mask == 0),
            'supervised_count_matches': len(supervised) == row['supervised_tokens'],
            'tokenizer_eos_supervised': any(ids[i] == eos for i in supervised),
            'whole_example_below_1024': len(ids) <= 1024,
            'messages_hash_matches': signature(row['messages']) == row['messages_hash']}
        examples.append({'query_id': row['query_id'], 'doc_id': row['doc_id'], 'checks': checks,
            'prompt_tokens': boundary, 'total_tokens': len(ids), 'supervised_tokens': len(supervised),
            'assistant_token_fraction': len(supervised) / len(ids),
            'last_token_ids': ids[-6:], 'last_label_ids': labels[-6:],
            'evidence_policy': row['evidence_policy']})
    return {'status': 'PASS' if all(all(r['checks'].values()) for r in examples) else 'FAIL',
        'inspected_n': len(examples), 'train_n': len(rows), 'train_answerability_counts': counts,
        'total_tokens_p50': statistics.median(lengths), 'total_tokens_max': max(lengths),
        'tokenizer_eos_token': token_name, 'tokenizer_eos_id': eos, 'model_config_eos_id': model_eos,
        'eos_ids_differ': model_eos != eos,
        'eos_runtime_verdict': 'HYPOTHESIS_ONLY: actual model generation_config and stopping must be inspected in S5',
        'checks': examples, 'sources': [artifact(prepared / 'train_examples.jsonl'), artifact(model / 'config.json'),
            artifact(model / 'tokenizer_config.json'), artifact(PROJECT / 'ragscope/training/evaluate.py')],
        'notes': 'Numeric mask checks do not replace tokenizer decoding or actual S5 supervision smoke.'}


def calibration_groups():
    groups = []
    for i in range(10):
        entity = f'Experiment {i+1}'
        groups.extend([
            {'category': 'paraphrase', 'premise': f'{entity} purchased a car.',
             'reference': f'{entity} bought an automobile.',
             'positive': f'{entity} purchased an automobile.', 'negative': f'{entity} sold an automobile.',
             'negative_relation': 'contradiction'},
            {'category': 'numeric', 'premise': f'{entity} used exactly {128+i} training samples.',
             'reference': f'The sample count for {entity} was {128+i}.',
             'positive': f'{entity} used {128+i} training samples.',
             'negative': f'{entity} used {129+i} training samples.', 'negative_relation': 'contradiction'},
            {'category': 'negation', 'premise': f'{entity} did not use dataset Delta.',
             'reference': f'Dataset Delta was not used by {entity}.',
             'positive': f'{entity} did not use Delta.', 'negative': f'{entity} used dataset Delta.',
             'negative_relation': 'contradiction'},
            {'category': 'comparison_direction', 'premise': f'In {entity}, method Alpha scored 14 and method Beta scored 12. Higher scores are better.',
             'reference': 'Alpha scored higher than Beta.', 'positive': 'Alpha performed better than Beta.',
             'negative': 'Beta scored higher than Alpha.', 'negative_relation': 'contradiction'},
            {'category': 'paper_and_citation', 'premise': f'Paper Alpha used {128+i} samples. Paper Beta used {256+i} samples.',
             'reference': f'Paper Alpha used {128+i} samples.',
             'positive': f'Paper Alpha used {128+i} samples.', 'negative': f'Paper Alpha used {256+i} samples.',
             'negative_relation': 'contradiction',
             'citation_positive_text': f'Paper Alpha used {128+i} samples.',
             'citation_negative_text': f'Paper Beta used {256+i} samples.',
             'wrong_citation_for_positive_claim_expected': 'unsupported'},
            {'category': 'insufficient_evidence', 'premise': f'{entity} evaluated method Alpha on dataset Delta.',
             'reference': f'{entity} evaluated Alpha on Delta.',
             'positive': f'{entity} evaluated Alpha on Delta.',
             'negative': f'{entity} used a learning rate of 0.0001.', 'negative_relation': 'neutral'},
        ])
    for i, group in enumerate(groups):
        group.update(group_id=f'CAL_{i+1:03d}', positive_relation='entailment',
            truth_source='DETERMINISTIC_CONSTRUCTION_NOT_HUMAN_REVIEW', human_review_status='NOT_REVIEWED')
    return groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    runtime = Path(configure(cpu_only=True)['home']).resolve()
    run = Path(args.campaign).resolve()
    if not run.is_relative_to(runtime / 'runs') or not run.name.startswith('v3_'):
        raise ValueError('Wrong v3 campaign')
    require_ram(2 * 1024**3)
    with FileLock(str(run / 'writer.lock'), timeout=0):
        receipt = run / 'S1_OLD_METRICS_RECEIPT.json'
        source = artifact(__file__)
        if receipt.exists():
            saved = json.loads(receipt.read_text('utf-8'))
            if not args.resume or saved['script'] != source:
                raise ValueError('Completed metric experiment cannot be overwritten')
            for item in saved['outputs']:
                if artifact(item['path']) != item:
                    raise ValueError('Completed metric artifact changed')
            print(json.dumps({'status': 'REUSED', 'new_llm_requests': 0}))
            return 0
        official_path = run / 'source_package/qasper_evaluator_v0.3.py'
        spec = importlib.util.spec_from_file_location('pinned_official_qasper', official_path)
        official = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(official)
        db = sqlite3.connect(run / 'S0_EVIDENCE.sqlite')
        queries = {qid: json.loads(payload) for qid, payload in db.execute('SELECT query_id,payload FROM queries')}
        records = [json.loads(payload) for (payload,) in db.execute('SELECT payload FROM records ORDER BY record_id')]
        db.close()
        for _, pair in journal_rows(run / 'LORA_100_PAIRS.jsonl', require_hash=False):
            for tag in ('baseline', 'adapted'):
                r = pair[tag]
                r['record_id'] = 'LORA:' + tag + ':' + r['query_id']
                records.append(r)
        scored, differences, buckets = [], [], defaultdict(list)
        for record in records:
            query = queries[record['query_id']]
            refs = record['answer_refs']
            official_refs = refs
            if record['query_id'].startswith('qasper:'):
                annotations = query.get('metadata', {}).get('annotations', [])
                if annotations:
                    fake = {'p': {'qas': [{'question_id': record['query_id'],
                        'answers': [{'answer': a['raw_answer']} for a in annotations]}]}}
                    extracted = official.get_answers_and_evidence(fake, True)[record['query_id']]
                    official_refs = [r['answer'] for r in extracted if r['type'] != 'none']
            score = None
            if record['document_answerable']:
                score = 0.0 if record['failed'] or record['model_abstained'] else max(
                    (official.token_f1_score(record['answer'], ref) for ref in official_refs), default=0.0)
                if not math.isclose(score, record['legacy_token_f1'], abs_tol=1e-12):
                    differences.append({'record_id': record['record_id'], 'legacy': record['legacy_token_f1'],
                                        'official_compatible': score})
            tokens, token_source = observed_output_tokens(record)
            out = {'record_id': record['record_id'], 'query_id': record['query_id'], 'doc_id': record['doc_id'],
                'experiment_id': record['experiment_id'], 'model_id': record['model_id'], 'method': record['method'],
                'source_record_hash': record['source_record_hash'], 'source': record['source'],
                'legacy_token_f1': record['legacy_token_f1'], 'qasper_compatible_answer_f1': score,
                'precision': record['precision'], 'recall': record['recall'], 'exact_match': record['exact_match'],
                'predicted_words': record['predicted_words'], 'reference_words': record['reference_words'],
                'answer_type': record['answer_type'], 'document_answerable': record['document_answerable'],
                'model_abstained': record['model_abstained'], 'failed': record['failed'],
                'actual_output_tokens': tokens, 'actual_output_token_source': token_source,
                'at_output_limit': tokens >= 512 if tokens is not None else None,
                'repeated_4gram_fraction': repeating_ngram_fraction(record['raw_output'] or ''),
                'visible_context_sufficient': 'UNKNOWN', 'semantic_metrics_status': 'NOT_RUN_YET',
                'finish_reason': record['finish_reason'], 'human_review_status': 'NOT_REVIEWED'}
            scored.append(out)
            strata = (record['failed'], record['model_abstained'], record['answer_type'])
            buckets[strata].append(record)
        write_lines(run / 'RECOUNT_ALL_OLD.jsonl', scored)
        sampled = []
        for rows in buckets.values():
            rows.sort(key=lambda r: signature([42, 'blind120', r['record_id']]))
        while len(sampled) < 120 and any(buckets.values()):
            for key in sorted(buckets):
                if buckets[key] and len(sampled) < 120:
                    sampled.append(buckets[key].pop(0))
        counts = Counter((r['failed'], r['model_abstained'], r['answer_type']) for r in records)
        selected = Counter((r['failed'], r['model_abstained'], r['answer_type']) for r in sampled)
        blind, private = [], []
        # Models and source files are intentionally only in the private join file.
        for i, r in enumerate(sampled):
            key = (r['failed'], r['model_abstained'], r['answer_type'])
            bid = 'BLIND_' + signature([42, r['record_id']])[:16]
            blind.append({'blind_id': bid, 'question': r['question'], 'answer': r['answer'],
                'references': r['answer_refs'], 'citations': r['citations'], 'context_hash': r['context_hash'],
                'stratum': {'failed': key[0], 'abstained': key[1], 'answer_type': key[2]},
                'stratum_population_n': counts[key], 'stratum_sample_n': selected[key],
                'sampling_probability': selected[key] / counts[key], 'sample_weight': counts[key] / selected[key],
                'sampling_design': 'deterministic hash-stratified diagnostic sample; not random human review',
                'human_review_status': 'NOT_REVIEWED'})
            private.append({'blind_id': bid, 'record_id': r['record_id'], 'source': r['source'],
                            'model_id': r['model_id'], 'experiment_id': r['experiment_id'], 'method': r['method']})
        write_lines(run / 'BLIND_REVIEW_PACKAGE.jsonl', blind)
        write_lines(run / 'BLIND_REVIEW_PRIVATE_JOIN.jsonl', private)
        write_lines(run / 'JUDGE_CALIBRATION_CASES.jsonl', calibration_groups())
        supervision = supervision_audit(runtime)
        atomic_json(run / 'SUPERVISION_20_AUDIT.json', supervision)
        atomic_json(run / 'RECOUNT_ALL_OLD.json', {'status': 'PASS', 'n': len(scored),
            'main_matrix_n': len(scored)-200, 'paired_lora_n': 200,
            'official_compatible_differences_n': len(differences), 'differences': differences,
            'official_evaluator': artifact(official_path), 'new_llm_requests': 0,
            'bert_score_status': 'NOT_RUN_YET', 'claim_metrics_status': 'NOT_RUN_YET',
            's0_projection_token_counter_correction': 'Use actual_output_tokens for BF16; S0 fallback completion_tokens is a reference-tokenizer count. Original S0 evidence retained.',
            'human_review_status': 'NOT_REVIEWED'})
        paths = [run / name for name in ['RECOUNT_ALL_OLD.json', 'RECOUNT_ALL_OLD.jsonl',
            'BLIND_REVIEW_PACKAGE.jsonl', 'BLIND_REVIEW_PRIVATE_JOIN.jsonl', 'JUDGE_CALIBRATION_CASES.jsonl',
            'SUPERVISION_20_AUDIT.json']]
        atomic_json(receipt, {'status': 'PASS', 'script': source, 'outputs': [artifact(p) for p in paths],
            'actual_exit_code': 0, 'new_llm_requests': 0, 'semantic_scoring_complete': False})
        print(json.dumps({'status': 'OLD_METRICS_AUDITED_SEMANTICS_PENDING', 'old_answer_n': len(scored),
            'official_compatible_differences_n': len(differences), 'blind_sample_n': len(blind),
            'calibration_groups': 60, 'supervision_20_status': supervision['status'], 'eos_ids_differ': supervision['eos_ids_differ'],
            'new_llm_requests': 0}))
        return 0


if __name__ == '__main__':
    raise SystemExit(main())
