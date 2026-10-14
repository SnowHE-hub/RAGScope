"""Freeze 24 exposed diagnostic questions, identical contexts and G0 reuse proofs."""
from __future__ import annotations
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sqlite3
import sys

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT)); sys.path.insert(0, str(PROJECT/'scripts'))
from freeze_generation_v2 import model_client_factory
from filelock import FileLock
from ragscope.experiments.runtime import configure, require_ram
from ragscope.generation.context import validate_bundle
from ragscope.generation.prompts import visible_output_schema
from ragscope.indexing.identity import artifact, atomic_json, canonical_json, signature
from ragscope.v3.generation_protocols import card_sampling, protocol_messages
from ragscope.v3.splits import assert_inference_safe, read_split


def save_lines(path, rows):
    with path.open('x', encoding='utf-8', newline='\n') as handle:
        for row in rows:
            handle.write(canonical_json(row)+'\n')
        handle.flush(); os.fsync(handle.fileno())
    return artifact(path)


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
    with FileLock(str(run/'protocol_input_preparation.lock'), timeout=0):
        receipt = run/'S2_PROTOCOL_STUDY_INPUTS.json'
        if receipt.exists():
            saved = json.loads(receipt.read_text('utf-8'))
            if not args.resume or saved['runner'] != artifact(__file__) or any(artifact(o['path']) != o for o in saved['outputs']):
                raise ValueError('Frozen S2 study identity changed')
            print(json.dumps({'status': 'REUSED_PROTOCOL_STUDY_INPUTS'})); return 0
        exposed = {r['query_id'] for r in read_split(run, 'D_search', purpose='inference')}
        labels = {r['record_id']: r for r in (json.loads(l) for l in (run/'RECOUNT_ALL_OLD.jsonl').read_text('utf-8').splitlines())}
        old_manifest_path = runtime/'runs/20261014_v2_generation/EXPERIMENT_MANIFEST.json'
        old = json.loads(old_manifest_path.read_text('utf-8'))
        dev = {json.loads(l)['query_id'] for l in (runtime/'corpora/20261014_v2/queries_dev.jsonl').read_text('utf-8').splitlines()}
        clients, model_specs = {}, {}
        cards = json.loads((run/'GENERATION_MODEL_CARDS.json').read_text('utf-8'))
        for model_id in ('qwen_2b', 'qwen_4b', 'qwen_9b'):
            spec = old['provenance']['spec']['models'][model_id]
            client, pilot = model_client_factory(spec, dev)
            card = next(c for c in cards['models'] if c['model'].endswith(model_id[-2:].upper()))
            clients[model_id] = client
            model_specs[model_id] = {'spec': spec, 'identity': client.identity, 'pilot': pilot,
                'sampling': card_sampling(card['artifact']['path'], card['model'])}
        db = sqlite3.connect('file:'+str(run/'S0_EVIDENCE.sqlite')+'?mode=ro', uri=True)
        baselines = {}
        for (payload,) in db.execute("SELECT payload FROM records WHERE json_extract(payload,'$.experiment_id')='E2' AND json_extract(payload,'$.method')='R2' AND json_extract(payload,'$.model_id')='qwen_2b'"):
            row = json.loads(payload)
            if row['query_id'] in exposed:
                baselines[row['query_id']] = row
        inputs = {}
        excluded = []
        for qid, row in baselines.items():
            pair = {}
            for kind, iid in (('retrieval_frozen', row['input_id']), ('oracle_audited', 'E3:oracle:'+qid)):
                found = db.execute('SELECT payload FROM inputs WHERE input_id=?', (iid,)).fetchone()
                if found is None:
                    raise ValueError('Exposed diagnostic lacks matched frozen Oracle input')
                original = json.loads(found[0])
                bundle = original['context_bundle']
                validate_bundle(bundle, scope=row['scope'])
                safe = {'query_id': qid, 'question': row['question'], 'scope': row['scope'],
                        'context_bundle': bundle, 'context_kind': kind}
                assert_inference_safe(safe)
                safe['input_id'] = 'S2:'+kind+':'+qid
                safe['input_hash'] = signature(safe)
                pair[kind] = safe
            counts = [clients[mid].prompt_token_counter(protocol_messages(row['question'], entry['context_bundle'], protocol))
                for entry in pair.values() for mid in clients for protocol in ('G0', 'G1')]
            if max(counts)+512 > 4096:
                excluded.append({'query_id': qid, 'reason': 'pre-result fixed full-prompt budget infeasible', 'maximum_prompt_tokens': max(counts)})
            else:
                inputs[qid] = pair
        ordered = sorted(inputs, key=lambda qid: signature([42, 'S2_predeclared24', qid]))
        selected, strata = [], {}
        for category in ('unanswerable', 'yes_no', 'extractive', 'free_form'):
            available = []
            for qid in ordered:
                if qid in selected:
                    continue
                row = baselines[qid]; label = labels[row['record_id']]['answer_type']
                eligible = not row['document_answerable'] if category == 'unanswerable' else row['document_answerable'] and (label == category or category == 'free_form' and 'free_form' in label)
                if eligible:
                    available.append(qid)
            if len(available) < 6:
                raise ValueError('Required six-case diagnostic stratum not available: '+category)
            for qid in available[:6]:
                selected.append(qid); strata[qid] = category
        # G0 raw journal records are streamed and validated, not rebuilt from summary scores.
        needed = {}
        for (payload,) in db.execute("SELECT payload FROM records WHERE (json_extract(payload,'$.experiment_id')='E2' AND json_extract(payload,'$.method')='R2') OR (json_extract(payload,'$.experiment_id')='E3' AND json_extract(payload,'$.method')='oracle')"):
            row = json.loads(payload)
            if row['query_id'] in selected and row['model_id'] in clients:
                needed.setdefault(row['source']['path'], {})[row['source']['line']] = row
        reuses = []
        for path, wanted in needed.items():
            with Path(path).open(encoding='utf-8') as handle:
                for number, line in enumerate(handle, 1):
                    if number not in wanted:
                        continue
                    original, normalized = json.loads(line), wanted[number]
                    digest = original.pop('record_hash')
                    if signature(original) != digest or digest != normalized['source_record_hash']:
                        raise ValueError('G0 source record hash differs')
                    kind = 'retrieval_frozen' if normalized['experiment_id'] == 'E2' else 'oracle_audited'
                    entry = inputs[normalized['query_id']][kind]
                    identity = model_specs[normalized['model_id']]['identity']
                    if original['model_identity'] != identity or original['prompt'] != protocol_messages(entry['question'], entry['context_bundle'], 'G0'):
                        raise ValueError('G0 model/prompt identity differs')
                    if original['output_schema'] != visible_output_schema(entry['context_bundle']['pieces']):
                        raise ValueError('G0 visible source constraint differs')
                    reuses.append({'record_id': 'S2:G0:'+normalized['model_id']+':'+entry['input_id'],
                        'model_id': normalized['model_id'], 'input_id': entry['input_id'], 'source': normalized['source'],
                        'raw_response': original['raw_output'], 'parsed_response': original['parsed_output'],
                        'error': original['error'], 'backend_metadata': original['backend_metadata'],
                        'model_identity': identity, 'prompt_hash': original['prompt_hash'],
                        'raw_backend_channels': 'NOT_RECORDED_IN_V2; new smoke tests inspect channels',
                        'reused_actual_model_result': True, 'new_llm_requests': 0})
        db.close()
        if len(reuses) != 144 or len({r['record_id'] for r in reuses}) != 144:
            raise ValueError('G0 reusable matrix incomplete')
        inference_rows = [inputs[qid][kind] for qid in selected for kind in ('retrieval_frozen', 'oracle_audited')]
        evaluation_rows = [{'query_id': qid, 'answer_refs': baselines[qid]['answer_refs'],
            'document_answerable': baselines[qid]['document_answerable'], 'official_answer_type': labels[baselines[qid]['record_id']]['answer_type'],
            'selection_stratum': strata[qid]} for qid in selected]
        evaluator = run/'splits/evaluator_only'; evaluator.mkdir(exist_ok=True)
        outputs = [save_lines(run/'S2_PROTOCOL_INPUTS.jsonl', inference_rows),
                   save_lines(run/'S2_G0_REUSED_RESULTS.jsonl', reuses),
                   save_lines(evaluator/'S2_DIAGNOSTIC_LABELS.jsonl', evaluation_rows)]
        result = {'status': 'PASS_FROZEN_PROTOCOL_STUDY', 'runner': artifact(__file__),
            'protocol_implementation': artifact(PROJECT/'ragscope/v3/generation_protocols.py'),
            'old_manifest': artifact(old_manifest_path), 'source_database': artifact(run/'S0_EVIDENCE.sqlite'),
            'model_cards': artifact(run/'GENERATION_MODEL_CARDS.json'), 'models': model_specs, 'outputs': outputs,
            'query_ids': selected, 'question_n': 24, 'strata': dict(Counter(strata.values())), 'context_n': 48,
            'G0_reused_actual_results': 144, 'new_main_requests_planned': 720,
            'planned_smoke_requests': 72, 'planned_oracle_padding_max': 36, 'planned_prior_diagnostic_requests': 12,
            'S2_request_ceiling': 1000, 'pre_result_budget_exclusions': excluded,
            'dataset_role': 'EXPOSED_DIAGNOSTIC', 'H_QA_gold_read': False, 'human_review_status': 'NOT_REVIEWED'}
        atomic_json(receipt, result)
        print(json.dumps({k: v for k, v in result.items() if k not in ('runner', 'protocol_implementation', 'old_manifest', 'source_database', 'models', 'model_cards', 'outputs', 'query_ids', 'pre_result_budget_exclusions')}))
        return 0


if __name__ == '__main__':
    raise SystemExit(main())
