"""Calibrate a pinned independent local model, then judge question/context only."""
import argparse
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT)); sys.path.insert(0, str(PROJECT/'scripts'))
from freeze_generation_v2 import model_client_factory
from filelock import FileLock
from ragscope.experiments.runtime import configure, require_ram
from ragscope.experiments.ledger import RecordStore, read_records
from ragscope.indexing.identity import artifact, atomic_json
from ragscope.v3.budget import Budget
from ragscope.v3.judge_protocol import (JUDGE_SYSTEM, calibration_cases, context_input,
                                      relation_schema, sufficiency_schema, validate_sufficiency)
from ragscope.v3.local_runtime import ManagedOllama, counted_chat


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
    with FileLock(str(run/'writer.lock'), timeout=0):
        old = json.loads((runtime/'runs/20261014_v2_generation/EXPERIMENT_MANIFEST.json').read_text('utf-8'))
        dev = {json.loads(l)['query_id'] for l in (runtime/'corpora/20261014_v2/queries_dev.jsonl').read_text('utf-8').splitlines()}
        original, pilot = model_client_factory(old['provenance']['spec']['models']['qwen_9b'], dev)
        budget = Budget(run)
        cases = calibration_cases(read_records(run/'JUDGE_CALIBRATION_CASES_V2.jsonl'))
        config = {'model_base_identity': original.identity, 'pilot': pilot, 'num_ctx': 8192,
            'system_prompt': JUDGE_SYSTEM, 'max_tokens': 512, 'temperature': 0,
            'cases': cases, 'source_cases': artifact(run/'JUDGE_CALIBRATION_CASES_V2.jsonl'),
            'protocol': artifact(PROJECT/'ragscope/v3/judge_protocol.py'), 'runner': artifact(__file__),
            'runtime': artifact(PROJECT/'ragscope/v3/local_runtime.py'), 'human_review_status': 'NOT_REVIEWED'}
        calibration = RecordStore(run/'LOCAL_JUDGE_CALIBRATION.jsonl', config, max_records=60)
        rows = read_records(run/'CONTEXT_ONLY_JUDGEMENT_INPUTS.jsonl')
        target = RecordStore(run/'CONTEXT_SUFFICIENCY_PROXY.jsonl',
            {**config, 'inputs': artifact(run/'CONTEXT_ONLY_JUDGEMENT_INPUTS.jsonl'), 'purpose': 'context_only_sufficiency'}, max_records=len(rows))
        pending_calibration = [c for c in cases if c['case_id'] not in calibration.ids]
        pending_rows = [r for r in rows if r['record_id'] not in target.ids]
        if pending_calibration or pending_rows:
            with ManagedOllama(run, runtime/'tools/ollama-0.34.4/ollama.exe') as service:
                client = service.client(original, num_ctx=8192)
                for case in pending_calibration:
                    require_ram(2*1024**3)
                    schema = sufficiency_schema() if case['mode'] == 'context' else relation_schema()
                    messages = [{'role': 'system', 'content': JUDGE_SYSTEM},
                                {'role': 'user', 'content': json.dumps(case['input'], ensure_ascii=False)}]
                    response, error, prediction, backend_error = {}, None, None, None
                    rid = 'judge_cal_v1:'+case['case_id']
                    try:
                        response = counted_chat(run, service, client, request_id=rid,
                            stage='S0_S1_evidence_metrics', messages=messages, max_tokens=512,
                            purpose='independent_judge_engineering_calibration', output_schema=schema)
                        parsed = json.loads(response['content'])
                        prediction = validate_sufficiency(parsed, case['input']['passage']) if case['mode'] == 'context' else parsed['relations']
                    except Exception as exc:
                        with budget.lock:
                            if rid not in budget._state()['reserved']:
                                raise
                        error = {'kind': type(exc).__name__, 'message': str(exc)}
                        backend_error = getattr(exc, 'kind', None)
                        if not response and client.raw_response is not None:
                            response = {'raw_backend_response': client.raw_response, 'backend_metadata': client.last_metadata,
                                        'model_identity': client.identity, 'request_id': rid}
                    calibration.add(case['case_id'], {'case': case, 'prediction': prediction, 'correct': prediction == case['expected'],
                        'model_response': response, 'error': error, 'human_review_status': 'NOT_REVIEWED'})
                    if backend_error in {'oom', 'timeout', 'transport_error', 'service_not_ready', 'device_policy_error'}:
                        raise RuntimeError('Backend unavailable; actual calibration failure retained, batch stopped')
                gate = len(calibration.ids) == 60 and sum(r['correct'] for r in calibration.records)/60 >= .90
                atomic_json(run/'LOCAL_JUDGE_CALIBRATION.json', {'status': 'PASS_ENGINEERING_CALIBRATION' if gate else 'FAIL_ENGINEERING_CALIBRATION',
                    'n': len(calibration.ids), 'correct': sum(r['correct'] for r in calibration.records), 'gate': gate,
                    'records': artifact(calibration.path), 'human_review_status': 'NOT_REVIEWED',
                    'natural_language_accuracy': 'NOT_ESTABLISHED'})
                if not gate:
                    print(json.dumps({'status': 'CONTEXT_JUDGE_NOT_RUN_CALIBRATION_GATE'})); return 2
                for row in pending_rows:
                    require_ram(2*1024**3)
                    body = context_input(row)
                    messages = [{'role': 'system', 'content': JUDGE_SYSTEM},
                                {'role': 'user', 'content': json.dumps(body, ensure_ascii=False)}]
                    response, error, verdict, backend_error = {}, None, 'UNKNOWN', None
                    rid = 'context_suff_v1:'+row['record_id']
                    try:
                        response = counted_chat(run, service, client, request_id=rid,
                            stage='S0_S1_evidence_metrics', messages=messages, max_tokens=512,
                            purpose='question_context_only_sufficiency', output_schema=sufficiency_schema())
                        verdict = validate_sufficiency(json.loads(response['content']), body['passage'])
                    except Exception as exc:
                        with budget.lock:
                            if rid not in budget._state()['reserved']:
                                raise
                        error = {'kind': type(exc).__name__, 'message': str(exc)}
                        backend_error = getattr(exc, 'kind', None)
                        if not response and client.raw_response is not None:
                            response = {'raw_backend_response': client.raw_response, 'backend_metadata': client.last_metadata,
                                        'model_identity': client.identity, 'request_id': rid}
                    target.add(row['record_id'], {'input_hash': row['input_hash'], 'visible_context_sufficient': verdict,
                        'model_response': response, 'error': error, 'answers_or_references_supplied': False,
                        'human_review_status': 'NOT_REVIEWED'})
                    print(json.dumps({'event': 'CONTEXT_JUDGE_PROGRESS', 'completed': len(target.ids), 'expected': len(rows)}), flush=True)
                    if backend_error in {'oom', 'timeout', 'transport_error', 'service_not_ready', 'device_policy_error'}:
                        raise RuntimeError('Backend unavailable; actual context failure retained, batch stopped')
        counts = {value: sum(r['visible_context_sufficient'] == value for r in target.records) for value in ('YES', 'NO', 'UNKNOWN')}
        output = {'status': 'PASS_CONTEXT_PROXY_BATCH' if len(target.ids) == len(rows) else 'INCOMPLETE',
            'n': len(target.ids), 'expected': len(rows), 'verdicts': counts, 'records': artifact(target.path),
            'human_review_status': 'NOT_REVIEWED', 'interpretation': 'AI proxy, independent of generator output; not human truth'}
        atomic_json(run/'CONTEXT_SUFFICIENCY_SUMMARY.json', output)
        print(json.dumps({k: v for k, v in output.items() if k != 'records'})); return 0


if __name__ == '__main__':
    raise SystemExit(main())
