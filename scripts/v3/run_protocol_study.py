"""Run the bounded S2 matrix on pinned models; reuse verified G0 outcomes."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT)); sys.path.insert(0, str(PROJECT/'scripts'))
from freeze_generation_v2 import model_client_factory
from filelock import FileLock
from ragscope.experiments.ledger import RecordStore, read_records
from ragscope.experiments.runtime import configure, require_ram
from ragscope.generation.context import pack_context
from ragscope.generation.pipeline import RAGPipeline
from ragscope.indexing.identity import artifact, atomic_json, signature
from ragscope.v3.audit import overlap
from ragscope.v3.budget import Budget
from ragscope.v3.generation_protocols import ProtocolClient, protocol_messages, protocol_schema
from ragscope.v3.local_runtime import ManagedOllama, counted_chat


def fixtures():
    data = [
        ('What dataset did the experiment use?', 'The experiment used dataset Delta.', 'Delta', False),
        ('Did the experiment use dataset Delta?', 'The experiment used dataset Delta.', 'Yes', False),
        ('Did the experiment use dataset Sigma?', 'The experiment did not use dataset Sigma.', 'No', False),
        ('Did the method increase accuracy?', 'The method did not increase accuracy.', 'No', False),
        ('Which method scored higher?', 'Method Alpha scored 14. Method Beta scored 12. Higher is better.', 'Alpha', False),
        ('What is the exact sample count?', 'The exact sample count was 128.', '128', False),
        ('What is the exact batch size?', 'The exact batch size was 16.', '16', False),
        ('What is the learning rate?', 'The learning rate was 0.001.', '0.001', False),
        ('What year was the dataset published?', 'The dataset was published in 2020.', '2020', False),
        ('What language did the experiment use?', 'The experiment used only English.', 'English', False),
        ('What is the capital of France?', '', 'Paris', True),
        ('How many degrees are in a right angle?', '', '90', True),
    ]
    result = []
    for index, (question, passage, expected, abstain) in enumerate(data):
        doc = 'v3_fixture:'+str(index)
        chunks = [{'chunk_id': doc+':c0', 'doc_id': doc, 'text': passage,
                   'canonical_start': 0, 'canonical_end': len(passage)}] if passage else []
        scope = {'kind': 'document', 'doc_ids': [doc]}
        result.append({'input_id': 'S2:fixture:'+str(index), 'query_id': 'fixture:'+str(index),
            'question': question, 'scope': scope, 'context_bundle': asdict(pack_context(chunks, scope=scope)),
            'expected_answer': expected, 'expected_abstain': abstain,
            'truth_source': 'CONSTRUCTED_UNAMBIGUOUS_RULE_NOT_QASPER'})
    return result


class PipelineCountedClient:
    def __init__(self, client, service, run, rid, temperature):
        self.client, self.service, self.run, self.rid, self.temperature = client, service, run, rid, temperature
        self.response = None
        self.last_metadata = {}

    def __getattr__(self, key):
        return getattr(self.client, key)

    def chat(self, messages, **kwargs):
        self.response = counted_chat(self.run, self.service, self.client, request_id=self.rid,
            stage='S2_generation_protocol', messages=messages, max_tokens=kwargs['max_tokens'],
            temperature=self.temperature, output_schema=kwargs.get('output_schema'), purpose='S2_pipeline_smoke')
        self.last_metadata = self.client.last_metadata
        return self.response['content']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--smoke-only', action='store_true')
    args = parser.parse_args()
    runtime = Path(configure(cpu_only=True)['home']).resolve()
    run = Path(args.campaign).resolve()
    if run.parent != runtime/'runs' or not run.name.startswith('v3_'):
        raise ValueError('Wrong campaign')
    require_ram(2*1024**3)
    with FileLock(str(run/'writer.lock'), timeout=0):
        study_path = run/'S2_PROTOCOL_STUDY_INPUTS.json'
        study = json.loads(study_path.read_text('utf-8'))
        if any(artifact(o['path']) != o for o in study['outputs']):
            raise ValueError('Frozen S2 inputs changed')
        if study['protocol_implementation'] != artifact(PROJECT/'ragscope/v3/generation_protocols.py'):
            raise ValueError('Protocol implementation changed after freeze')
        rows = read_records(run/'S2_PROTOCOL_INPUTS.jsonl')
        labels = {r['query_id']: r for r in read_records(run/'splits/evaluator_only/S2_DIAGNOSTIC_LABELS.jsonl')}
        fixture_rows = fixtures()
        config = {'study': artifact(study_path), 'protocol': study['protocol_implementation'],
            'runner': artifact(__file__), 'runtime': artifact(PROJECT/'ragscope/v3/local_runtime.py'),
            'output_cap': 512, 'num_ctx': 4096, 'think': False, 'resource_session_max_requests': 80,
            'answer_cache': False, 'fixture_definition': fixture_rows, 'G3_seeds': [42, 43, 44],
            'human_review_status': 'NOT_REVIEWED'}
        store = RecordStore(run/'S2_PROTOCOL_RESULTS.jsonl', config, max_records=1000)
        for row in read_records(run/'S2_G0_REUSED_RESULTS.jsonl'):
            store.add(row['record_id'], {**row, 'protocol': 'G0', 'dataset_role': 'EXPOSED_DIAGNOSTIC',
                'context_kind': row['input_id'].split(':')[1], 'source_inference_unchanged': True})
        dev = {json.loads(l)['query_id'] for l in (runtime/'corpora/20261014_v2/queries_dev.jsonl').read_text('utf-8').splitlines()}
        budget = Budget(run)
        for model_id, model in study['models'].items():
            original, pilot = model_client_factory(model['spec'], dev)
            if original.identity != model['identity']:
                raise ValueError('Pinned generator differs from frozen S2 study')
            tasks = []
            for fixture in fixture_rows:
                for mode in ('direct', 'pipeline'):
                    tasks.append({'record_id': 'S2:smoke:'+model_id+':'+mode+':'+fixture['input_id'],
                        'kind': 'smoke', 'mode': mode, 'protocol': 'G0', 'seed': 42, 'input': fixture})
            if not args.smoke_only:
                for entry in rows:
                    for protocol in ('G1', 'G2', 'G3'):
                        for seed in (42, 43, 44) if protocol == 'G3' else (42,):
                            tasks.append({'record_id': 'S2:'+protocol+':'+model_id+':'+str(seed)+':'+entry['input_id'],
                                'kind': 'matrix', 'mode': 'direct', 'protocol': protocol, 'seed': seed, 'input': entry})
                for fixture in fixture_rows[-2:]:
                    tasks.append({'record_id': 'S2:prior:'+model_id+':'+fixture['input_id'], 'kind': 'prior',
                        'mode': 'direct', 'protocol': 'G0', 'seed': 42, 'input': fixture})
            pending = [t for t in tasks if t['record_id'] not in store.ids]
            for offset in range(0, len(pending), 80):
                # Fixed resource sessions, not retries or extra experiment combinations.
                with ManagedOllama(run, runtime/'tools/ollama-0.34.4/ollama.exe', num_ctx=4096) as service:
                    for task in pending[offset:offset+80]:
                        require_ram(2*1024**3)
                        entry, protocol = task['input'], task['protocol']
                        sampling = model['sampling'] if protocol == 'G3' else None
                        temperature = sampling['temperature'] if sampling else 0.0
                        client = ProtocolClient(original, service, protocol=protocol, seed=task['seed'], sampling=sampling)
                        messages = protocol_messages(entry['question'], entry['context_bundle'], protocol)
                        if task['kind'] == 'prior':
                            messages[0]['content'] = ('Answer this general factual question using your prior knowledge. '
                                'No evidence or citations are available or required; citations must be []. '
                                'Return JSON with answer, citations and abstain; abstain only when uncertain.')
                        schema = protocol_schema(entry['context_bundle'], protocol)
                        response, parsed, error, backend_error = {}, None, None, None
                        started = time.perf_counter()
                        try:
                            if task['mode'] == 'pipeline':
                                wrapper = PipelineCountedClient(client, service, run, task['record_id'], temperature)
                                pipeline = RAGPipeline(None, wrapper, use_cache=False)
                                result = pipeline.answer(entry['question'], context_bundle=entry['context_bundle'], scope=entry['scope'], formal_evaluation=True)
                                response = wrapper.response or {}
                                error = result.error
                                parsed = result.parsed_output if not error else None
                                if result.prompt != messages:
                                    raise RuntimeError('Direct and pipeline messages differ: unresolved wrapper defect')
                            else:
                                response = counted_chat(run, service, client, request_id=task['record_id'],
                                    stage='S2_generation_protocol', messages=messages, max_tokens=512, temperature=temperature,
                                    output_schema=schema, purpose='S2_'+task['kind'])
                                parsed = RAGPipeline.parse_output(response['content'])
                            if client.last_metadata.get('token_count_match') is not True:
                                raise RuntimeError('Actual template token count differs: stop protocol study')
                            visible = {p['chunk_id'] for p in entry['context_bundle']['pieces']}
                            if parsed and any(cid not in visible for cid in parsed['citations']):
                                raise ValueError('Output citation not actually visible')
                        except Exception as exc:
                            with budget.lock:
                                if task['record_id'] not in budget._state()['reserved']:
                                    raise
                            error = {'kind': type(exc).__name__, 'message': str(exc)}
                            backend_error = getattr(exc, 'kind', None)
                            if not response:
                                response = {'content': (client.raw_response or {}).get('message', {}).get('content'),
                                    'raw_backend_response': client.raw_response, 'backend_metadata': client.last_metadata,
                                    'model_identity': client.identity, 'request_id': task['record_id']}
                        if task['kind'] == 'matrix':
                            label = labels[entry['query_id']]
                            answerable, refs = label['document_answerable'], label['answer_refs']
                        else:
                            answerable, refs = not entry['expected_abstain'] or task['kind'] == 'prior', [entry['expected_answer']]
                        failed = bool(error) or parsed is None
                        abstained = parsed['abstain'] if parsed else None
                        word_metrics = overlap(parsed['answer'], refs) if parsed and not failed and not abstained and answerable else None
                        word_f1 = None if not answerable else 0.0 if failed or abstained else word_metrics['legacy_token_f1']
                        if parsed:
                            parsed = {k: parsed[k] for k in ('answer', 'citations', 'abstain')}
                        store.add(task['record_id'], {'campaign_id': run.name, 'task_id': 'S2_'+task['kind'],
                            'question_id': entry['query_id'], 'paper_ids': entry['scope'].get('doc_ids', []),
                            'dataset_role': 'EXPOSED_DIAGNOSTIC' if task['kind'] == 'matrix' else 'CONSTRUCTED_RULE_DIAGNOSTIC',
                            'exposure_status': 'EXPOSED_DIAGNOSTIC', 'model_id': model_id, 'protocol': protocol,
                            'seed': task['seed'], 'context_kind': entry.get('context_kind', task['kind']), 'input_id': entry['input_id'],
                            'user_scope': entry['scope'], 'prompt_hash': signature(messages), 'model_identity': client.identity,
                            'actual_prompt_tokens': client.last_metadata.get('prompt_eval_count'),
                            'context_intervals': [{'doc_id': p['doc_id'], 'start': p.get('canonical_start'), 'end': p.get('canonical_end')} for p in entry['context_bundle']['pieces']],
                            'raw_response': response, 'parsed_response': parsed, 'stop_reason': client.last_metadata.get('done_reason'),
                            'output_tokens': client.last_metadata.get('eval_count'), 'success': not failed,
                            'error_type': error, 'wall_time_ms': (time.perf_counter()-started)*1000,
                            'measured_resource': client.last_metadata.get('resources'), 'budget_ticket': task['record_id'],
                            'actual_sent_payload': client.sent_payload, 'document_answerable': answerable, 'model_abstained': abstained,
                            'legacy_token_f1': word_f1, 'word_metrics': word_metrics, 'human_review_status': 'NOT_REVIEWED'})
                        print(json.dumps({'event': 'S2_PROTOCOL_PROGRESS', 'records': len(store.ids), 'model': model_id,
                                          'task': task['kind'], 'protocol': protocol}), flush=True)
                        if backend_error in {'oom', 'timeout', 'transport_error', 'device_policy_error'} or error and 'stop protocol study' in error.get('message', ''):
                            raise RuntimeError('Recorded backend/template failure; preserve batch and stop')
        output = {'status': 'SMOKE_TERMINAL' if args.smoke_only else 'PROTOCOL_MATRIX_TERMINAL',
            'record_n': len(store.ids), 'new_result_n': sum(not r.get('reused_actual_model_result', False) for r in store.records),
            'records': artifact(store.path), 'human_review_status': 'NOT_REVIEWED',
            'oracle_padded_status': 'PENDING_SEPARATE_BOUNDED36', 'NLI_full_old_status': 'BLOCKED_RESOURCE'}
        atomic_json(run/'S2_PROTOCOL_PROGRESS.json', output)
        print(json.dumps({k: v for k, v in output.items() if k != 'records'})); return 0


if __name__ == '__main__':
    raise SystemExit(main())
