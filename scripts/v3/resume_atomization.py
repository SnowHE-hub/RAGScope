"""Bounded resource repair: retain actual outcomes, resume only unissued calls."""
import argparse
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT)); sys.path.insert(0, str(PROJECT/'scripts'))
from filelock import FileLock
from freeze_generation_v2 import model_client_factory
from ragscope.experiments.runtime import configure, require_ram
from ragscope.experiments.ledger import RecordStore, read_records
from ragscope.indexing.identity import artifact, atomic_json, signature
from ragscope.v3.budget import Budget
from ragscope.v3.local_runtime import ManagedOllama, counted_chat
from ragscope.v3.quote_atoms import QUOTE_SYSTEM, quote_schema, quote_spans


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
        groups = [json.loads(l) for l in (run/'ATOMIZATION_INPUTS.jsonl').read_text('utf-8').splitlines()]
        parent = run/'ATOMIZATION_QUOTES_V3.jsonl'
        old_config = json.loads(parent.with_suffix('.config.json').read_text('utf-8'))
        budget = Budget(run)
        with budget.lock:
            state = budget._state()
        old = json.loads((runtime/'runs/20261014_v2_generation/EXPERIMENT_MANIFEST.json').read_text('utf-8'))
        dev = {json.loads(l)['query_id'] for l in (runtime/'corpora/20261014_v2/queries_dev.jsonl').read_text('utf-8').splitlines()}
        original, pilot = model_client_factory(old['provenance']['spec']['models']['qwen_4b'], dev)
        semantic = {'source_inputs': artifact(run/'ATOMIZATION_INPUTS.jsonl'), 'prompt': QUOTE_SYSTEM,
            'prompt_version': 'quote-only-v2', 'model_base_identity': original.identity, 'num_ctx': 8192,
            'max_tokens': 4096, 'temperature': 0.0, 'implementation': artifact(PROJECT/'ragscope/v3/quote_atoms.py'),
            'runtime_implementation': artifact(PROJECT/'ragscope/v3/local_runtime.py'), 'pilot': pilot,
            'truth_status': 'SEGMENTATION_NOT_JUDGEMENT', 'human_review_status': 'NOT_REVIEWED'}
        if any(old_config.get(k) != v for k, v in semantic.items()):
            raise ValueError('Segmentation semantic identity changed')
        config = {**semantic, 'runner': artifact(__file__), 'parent': artifact(parent),
            'repair': 'one resource scheduling repair; retain parent pre-call failures as provenance, no duplicate issued requests'}
        store = RecordStore(run/'ATOMIZATION_QUOTES_V4.jsonl', config, max_records=len(groups))
        unissued = []
        for number, row in enumerate(read_records(parent), 1):
            if row['config_hash'] != signature(old_config):
                raise ValueError('Parent record configuration differs')
            rid = 'atom_quote_v2:'+row['record_id']
            issued = rid in state['reserved']
            if not issued:
                if row['error'] is None or row['real_model'] or not row['error']['message'].startswith('RAM budget:'):
                    raise ValueError('Unissued row is not a proved resource preflight failure')
                unissued.append({'group_id': row['record_id'], 'parent_line': number, 'parent_hash': row['record_hash']})
                continue
            if rid not in state['finalized']:
                raise ValueError('Issued request not finalized; reconcile without calling again')
            body = {k: v for k, v in row.items() if k not in ('record_id', 'config_hash', 'record_hash', 'recorded_at')}
            body['inherited_source'] = {'path': str(parent), 'line': number, 'record_hash': row['record_hash']}
            store.add(row['record_id'], body)
        atomic_json(run/'ATOMIZATION_RESOURCE_REPAIR.json', {'initial_actual_requests': len(state['reserved']),
            'unissued_preflight_failures': unissued, 'actual_parent_results_reused': len(groups)-len(unissued),
            'old_failures_preserved': True, 'simultaneous_cpu_judge_forbidden': True, 'config': config})
        pending = [g for g in groups if g['group_id'] not in store.ids]
        failures = 0
        if pending:
            with ManagedOllama(run, runtime/'tools/ollama-0.34.4/ollama.exe') as service:
                client = service.client(original, num_ctx=8192)
                for group in pending:
                    # Preflight is outside per-question handling: do not manufacture an entire batch of failed calls.
                    require_ram(2*1024**3)
                    service.verify()
                    messages = [{'role': 'system', 'content': QUOTE_SYSTEM}, {'role': 'user',
                        'content': json.dumps({k: group[k] for k in ('query_id', 'question', 'texts')}, ensure_ascii=False)}]
                    rid = 'atom_quote_v2:'+group['group_id']
                    result, error, validated, backend_error = {}, None, [], None
                    try:
                        result = counted_chat(run, service, client, request_id=rid, stage='S0_S1_evidence_metrics',
                            messages=messages, max_tokens=4096, purpose='old_answer_quote_segmentation',
                            output_schema=quote_schema([t['source_id'] for t in group['texts']]))
                        returned = json.loads(result['content'])['sources']
                        expected = {t['source_id']: t['text'] for t in group['texts']}
                        if len(returned) != len(expected) or {r['source_id'] for r in returned} != set(expected):
                            raise ValueError('Missing or duplicate quote sources')
                        for source in returned:
                            validated.append({'source_id': source['source_id'],
                                'units': quote_spans(expected[source['source_id']], source['units'])})
                    except Exception as exc:
                        with budget.lock:
                            issued = rid in budget._state()['reserved']
                        if not issued:
                            raise  # Leave the group pending, not falsely model-failed.
                        error = {'kind': type(exc).__name__, 'message': str(exc)}
                        backend_error = getattr(exc, 'kind', None)
                        failures += 1
                        if not result and client.raw_response is not None:
                            result = {'raw_backend_response': client.raw_response,
                                'backend_metadata': client.last_metadata, 'model_identity': client.identity,
                                'request_id': rid, 'service_receipt': str(service.path)}
                    store.add(group['group_id'], {'query_id': group['query_id'], 'input_hash': group['input_hash'],
                        'model_response': result, 'segments': validated if error is None else None, 'error': error,
                        'human_review_status': 'NOT_REVIEWED', 'real_model': bool(result.get('model_identity')),
                        'original_texts_unchanged': True, 'resource_retry_of_unissued_parent': True})
                    atomic_json(run/'ATOMIZATION_PROGRESS.json', {'expected_groups': len(groups),
                        'recorded_groups': len(store.ids), 'valid_groups': sum(r['error'] is None for r in store.records),
                        'failures': sum(r['error'] is not None for r in store.records), 'last_group_id': group['group_id'],
                        'service_receipt': str(service.path), 'journal_version': 'V4',
                        'new_llm_calls_budget_authority': 'BUDGET.jsonl'})
                    print(json.dumps({'event': 'ATOMIZATION_PROGRESS', 'completed': len(store.ids),
                                      'expected': len(groups), 'errors_this_invocation': failures}), flush=True)
                    if backend_error in {'oom', 'transport_error', 'timeout'}:
                        raise RuntimeError('Backend failure recorded; stop rather than repeat unavailable calls')
        print(json.dumps({'status': 'SEGMENTATION_BATCH_TERMINAL', 'recorded': len(store.ids), 'failures': failures}))
        return 0 if failures == 0 else 2


if __name__ == '__main__':
    raise SystemExit(main())
