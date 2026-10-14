"""Generate bounded source-backed H_tasks references, retaining teacher failures."""
import argparse
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT)); sys.path.insert(0, str(PROJECT/'scripts'))
from freeze_generation_v2 import model_client_factory
from filelock import FileLock
from ragscope.experiments.ledger import RecordStore, read_records
from ragscope.experiments.runtime import configure, require_ram
from ragscope.indexing.identity import artifact, atomic_json, canonical_json
from ragscope.v3.budget import Budget
from ragscope.v3.local_runtime import ManagedOllama, counted_chat
from ragscope.v3.splits import assert_inference_safe
from ragscope.v3.task_references import TEACHER_SYSTEM, schema, validate_reference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    runtime = Path(configure(cpu_only=True)['home']).resolve(); run = Path(args.campaign).resolve()
    if run.parent != runtime/'runs' or not run.name.startswith('v3_'):
        raise ValueError('Wrong campaign')
    require_ram(2*1024**3)
    with FileLock(str(run/'writer.lock'), timeout=0):
        papers = read_records(run/'H_tasks.source_documents.jsonl')
        old = json.loads((runtime/'runs/20261014_v2_generation/EXPERIMENT_MANIFEST.json').read_text('utf-8'))
        dev = {json.loads(l)['query_id'] for l in (runtime/'corpora/20261014_v2/queries_dev.jsonl').read_text('utf-8').splitlines()}
        original, pilot = model_client_factory(old['provenance']['spec']['models']['qwen_4b'], dev)
        config = {'source_papers': artifact(run/'H_tasks.source_documents.jsonl'), 'runner': artifact(__file__),
            'protocol': artifact(PROJECT/'ragscope/v3/task_references.py'), 'system': TEACHER_SYSTEM,
            'model_base_identity': original.identity, 'pilot': pilot, 'num_ctx': 32768, 'max_output_tokens': 4096,
            'temperature': 0, 'source_scope': 'WHOLE_PAPER_NEVER_TRUNCATED', 'human_review_status': 'NOT_REVIEWED'}
        evaluator = run/'splits/evaluator_only'
        store = RecordStore(evaluator/'H_tasks_teacher.jsonl', config, max_records=30)
        pending = [p for p in papers if p['source_id'] not in store.ids]
        budget = Budget(run)
        for offset in range(0, len(pending), 5):
            with ManagedOllama(run, runtime/'tools/ollama-0.34.4/ollama.exe', num_ctx=32768) as service:
                client = service.client(original, num_ctx=32768)
                for paper in pending[offset:offset+5]:
                    require_ram(2*1024**3)
                    messages = [{'role': 'system', 'content': TEACHER_SYSTEM}, {'role': 'user',
                        'content': json.dumps({'paper_id': paper['source_id'], 'title': paper['title'], 'whole_paper': paper['text']}, ensure_ascii=False)}]
                    if original.prompt_token_counter(messages)+4096 > 32768:
                        store.add(paper['source_id'], {'status': 'BLOCKED_FULL_SOURCE_TOO_LONG', 'source_id': paper['source_id'],
                            'raw_response': None, 'tasks': [], 'new_llm_requests': 0, 'human_review_status': 'NOT_REVIEWED'})
                        continue
                    rid = 'H_tasks_teacher_v1:'+paper['source_id']
                    response, tasks, error, backend_error = {}, [], None, None
                    try:
                        response = counted_chat(run, service, client, request_id=rid, stage='S0_S1_evidence_metrics',
                            messages=messages, max_tokens=4096, output_schema=schema(), purpose='synthetic_whole_paper_task_reference')
                        parsed = json.loads(response['content'])
                        for field, kind in (('summary', 'summarize'), ('comparison', 'compare')):
                            if parsed[field] is None:
                                continue
                            public, gold = validate_reference(paper, parsed[field], kind)
                            assert_inference_safe(public)
                            tasks.append({'public': public, 'gold': gold})
                    except Exception as exc:
                        with budget.lock:
                            if rid not in budget._state()['reserved']:
                                raise
                        error = {'kind': type(exc).__name__, 'message': str(exc)}
                        backend_error = getattr(exc, 'kind', None)
                    store.add(paper['source_id'], {'status': 'PASS_SOURCE_QUOTES' if error is None else 'FAIL_SOURCE_REFERENCE',
                        'source_id': paper['source_id'], 'raw_response': response, 'tasks': tasks if error is None else [],
                        'error': error, 'human_review_status': 'NOT_REVIEWED'})
                    print(json.dumps({'event': 'H_TASK_REFERENCE_PROGRESS', 'papers': len(store.ids), 'expected': 30}), flush=True)
                    if backend_error in {'oom', 'timeout', 'transport_error'}:
                        raise RuntimeError('Teacher backend failure retained; stop batch')
        tasks = [t for row in store.records for t in row['tasks']]
        for path, rows in ((run/'splits/H_tasks.questions.jsonl', [t['public'] for t in tasks]),
                           (evaluator/'H_tasks.gold.jsonl', [t['gold'] for t in tasks])):
            if not path.exists():
                with path.open('x', encoding='utf-8', newline='\n') as handle:
                    for row in rows: handle.write(canonical_json(row)+'\n')
            elif read_records(path) != rows:
                raise ValueError('Frozen task population differs')
        output = {'status': 'TASK_REFERENCE_CONSTRUCTION_TERMINAL', 'attempted_papers': len(store.ids),
            'expected_papers': 30, 'valid_task_n': len(tasks), 'failed_or_blocked_papers': sum(r['status'] != 'PASS_SOURCE_QUOTES' for r in store.records),
            'question_file': artifact(run/'splits/H_tasks.questions.jsonl'), 'gold_file': artifact(evaluator/'H_tasks.gold.jsonl'),
            'teacher': artifact(store.path), 'not_official_qasper': True, 'human_review_status': 'NOT_REVIEWED',
            'operator_blindness': 'NOT_OPERATOR_BLIND', 'all_teacher_failures_kept': True}
        atomic_json(run/'H_TASKS_CONSTRUCTION.json', output)
        print(json.dumps(output)); return 0


if __name__ == '__main__':
    raise SystemExit(main())
