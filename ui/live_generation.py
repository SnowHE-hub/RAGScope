"""Count and serialize application requests with the existing local campaign."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import sys
from uuid import uuid4


def answer_live(client, retriever, packing, question, k, scope, execution, *, model_entry):
    from filelock import FileLock
    from ragscope.experiments.generation_campaign import CountedClient, GenerationBudget
    from ragscope.experiments.runtime import runtime_home
    from ragscope.generation.pipeline import RAGPipeline
    from ragscope.generation.resources import MeasuredClient
    if getattr(client, 'real_model', False) is not True:
        raise ValueError('应用问答不接受 Mock 客户端')
    # Reuse the already-audited process/lease verifier without changing frozen
    # experiment implementation files or duplicating their safety conditions.
    scripts = str(Path(__file__).resolve().parents[1]/'scripts')
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    from run_generation_campaign import GuardedClient, verify_owned_service
    from ui.model_identity import PinnedUIClient, verify_model_package
    run = Path(execution['campaign_run'])
    service, lease = run/'generation_service.json', run/'gpu_lease.json'
    budget_path = run/'generation_budget.json'
    if not budget_path.is_file():
        raise ValueError('应用生成必须使用已存在的项目调用预算')
    # No waiting behind an experiment or simultaneous model loads from UI tabs.
    with FileLock(str(runtime_home()/'runs/generation-client.lock'), timeout=0):
        verify_owned_service(service, lease, client.identity)
        verify_model_package(model_entry, service)
        if getattr(client, 'num_gpu', 0) != 0 and getattr(client, 'service_device_verified', None) == 'gpu':
            client.remote_gpu_authorizer = lambda identity: verify_owned_service(service, lease, identity)
        request_id = 'ui:'+uuid4().hex
        pinned = PinnedUIClient(client, model_entry, service)
        measured = MeasuredClient(GuardedClient(pinned, service, lease), service, gpu_device=0)
        counted = CountedClient(measured, GenerationBudget(budget_path), 'UI', request_id)
        pipeline = RAGPipeline(retriever, counted, use_cache=False, context_config=packing)
        record = asdict(pipeline.answer(question, k, scope=scope))
        record.update(application_request_id=request_id, call_ticket=counted.last_ticket,
                      experiment_type='interactive_application_not_benchmark')
        # Commit raw output/ticket before releasing the generation lock. UI
        # rendering and supplementary timings can fail without losing it.
        record['base_record_path'] = save_live_record(record, execution)
        return record


def save_live_record(record, execution, *, supplement=False):
    from filelock import FileLock
    from ragscope.experiments.ledger import RecordStore
    from ragscope.indexing.identity import signature
    config = {'kind': 'interactive_application_not_benchmark',
              'pipeline_config_hash': record['config_hash'], 'model_identity': record['model_identity'],
              'record_stage': 'ui_supplement' if supplement else 'raw_request'}
    path = Path(execution['campaign_run'])/'ui'/f'{signature(config)}.jsonl'
    with FileLock(str(path)+'.lock', timeout=0):
        store = RecordStore(path, config, real_only=True)
        store.add(record['application_request_id'], record)
    return str(path)
