"""Prepare, then explicitly run one owned CPU4 SQuAD query-only service window.

Default is metadata preparation only. --run requires an immutable coordinator
handoff receipt and fresh resource/process/port checks. No automatic retries.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import psutil
import requests
from filelock import FileLock
from ragscope.experiments.aux_preencode import frozen_job
from ragscope.experiments.ledger import now, read_records
from ragscope.experiments.runtime import configure
from ragscope.indexing.identity import artifact, atomic_json, file_hash, signature, verify_artifacts
from ragscope.indexing.local_embedding import HFJsonTokenizer, LocalEmbeddingService, QUERY_INSTRUCTION

HOME = Path('D:/RAGScope_Runtime')
CAMPAIGN = '20261014_v2'
BINARY_SHA = '7e236243c9b670c6a49410c03186fd575c57a0efe3fd8a21d8727503d571d5b6'
MODEL_SHA = '06507c7b42688469c4e7298b0a1e16deff06caf291cf0a5b278c308249c3e439'
AUX_HANDOFF_SHA = 'de498e5a3980bc42ad56c3369a786b8b625a209c2fc9c0c1309c1cbbc2239113'


class PlannedStop(SystemExit):
    """Leave unattempted questions pending; do not convert a guard into QA errors."""


class QueryOnlyEmbedding(LocalEmbeddingService):
    allowed_text_hashes = set()
    deadline = None
    backend_failed = False
    query_observations = []

    def embed(self, texts):
        if any(signature(text) not in self.allowed_text_hashes for text in texts):
            raise PlannedStop('Unexpected text: only the frozen 100 SQuAD questions may be encoded')
        if self.backend_failed:
            raise PlannedStop('Prior actual backend failure retained; no retry is permitted')
        if self.deadline and datetime.now(timezone.utc) >= self.deadline:
            raise PlannedStop('Execution deadline reached')
        if psutil.virtual_memory().available < 1024**3:
            raise PlannedStop('Available RAM below 1 GiB before the next query request')
        try:
            result = super().embed(texts)
        except requests.RequestException:
            type(self).backend_failed = True
            raise  # The actual attempted query remains an error row in the original runner.
        type(self).query_observations.extend(self.last_metrics)
        return result


def port_free(port):
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', port))


def process_absent(receipt):
    try:
        return abs(psutil.Process(receipt['pid']).create_time()-receipt['creation_time']) > .01
    except psutil.NoSuchProcess:
        return True


def prepare():
    from ragscope.experiments.retrieval_campaign import read_jsonl
    main = HOME / 'runs' / CAMPAIGN
    aux = HOME / 'runs' / (CAMPAIGN+'_aux_preencode')
    job = frozen_job(HOME, CAMPAIGN, 'squad')
    ready = json.loads((aux/'squad_R512_READY.json').read_text())
    if file_hash(aux/'AUX_INDEX_HANDOFF.json') != AUX_HANDOFF_SHA:
        raise ValueError('Sealed auxiliary handoff changed')
    handoff = json.loads((aux/'AUX_INDEX_HANDOFF.json').read_text())
    if not verify_artifacts(handoff['artifacts'] + [ready['dense_manifest'], ready['chunk_manifest']]):
        raise ValueError('Auxiliary index handoff or actual index artifact changed')
    manifest = json.loads(Path(ready['dense_manifest']['path']).read_text())
    chunk = json.loads(Path(ready['chunk_manifest']['path']).read_text())
    identity = json.loads((main/'embedding_identity.json').read_text())
    expected = {**chunk['identity'], 'engine': 'ArrayIndex-v2', 'bm25': job['config']['bm25'],
                'embedding': identity, 'embedding_input': job['config']['embedding_input']}
    if (manifest['identity'] != expected or manifest['signature'] != signature(expected)
            or manifest['count'] != 1212 or identity['weight_sha256'] != MODEL_SHA
            or not verify_artifacts(manifest['artifacts'])):
        raise ValueError('Frozen SQuAD dense index identity or content differs')
    corpus = HOME/'corpora'/CAMPAIGN/'squad_e4'
    queries = list(read_jsonl(corpus/'queries_test.jsonl'))
    if len(queries) != 100 or len({q['query_id'] for q in queries}) != 100:
        raise ValueError('The original full 100-question population is required')
    tokenizer_path = HOME/'models/qwen3-embedding-0.6b/tokenizer.json'
    if file_hash(tokenizer_path) != job['chunking']['tokenizer_file_sha256']:
        raise ValueError('Frozen tokenizer changed')
    tokenizer = HFJsonTokenizer(tokenizer_path, job['chunking']['tokenizer_id'], job['chunking']['tokenizer_revision'])
    service = LocalEmbeddingService('http://127.0.0.1:11436', identity, tokenizer,
                                    HOME/'cache/ragscope/v2_embeddings')
    query_info = []
    for query in queries:
        text = QUERY_INSTRUCTION+query['question']
        tokens = len(tokenizer.encode(text))
        if tokens > service.max_input_tokens:
            raise ValueError('Frozen query body exceeds the original embedding budget')
        vector = service.cache.get(text)
        if vector is not None and vector.shape != (1024,):
            raise ValueError('Cached SQuAD query has the wrong dimension')
        query_info.append({'query_id': query['query_id'], 'text_hash': signature(text),
            'scope': query['scope'], 'body_tokens': tokens, 'cached_before': vector is not None})
    binary = HOME/'tools/llama-b11429-cpu/llama-server.exe'
    model = HOME/'models/qwen3-embedding-0.6b/Qwen3-Embedding-0.6B-Q8_0.gguf'
    if file_hash(binary) != BINARY_SHA or file_hash(model) != MODEL_SHA:
        raise ValueError('Pinned CPU binary or model bytes changed')
    command = [str(binary), '-m', str(model), '--embedding', '--pooling', 'last', '--embd-normalize', '2',
        '--host', '127.0.0.1', '--port', '11436', '-ngl', '0', '--no-op-offload', '-t', '4', '-tb', '4',
        '-c', '2048', '-b', '1100', '-ub', '1100', '-np', '1', '--threads-http', '2']
    report = {'status': 'PREPARED_NO_MODEL_LOADED', 'time': now(), 'queries': query_info,
        'frozen_query_count': 100, 'index_chunks': 1212, 'cold_queries': sum(not q['cached_before'] for q in query_info),
        'ready_index': ready['dense_manifest'], 'index_signature': manifest['signature'],
        'embedding_identity': identity, 'auxiliary_handoff': artifact(aux/'AUX_INDEX_HANDOFF.json'),
        'source_artifacts': job['sources']+[artifact(corpus/'queries_test.jsonl'), artifact(tokenizer_path)],
        'binary_sha256': BINARY_SHA, 'model_sha256': MODEL_SHA, 'command': command,
        'start_guard_bytes': 3*1024**3, 'request_reserve_bytes': 1024**3,
        'health_timeout_seconds': 60, 'control_request_timeout_seconds': 30,
        'automatic_retry': False, 'historical_backend_recovery_budget': '3/3 unchanged',
        'model_loaded': False, 'new_backend_calls': 0, 'gpu_used': False}
    return report


def run(prepared, handoff_path, handoff_sha, directory):
    from ragscope.experiments import retrieval_campaign as module
    main, retrieval = HOME/'runs'/CAMPAIGN, HOME/'runs'/(CAMPAIGN+'_retrieval')
    if file_hash(handoff_path) != handoff_sha:
        raise ValueError('Coordinator handoff receipt SHA mismatch')
    port_free(11435)
    port_free(11436)
    previous = json.loads((main/'embedding_service.json').read_text())
    generation = json.loads((main/'generation_service.json').read_text())
    if not process_absent(previous) or not process_absent(generation):
        raise PlannedStop('Old owned service still exists; do not start another model')
    for lock_path in (HOME/'runs/gpu.lock', HOME/'runs/generation-client.lock', retrieval/'writer.lock'):
        with FileLock(str(lock_path), timeout=0):
            pass
    journal = retrieval/'embedding_runtime_failures.jsonl'
    if len(read_records(journal)) != 3:
        raise PlannedStop('Historical failure journal changed')
    deadline_path = main/'EXECUTION_DEADLINE.json'
    deadline = datetime.fromisoformat(json.loads(deadline_path.read_text())['deadline_utc'])
    if datetime.now(timezone.utc) >= deadline:
        raise PlannedStop('Execution deadline reached before loading')
    available = psutil.virtual_memory().available
    if available < prepared['start_guard_bytes']:
        atomic_json(directory/'START_RESOURCE_BLOCK.json', {'time': now(), 'available': available,
            'required': prepared['start_guard_bytes'], 'model_loaded': False})
        raise PlannedStop('Original 3 GiB load guard is not met')
    receipt_path = directory/'SERVICE_LIFETIME.json'
    if receipt_path.exists():
        raise PlannedStop('This planned service window was already attempted; no automatic retry')
    worker = psutil.Process()
    receipt = {'status': 'RESERVED_BEFORE_LOAD', 'time': now(), 'worker_pid': worker.pid,
        'worker_creation_time': worker.create_time(), 'worker_command': worker.cmdline(),
        'handoff': artifact(handoff_path), 'prepared': artifact(directory/'PREPARATION.json'),
        'implementation': [artifact(__file__), artifact(module.__file__), artifact(ROOT/'ragscope/indexing/local_embedding.py')],
        'failure_journal_before': artifact(journal), 'previous_service': previous,
        'available_before': available, 'command': prepared['command'], 'gpu_used': False, 'automatic_retry': False}
    with receipt_path.open('x', encoding='utf-8') as handle:
        json.dump(receipt, handle, ensure_ascii=False)
        handle.flush()
        os.fsync(handle.fileno())
    child, child_identity, created, service = None, None, None, None
    with (directory/'embedding_cpu4.log').open('w', encoding='utf-8') as log:
        try:
            child = subprocess.Popen(prepared['command'], stdin=subprocess.DEVNULL, stdout=log,
                stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW,
                env={**os.environ, 'CUDA_VISIBLE_DEVICES': ''})
            child_identity = psutil.Process(child.pid)
            created = child_identity.create_time()
            receipt.update(server_pid=child.pid, server_creation_time=created)
            atomic_json(receipt_path, receipt)
            end = time.monotonic()+60
            while time.monotonic() < end:
                if child.poll() is not None or psutil.virtual_memory().available < 1024**3:
                    raise PlannedStop('Service exited or RAM floor reached during load')
                try:
                    if requests.get('http://127.0.0.1:11436/health', timeout=1).status_code == 200:
                        break
                except requests.RequestException:
                    pass
                time.sleep(.25)
            else:
                raise PlannedStop('CPU health check exceeded 60 seconds')
            control = requests.post('http://127.0.0.1:11436/v1/embeddings',
                json={'input': 'Frozen SQuAD CPU four thread query window verification'}, timeout=30)
            control.raise_for_status()
            payload = control.json()
            if len(payload['data'][0]['embedding']) != 1024 or psutil.virtual_memory().available < 1024**3:
                raise PlannedStop('CPU control request dimensions/RAM verification failed')
            service = {'status': 'RUNNING_SQUAD_FINAL', 'pid': child.pid, 'creation_time': created,
                'command': prepared['command'], 'device': 'cpu', 'cuda_mask': '',
                'binary_sha256': BINARY_SHA, 'model_sha256': MODEL_SHA,
                'parent_pid': worker.pid, 'parent_alive': True, 'supervisor_pid': worker.pid,
                'time': now(), 'performance_identity': {'threads': 4, 'batch_threads': 4},
                'lifetime_receipt': str(receipt_path)}
            atomic_json(main/'embedding_service.json', service)
            receipt.update(status='RUNNING_QUERY_ONLY', control_dimensions=1024, control_usage=payload.get('usage'))
            atomic_json(receipt_path, receipt)
            print(json.dumps({'worker': worker.pid, 'service': child.pid, 'status': receipt['status']}), flush=True)
            QueryOnlyEmbedding.allowed_text_hashes = {q['text_hash'] for q in prepared['queries']}
            QueryOnlyEmbedding.deadline = deadline
            with patch.object(module, 'LocalEmbeddingService', QueryOnlyEmbedding):
                result = module.run_final_contexts(HOME, CAMPAIGN, 'squad', 'http://127.0.0.1:11436',
                    main/'embedding_identity.json', methods=['hybrid'], family='structured')
            path = retrieval/'records/E4_CONTEXT_R512_hybrid_neighbor.jsonl'
            rows = read_records(path, repair_tail=False)
            config_path = path.with_suffix('.config.json')
            expected = {q['query_id']: q for q in prepared['queries']}
            config_hash = signature(json.loads(config_path.read_text()))
            if (len(rows) != 100 or set(r['query_id'] for r in rows) != set(expected)
                    or any(not r.get('record_hash') or r['config_hash'] != config_hash
                           or r['scope'] != expected[r['query_id']]['scope']
                           or r.get('index_signature') != prepared['index_signature']
                           or r.get('real_model') is not True for r in rows)):
                raise ValueError('SQuAD records population/scope/identity mismatch')
            failures = sum(bool(r.get('error')) for r in rows)
            receipt.update(status='PASS' if result['status']=='PASS' and failures==0 else 'FAILED_RETAINED_DENOMINATOR',
                formal_rows=100, failed_rows=failures, records=artifact(path), config=artifact(config_path),
                query_embedding_observations=QueryOnlyEmbedding.query_observations,
                index_manifest_unchanged=verify_artifacts([prepared['ready_index']]))
        except BaseException as exc:
            receipt.update(status='PENDING' if isinstance(exc, PlannedStop) else 'FAILED',
                error={'type': type(exc).__name__, 'message': str(exc)})
            raise
        finally:
            if (child is not None and child.poll() is None and child_identity is not None and created is not None
                    and abs(child_identity.create_time()-created)<.01 and child_identity.cmdline()==prepared['command']):
                child_identity.terminate()
                child_identity.wait(timeout=15)
            receipt.update(ended_at=now(), service_exited=child is None or child.poll() is not None,
                service_exit_code=None if child is None else child.poll(),
                journal_unchanged=file_hash(journal)==receipt['failure_journal_before']['sha256'])
            if service is not None:
                service.update(status='EXITED_SQUAD_FINAL' if receipt['service_exited'] else 'CLEANUP_INCOMPLETE',
                    ended_at=now(), exit_code=receipt['service_exit_code'], parent_alive=False)
                atomic_json(main/'embedding_service.json', service)
            atomic_json(receipt_path, receipt)
            print(json.dumps({'status': receipt['status'], 'service_exited': receipt['service_exited']}), flush=True)
    return 0 if receipt['status']=='PASS' else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--handoff-receipt', type=Path)
    parser.add_argument('--handoff-sha256')
    args = parser.parse_args()
    configure(cpu_only=True)
    directory = HOME/'runs'/(CAMPAIGN+'_retrieval')/'SQUAD_FINAL'
    directory.mkdir(exist_ok=True)
    prepared = prepare()
    atomic_json(directory/'PREPARATION.json', prepared)
    print(json.dumps({k: prepared[k] for k in ('status','frozen_query_count','index_chunks','cold_queries','model_loaded')}), flush=True)
    if not args.run:
        return 0
    if not args.handoff_receipt or not args.handoff_sha256:
        raise ValueError('--run requires the actual coordinator handoff receipt and SHA')
    return run(prepared, args.handoff_receipt, args.handoff_sha256, directory)


if __name__ == '__main__':
    raise SystemExit(main())
