"""Materialize the frozen simple comparator using verified real vectors only.

No service is started. Any cache miss stops before a network request and leaves
the missing work pending. Ordinary experiment configs and record schemas stay
owned by run_final_contexts; this receipt records the execution condition.
"""
from __future__ import annotations

from contextlib import contextmanager
import argparse
import json
import os
from pathlib import Path
import socket
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import psutil
from filelock import FileLock
from ragscope.indexing.identity import artifact, atomic_json, file_hash, signature, verify_artifacts
from ragscope.indexing.local_embedding import HFJsonTokenizer, LocalEmbeddingService, QUERY_INSTRUCTION
from ragscope.experiments.ledger import now, read_records


class CacheOnlyPending(SystemExit):
    """Cooperative stop bypasses per-query Exception-to-failure conversion."""


def cached_item(service, text, historical_usage=None):
    vector = service.cache.get(text)
    if vector is None:
        raise CacheOnlyPending('PENDING_CACHE: no verified vector')
    usage_path = service.cache.path / (service.cache.key(text) + '.usage.json')
    body_tokens = len(service.tokenizer.encode(text))
    if not usage_path.is_file():
        historical = (historical_usage or {}).get(service.cache.key(text))
        if historical is None:
            raise CacheOnlyPending('PENDING_CACHE: no usage sidecar')
        if (historical['body_tokens'] != body_tokens
                or not verify_artifacts([historical['vector'], historical['vector_manifest']])
                or Path(historical['vector']['path']).resolve() != usage_path.with_suffix('.npy').with_name(service.cache.key(text)+'.npy').resolve()
                or body_tokens > service.max_input_tokens
                or vector.shape != (service.get_dimension(),)
                or not np.isclose(np.linalg.norm(vector), 1, atol=1e-5)):
            raise CacheOnlyPending('PENDING_CACHE: historical N/A provenance mismatch')
        return {'cache_key': service.cache.key(text), 'body_tokens': body_tokens,
            'backend_usage_observed': False, 'backend_usage_status': 'historical_NA_no_new_inference',
            'artifacts': [historical['vector'], historical['vector_manifest']]}
    usage = json.loads(usage_path.read_text(encoding='utf-8'))
    if (usage.get('key') != service.cache.key(text) or usage.get('body_tokens') != body_tokens
            or body_tokens > service.max_input_tokens
            or vector.shape != (service.get_dimension(),)
            or not np.isclose(np.linalg.norm(vector), 1, atol=1e-5)):
        raise CacheOnlyPending('PENDING_CACHE: vector/usage identity mismatch')
    backend_usage = usage.get('usage')
    if backend_usage is not None and (not isinstance(backend_usage, dict)
            or type(backend_usage.get('prompt_tokens')) is not int
            or not 0 < backend_usage['prompt_tokens'] <= service.max_input_tokens):
        raise CacheOnlyPending('PENDING_CACHE: invalid retained backend usage')
    stem = usage_path.with_name(service.cache.key(text))
    return {'cache_key': service.cache.key(text), 'body_tokens': body_tokens,
        'backend_usage_observed': backend_usage is not None,
        'artifacts': [artifact(str(stem)+suffix) for suffix in ('.npy', '.json', '.usage.json')]}


class CacheOnlyEmbedding(LocalEmbeddingService):
    observed_hits = 0
    historical_usage = {}
    def embed(self, texts):
        if psutil.virtual_memory().available < 1024**3:
            raise CacheOnlyPending('PENDING_RESOURCE: 1 GiB cache-only request reserve')
        for text in texts:
            cached_item(self, text, self.historical_usage)
        result = super().embed(texts)
        if len(self.last_metrics) != len(texts) or not all(m['cache_hit'] for m in self.last_metrics):
            raise CacheOnlyPending('PENDING_CACHE: unexpected cache race')
        type(self).observed_hits += len(texts)
        return result


@contextmanager
def no_network():
    attempts = []
    def denied(*args, **kwargs):
        attempts.append('network_request_blocked')
        raise CacheOnlyPending('PENDING_CACHE: network access is forbidden')
    with patch('requests.sessions.Session.request', denied), patch.object(socket.socket, 'connect', denied), \
            patch.object(socket.socket, 'connect_ex', denied):
        yield attempts


def port_closed():
    with socket.socket() as sock:
        sock.settimeout(1)
        return sock.connect_ex(('127.0.0.1', 11436)) != 0


def verified_historical_usage(path):
    if file_hash(path) != '858dc33c87d42600b4689d8983186e3d2bbe4b70c095cc110991a4dfe78f0950':
        raise CacheOnlyPending('PENDING_SOURCE: historical N/A receipt changed')
    body = json.loads(path.read_text())
    expected = {'qasper:000549a217ea24432c0656598279dbb85378c113',
                'qasper:5bfbc9ca7fd41be9627f6ef587bb7e21c7983be0',
                'qasper:3d662fb442d5fc332194770aac835f401c2148d9'}
    if (len(body['items']) != 3 or {item['query_id'] for item in body['items']} != expected
            or not verify_artifacts(body['source_records'])):
        raise CacheOnlyPending('PENDING_SOURCE: historical N/A population/source mismatch')
    allowed = {}
    for item in body['items']:
        records = item['existing_structured_records']
        if (len(records) != 2 or {r['run_id'] for r in records}
                != {'TEST_CONTEXT_R512_dense_neighbor', 'TEST_CONTEXT_R512_hybrid_neighbor'}):
            raise CacheOnlyPending('PENDING_SOURCE: missing original real query observations')
        metrics = [r['query_encoding_metrics'][0] for r in records]
        if any(not r['real_model'] or len(r['query_encoding_metrics']) != 1 for r in records) or any(
                m['cache_hit'] is not True or m['backend_usage'] is not None
                or m['body_tokens'] != metrics[0]['body_tokens'] for m in metrics):
            raise CacheOnlyPending('PENDING_SOURCE: original usage is not consistent N/A')
        allowed[item['cache_key']] = {**item, 'body_tokens': metrics[0]['body_tokens']}
    return allowed


def preflight(home, campaign, historical_usage=None):
    from ragscope.experiments.retrieval_campaign import read_jsonl, verify_corpus_files
    aux = home / 'runs' / (campaign + '_aux_preencode')
    handoff_path = aux / 'AUX_INDEX_HANDOFF.json'
    handoff = json.loads(handoff_path.read_text())
    if (file_hash(handoff_path) != 'de498e5a3980bc42ad56c3369a786b8b625a209c2fc9c0c1309c1cbbc2239113'
            or not verify_artifacts(handoff['artifacts'])):
        raise CacheOnlyPending('PENDING_SOURCE: auxiliary handoff changed')
    ready = json.loads((aux / 'qasper_F512_O0_READY.json').read_text())
    if ready['chunks'] != 777 or not verify_artifacts([ready['dense_manifest'], ready['chunk_manifest']]):
        raise CacheOnlyPending('PENDING_SOURCE: simple index changed')
    manifest = json.loads(Path(ready['dense_manifest']['path']).read_text())
    identity = json.loads((home / 'runs' / campaign / 'embedding_identity.json').read_text())
    if (manifest['signature'] != signature(manifest['identity']) or manifest['count'] != 777
            or manifest['identity']['embedding'] != identity or not verify_artifacts(manifest['artifacts'])):
        raise CacheOnlyPending('PENDING_SOURCE: simple manifest identity/artifact mismatch')
    corpus = home / 'corpora' / campaign
    verify_corpus_files(corpus, ('queries_test.jsonl',))
    queries = list(read_jsonl(corpus / 'queries_test.jsonl'))
    if len(queries) != 200 or len({q['query_id'] for q in queries}) != 200:
        raise CacheOnlyPending('PENDING_SOURCE: wrong frozen test population')
    config = json.loads((ROOT / 'configs/v2/chunking.json').read_text())
    tokenizer_path = home / 'models/qwen3-embedding-0.6b/tokenizer.json'
    if file_hash(tokenizer_path) != config['tokenizer_file_sha256']:
        raise CacheOnlyPending('PENDING_SOURCE: tokenizer changed')
    tokenizer = HFJsonTokenizer(tokenizer_path, config['tokenizer_id'], config['tokenizer_revision'])
    service = CacheOnlyEmbedding('http://127.0.0.1:11436', identity, tokenizer,
                                home / 'cache/ragscope/v2_embeddings')
    cache, missing = [], []
    for query in queries:
        try:
            item = cached_item(service, QUERY_INSTRUCTION + query['question'], historical_usage)
            cache.append({'query_id': query['query_id'], **item})
        except CacheOnlyPending as exc:
            missing.append({'query_id': query['query_id'], 'reason': str(exc)})
    return {'status': 'PASS' if not missing else 'PENDING_CACHE', 'query_count': 200,
        'verified_query_cache_count': len(cache), 'query_cache': cache, 'missing': missing,
        'sources': [artifact(handoff_path), ready['dense_manifest'], ready['chunk_manifest'],
                    artifact(corpus / 'queries_test.jsonl'), artifact(tokenizer_path)],
        'embedding_identity': identity, 'query_ids': sorted(q['query_id'] for q in queries)}


def main():
    from ragscope.experiments import retrieval_campaign as module
    from ragscope.experiments.runtime import configure
    home, campaign = Path('D:/RAGScope_Runtime'), '20261014_v2'
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--allow-historical-usage', action='store_true',
                        help='Only the three coordinator-authorized immutable N/A observations')
    args = parser.parse_args()
    configure(cpu_only=True)
    run = home / 'runs' / (campaign + '_retrieval')
    control = run / 'SIMPLE_CACHE_ONLY'
    if args.allow_historical_usage:
        control /= 'HISTORICAL_NA_REUSE'
    control.mkdir(exist_ok=True)
    process = psutil.Process()
    worker = {'pid': process.pid, 'creation_time': process.create_time(), 'command': process.cmdline(),
              'started_at': now(), 'device': 'cpu', 'threads': 4, 'status': 'PREFLIGHT'}
    atomic_json(control / 'worker.json', worker)
    failures = artifact(run / 'embedding_runtime_failures.jsonl')
    receipt = {'status': 'PENDING', 'started_at': now(), 'worker': worker,
        'implementation': [artifact(__file__), artifact(module.__file__)],
        'historical_backend_failures': failures, 'model_service_started': False,
        'cache_only': True, 'real_model_cache_not_mock': True, 'new_backend_calls': 0}
    exit_code = 2
    try:
        historical = run / 'SIMPLE_CACHE_ONLY/MISSING_USAGE_PROVENANCE.json'
        if args.allow_historical_usage:
            CacheOnlyEmbedding.historical_usage = verified_historical_usage(historical)
            decision = {'time': now(), 'authorization': 'Coordinator explicitly authorized reuse of exactly three verified historical backend_usage=null real vectors; no new inference or usage fabrication',
                'source': artifact(historical), 'implementation': receipt['implementation'],
                'allowed_cache_keys': sorted(CacheOnlyEmbedding.historical_usage),
                'service_policy': 'closed; all HTTP/socket requests prohibited',
                'latest_user_gpu_authorization': 'GPU permitted in any stage by latest explicit user direction; this cache-only operation still uses no GPU or model service',
                'experiment_identity_changed': False, 'historical_files_modified': False}
            atomic_json(control / 'DECISION_BEFORE_RUN.json', decision)
            receipt['historical_NA_decision'] = artifact(control / 'DECISION_BEFORE_RUN.json')
        if not port_closed():
            raise CacheOnlyPending('PENDING_RESOURCE: embedding service must remain closed')
        pause = run / 'R3_RUNTIME/RESOURCE_PAUSE_PID30108.json'
        if file_hash(pause) != '8b6e9e388cab318355eed66315a3889c91e8ae6cf62b3d6559cd23bbb8b1d4b5':
            raise CacheOnlyPending('PENDING_SOURCE: R3 release receipt mismatch')
        with FileLock(str(run / 'writer.lock'), timeout=0):
            receipt['writer_lock_available_before'] = True
        pre = preflight(home, campaign, CacheOnlyEmbedding.historical_usage)
        atomic_json(control / 'PREFLIGHT.json', pre)
        methods = ['bm25', 'dense', 'hybrid'] if pre['status'] == 'PASS' else ['bm25']
        worker['status'] = 'RUNNING'
        atomic_json(control / 'worker.json', worker)
        with no_network() as blocked, patch.object(module, 'LocalEmbeddingService', CacheOnlyEmbedding):
            result = module.run_final_contexts(home, campaign, 'qasper', 'http://127.0.0.1:11436',
                home / 'runs' / campaign / 'embedding_identity.json', methods=methods, family='simple')
        receipt.update(result_status=result['status'], methods=methods,
            blocked_network_attempts=len(blocked), observed_cache_hits=CacheOnlyEmbedding.observed_hits,
            query_preflight=artifact(control / 'PREFLIGHT.json'))
        if result['status'] != 'PASS':
            raise CacheOnlyPending('PENDING_RESOURCE_OR_RESULT: inspect original runner summary')
        checks = []
        for method in methods:
            path = run / 'records' / f'TEST_CONTEXT_F512_O0_{method}_neighbor.jsonl'
            config_path = path.with_suffix('.config.json')
            expected_hash = signature(json.loads(config_path.read_text()))
            rows = read_records(path, repair_tail=False)
            if (sorted(r['query_id'] for r in rows) != pre['query_ids']
                    or any(r.get('error') or not r.get('record_hash') or r['config_hash'] != expected_hash for r in rows)):
                raise ValueError('Simple records have an incomplete population, failures or identity mismatch')
            if method != 'bm25' and any(r['embedding_identity'] != pre['embedding_identity']
                    or r['real_model'] is not True or len(r['query_encoding_metrics']) != 1
                    or r['query_encoding_metrics'][0]['cache_hit'] is not True for r in rows):
                raise ValueError('Real cached query encoding is not recorded for every neural row')
            checks.append({'method': method, 'count': len(rows), 'failures': 0,
                           'artifacts': [artifact(path), artifact(config_path)]})
        receipt.update(status='PASS' if len(methods) == 3 else 'PARTIAL_BM25_NEURAL_PENDING',
            independent_record_checks=checks, test_used_for_selection=False)
        exit_code = 0 if len(methods) == 3 else 2
    except CacheOnlyPending as exc:
        receipt.update(status='PENDING', reason=str(exc))
    except Exception as exc:
        receipt.update(status='FAILED', error={'type': type(exc).__name__, 'message': str(exc)})
    finally:
        worker.update(status='EXITING', ended_at=now(), exit_code=exit_code)
        atomic_json(control / 'worker.json', worker)
        receipt.update(ended_at=now(), exit_code=exit_code, service_closed_after=port_closed(),
            available_after_bytes=psutil.virtual_memory().available,
            historical_failure_journal_unchanged=artifact(run / 'embedding_runtime_failures.jsonl') == failures)
        with FileLock(str(run / 'writer.lock'), timeout=0):
            receipt['writer_lock_released'] = True
        atomic_json(control / 'RESULT.json', receipt)
        print(json.dumps({k: receipt.get(k) for k in ('status', 'methods', 'observed_cache_hits', 'reason', 'error', 'writer_lock_released')}), flush=True)
    raise SystemExit(exit_code)


if __name__ == '__main__':
    main()
