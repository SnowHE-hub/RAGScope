"""Build a verified runtime UI candidate, without publishing or starting models.

Only loopback GET metadata probes are permitted. The caller's output must stay
under the runtime candidate directory; final application QA/publication is separate.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

from ragscope.indexing.identity import artifact, atomic_json, file_hash, signature, verify_artifacts


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def metadata_get(base_url, route):
    from ragscope.generation.llm_client import _local_url, _NoRedirect
    if route not in {'/health', '/api/version', '/api/tags', '/api/ps'}:
        raise ValueError('Only read-only service metadata routes are allowed')
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())
    try:
        with opener.open(_local_url(base_url) + route, timeout=2) as response:
            return {'available': True, 'payload': json.load(response)}
    except Exception as exc:
        return {'available': False, 'error': type(exc).__name__}


def validate_selection(index, candidate, context, candidate_hash):
    identity = index['identity']
    if (index['signature'] != signature(identity)
            or candidate['split'] != 'dev' or context['split'] != 'dev'
            or context['test_used_for_selection'] is not False
            or context['candidate_selection_sha256'] != candidate_hash
            or candidate['selected']['structured'] != identity['group']['id']
            or candidate['data_manifest_sha256'] != identity['data_manifest_sha256']
            or identity['scope'] != 'global_or_explicit_paper'
            or len(set(identity['selected_doc_ids'])) != 509
            or index['count'] != 6551):
        raise ValueError('UI candidate is not bound to the frozen all509 dev selection')
    return context['selected'][identity['group']['id']]


def assemble_candidate(index, manifest_path, manifest_hash, chunk_manifest_path,
                       expansion, embedding, reranker, models, execution, provenance):
    """Pure assembly: installed packages are distinct from live readiness."""
    entries = []
    for method in ('bm25', 'dense', 'hybrid', 'hybrid+rerank'):
        ready = method == 'bm25' or embedding['ready']
        if method == 'hybrid+rerank':
            ready = ready and reranker['ready']
        entries.append({'id': 'R512_' + method.upper() + '_all509',
            'method': method, 'ready': ready, 'artifacts_verified': True,
            'manifest_path': str(manifest_path), 'manifest_sha256': manifest_hash,
            'index_signature': index['signature'], 'documents': 509, 'chunks': index['count'],
            'chunk_manifest_path': str(chunk_manifest_path), 'expansion': expansion,
            'parent_ready': False, 'candidate_depth': 50, 'final_k': 10,
            'embedding': deepcopy(embedding), 'reranker': deepcopy(reranker),
            'reason': ('全509篇索引已验证；' + ('当前路径可用' if ready else '当前路径不可用：CPU embedding 服务未就绪'))})
    return {'schema_version': 'ragscope-ui-v2', 'publication_status': 'CANDIDATE_AWAITING_GUI_QA',
        'status': '全509篇 R512/neighbor 候选；服务状态是本次只读快照，实际运行仍须重新核验。',
        'default_model': 'qwen_2b', 'default_retriever': 'R512_HYBRID_all509',
        'default_router': 'fixed', 'default_scope': 'qasper:1503.00841',
        'reference_budget': 2048, 'indexes': entries, 'models': models, 'routers': [],
        'routing_readiness': {'default': 'fixed', 'nonfixed_enabled': False,
            'reason': 'No bound full-dev quality/paired online threshold for cold application loads'},
        'default_policy': {'model': 'smallest primary Qwen (2B) with completed dev preflight; no test-based selection',
            'retrieval': 'frozen dev-only structured selection, fixed Hybrid route',
            'unavailable_behavior': 'Existing UI exposes only ready entries; retrieval-only/BM25 while services are unavailable'},
        'execution': execution, 'result_files': [], 'provenance': provenance}


def build_candidate(home, campaign, output):
    from filelock import FileLock, Timeout
    from ui.chat_app import validate_index_entry, make_client
    from ui.model_identity import verified_manifest, tag_matches
    from ragscope.retrieval.verified_reranker import verified_checkpoint
    from run_generation_campaign import verify_owned_service

    home, output = Path(home).resolve(), Path(output).resolve()
    directory = home / 'runs' / campaign / 'ui_candidate'
    if not output.is_relative_to(directory.resolve()):
        raise ValueError('Output must remain under runtime ui_candidate; no project publication')
    run, retrieval = home / 'runs' / campaign, home / 'runs' / (campaign + '_retrieval')
    published = ROOT / 'configs/v2/ui.json'
    before = artifact(published)
    candidate_path, context_path = retrieval / 'CANDIDATE_SELECTION.json', retrieval / 'CONTEXT_SELECTION.json'
    selected, context = read(candidate_path), read(context_path)
    ready_receipt = read(retrieval / 'STRUCTURED_INDEX_READY.json')
    if not verify_artifacts([ready_receipt['manifest']]):
        raise ValueError('Structured index readiness artifact changed')
    manifest_path = Path(ready_receipt['manifest']['path'])
    index = validate_index_entry({'manifest_path': str(manifest_path), 'manifest_sha256': ready_receipt['manifest']['sha256']})
    expansion = validate_selection(index, selected, context, file_hash(candidate_path))
    if expansion != 'neighbor' or not verify_artifacts(context['evidence']['R512']):
        raise ValueError('Frozen dev context choice is not verified')
    chunk_manifest = manifest_path.parent.parent / 'chunk_manifest.json'
    chunk = read(chunk_manifest)
    if chunk['signature'] != signature(chunk['identity']) or not verify_artifacts(chunk['artifacts']):
        raise ValueError('Chunk coordinates/artifacts changed')
    embedding_identity = read(run / 'embedding_identity.json')
    if embedding_identity != index['identity']['embedding']:
        raise ValueError('Embedding identity differs from the built index')
    tokenizer = home / 'models/qwen3-embedding-0.6b/tokenizer.json'
    tokenizer_identity = index['identity']['tokenizer']
    if file_hash(tokenizer) != tokenizer_identity['tokenizer_file_sha256']:
        raise ValueError('Embedding tokenizer bytes changed')
    probes = {'embedding_health': metadata_get('http://127.0.0.1:11436', '/health')}
    embedding_ready, embedding_reason = False, 'CPU embedding service is unavailable'
    if probes['embedding_health']['available']:
        import psutil
        service = read(run / 'embedding_service.json')
        try:
            proc = psutil.Process(service['pid'])
            command = proc.cmdline()
            embedding_ready = (abs(proc.create_time() - service['creation_time']) < .01
                and command == service['command'] and '-ngl' in command
                and command[command.index('-ngl')+1] == '0'
                and command[command.index('-t')+1] == '4'
                and command[command.index('-tb')+1] == '4'
                and '--no-op-offload' in command
                and service['model_sha256'] == embedding_identity['weight_sha256'])
            embedding_reason = 'Matched owned CPU4 service' if embedding_ready else 'CPU service identity mismatch'
        except (psutil.NoSuchProcess, KeyError, ValueError) as exc:
            embedding_reason = type(exc).__name__
    embedding = {'ready': embedding_ready, 'reason': embedding_reason,
        'base_url': 'http://127.0.0.1:11436', 'service_device_verified': 'cpu',
        'identity': embedding_identity, 'tokenizer_path': str(tokenizer),
        'tokenizer_id': tokenizer_identity['tokenizer_id'],
        'tokenizer_revision': tokenizer_identity['tokenizer_revision'],
        'cache_dir': str(home / 'cache/ragscope/v2_embeddings'), 'max_input_tokens': 1100}

    # Completed dev reranking identity only; no test answers are read to choose defaults.
    rerank_summary_path = retrieval / 'summaries/E1_R512_hybrid_rerank_neighbor.json'
    rerank_summary = read(rerank_summary_path)
    if rerank_summary['split'] != 'dev' or rerank_summary['n_total'] != 100 or rerank_summary['n_failed'] != 0:
        raise ValueError('Real dev reranker verification is incomplete')
    _, checkpoint = verified_checkpoint(home)
    reranker_identity = rerank_summary['reranker_identity']
    if checkpoint != reranker_identity['checkpoint_manifest']:
        raise ValueError('Reranker package differs from real dev execution')
    reranker = {'ready': True, 'identity': reranker_identity, 'resident': False,
        'reason': 'Verified real CPU dev run and package; load only on explicit request after RAM guard',
        'required_python': str(home / 'envs/models/Scripts/python.exe')}

    pins_path = run / 'UI_MODEL_PACKAGE_PINS.json'
    pins = read(pins_path)
    if {m['id'] for m in pins['entries']} != {'lfm', 'qwen_2b', 'qwen_4b', 'qwen_9b'}:
        raise ValueError('Expected the four pinned application model packages')
    for route in ('/api/version', '/api/tags', '/api/ps'):
        probes[route] = metadata_get('http://127.0.0.1:11435', route)
    models, model_checks = [], []
    for pin in pins['entries']:
        verified_manifest(pin)
        if not verify_artifacts([pin['weight']]) or pin['weight']['sha256'] != pin['identity']['weight_sha256']:
            raise ValueError('Pinned model weight bytes changed: ' + pin['id'])
        client = make_client(pin)  # Tokenizer/template only; never model weights or chat.
        if client.identity != pin['identity']:
            raise ValueError('UI client identity differs from the pinned model')
        available, reason = False, 'Generation metadata service unavailable'
        try:
            verify_owned_service(run / 'generation_service.json', run / 'gpu_lease.json', pin['identity'])
            available = (probes['/api/version']['available'] and probes['/api/tags']['available']
                and probes['/api/version']['payload'].get('version') == pin['identity']['backend_version']
                and tag_matches(pin, probes['/api/tags']['payload'].get('models', [])))
            reason = 'Owned service metadata identity verified' if available else 'Service version/tag mismatch'
        except Exception as exc:
            reason = 'Owned service unavailable: ' + type(exc).__name__
        model_checks.append({'id': pin['id'], 'package_verified': True, 'client_identity_verified': True,
            'service_metadata_verified': available, 'reason': reason})
        models.append({**deepcopy(pin), 'ready': available, 'package_verified': True, 'reason': reason})
        del client
    generation_free = False
    try:
        with FileLock(str(home / 'runs/generation-client.lock'), timeout=0):
            generation_free = True
    except Timeout:
        pass
    if not generation_free:
        for model in models:
            model.update(ready=False, reason='项目生成正在运行；需等待 generation-client.lock 释放后刷新配置')
    sources = [artifact(p) for p in (candidate_path, context_path, manifest_path, chunk_manifest,
        run / 'embedding_identity.json', tokenizer, pins_path, rerank_summary_path)]
    config = assemble_candidate(index, manifest_path, file_hash(manifest_path), chunk_manifest,
        expansion, embedding, reranker, models, {'campaign_run': str(run)}, sources)
    # Register only independently completed immutable sources. This does not choose defaults.
    for method in ('bm25', 'dense', 'hybrid', 'hybrid_rerank'):
        recorded_expansion = 'none' if method == 'dense' else 'neighbor'
        config['result_files'].append(str(retrieval / 'records' / f'E1_R512_{method}_{recorded_expansion}.jsonl'))
    config['result_files'] += [str(home / 'runs' / (campaign + '_generation_e3') / 'records' / f'E3_{model}.jsonl')
                               for model in ('qwen_2b', 'qwen_4b', 'qwen_9b')]
    for path in config['result_files']:
        if not Path(path).is_file() or not Path(path).with_suffix('.config.json').is_file():
            raise ValueError('Registered result or sidecar is absent')
    if before != artifact(published):
        raise ValueError('Published project configuration changed concurrently')
    atomic_json(output, config)
    receipt = {'status': 'PASS_READ_ONLY_CANDIDATE_NOT_PUBLISHED',
        'observed_at': datetime.now(timezone.utc).isoformat(), 'candidate': artifact(output),
        'script': artifact(__file__), 'published_config_unchanged': before,
        'sources': sources, 'index_count': index['count'], 'document_count': 509,
        'index_signature': index['signature'], 'expansion': expansion,
        'model_checks': model_checks, 'generation_client_lock_available': generation_free,
        'embedding_ready': embedding_ready, 'metadata_probes': probes,
        'model_inference_calls': 0, 'model_service_started': False, 'main_campaign_writer_used': False,
        'test_scores_used_for_defaults': False, 'nonfixed_routes_enabled': False,
        'limitations': ['Runtime availability is a snapshot; GUI/GPU QA and publication are still pending',
                       'BGE package verification does not claim a resident model or concurrent RAM availability']}
    atomic_json(output.with_suffix('.verification.json'), receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--home', default='D:/RAGScope_Runtime')
    parser.add_argument('--campaign', default='20261014_v2')
    parser.add_argument('--output')
    args = parser.parse_args()
    output = args.output or str(Path(args.home) / 'runs' / args.campaign / 'ui_candidate/ui.json')
    receipt = build_candidate(args.home, args.campaign, output)
    print(json.dumps({key: receipt[key] for key in ('status', 'candidate', 'embedding_ready', 'generation_client_lock_available')}))


if __name__ == '__main__':
    main()
