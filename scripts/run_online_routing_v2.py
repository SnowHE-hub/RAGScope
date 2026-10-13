"""Freeze dev-selected policies, or run their real paired local online comparison."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import json
from time import perf_counter

from filelock import FileLock

from freeze_generation_v2 import model_client_factory, shared_context, verified_corpus
from run_generation_campaign import GuardedClient, verify_owned_service
from ragscope.experiments.generation_campaign import GenerationBudget, load_campaign
from ragscope.experiments.runtime import configure, runtime_home, require_ram
from ragscope.generation.resources import MeasuredClient
from ragscope.indexing.identity import verify_artifacts
from ragscope.routing.online_campaign import freeze_online, load_router, run_online, verified_body


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', required=True, type=Path)
    parser.add_argument('--routing-dir', required=True, type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--mode', required=True, choices=['freeze', 'run'])
    parser.add_argument('--embedding-url', default='http://127.0.0.1:11436')
    parser.add_argument('--embedding-identity', type=Path)
    parser.add_argument('--lease', type=Path)
    parser.add_argument('--service-receipt', type=Path)
    parser.add_argument('--max-tasks', type=int)
    args = parser.parse_args()
    configure(cpu_only=True)  # All in-process router/reranker work is CPU-only.
    output = args.output or args.routing_dir/'online'
    if args.mode == 'freeze':
        result = freeze_online(args.campaign, args.routing_dir, output)
        print(json.dumps({'status': 'FROZEN', 'selected': result['selected'], 'online_hash': result['online_hash']}))
        return 0
    frozen = verified_body(output/'ONLINE_MANIFEST.json', 'online_hash')
    if not verify_artifacts(frozen['sources']+[frozen['implementation']]):
        raise ValueError('Frozen online source artifact changed')
    manifest, _ = load_campaign(args.campaign)
    spec = manifest['provenance']['spec']
    home = runtime_home()
    rootrun = home/'runs'/spec['campaign']
    lease = args.lease or rootrun/'gpu_lease.json'
    receipt = args.service_receipt or rootrun/'generation_service.json'
    # Validate process, GPU lease and backend settings before CPU model loading.
    verify_owned_service(receipt, lease, frozen['model_identity'])
    queries, units, _ = verified_corpus(spec['corpus_dir'])
    dev_ids = {q['query_id'] for q in queries['dev']}
    common = [model_client_factory(spec['models'][name], dev_ids)[0]
              for name in ('qwen_2b', 'qwen_4b', 'qwen_9b')]
    for name, client in zip(('qwen_2b', 'qwen_4b', 'qwen_9b'), common):
        if client.identity != manifest['models'][name]:
            raise ValueError('Frozen generator counter identity changed')
    from ragscope.routing.online_retrieval import create_live_retriever
    from ragscope.retrieval.verified_reranker import load_verified_reranker
    # Backend owns runs/gpu.lock. Client lock excludes any other local generation
    # caller while models/index are loaded and throughout the measured requests.
    with FileLock(str(home/'runs/generation-client.lock'), timeout=0):
        # BGE adds CPU weight/activation memory beside resident embedding and
        # the GPU service. Reject before Torch/model import if margin is absent.
        require_ram(2*1024**3)
        loaded_at = perf_counter()
        reranker = load_verified_reranker(home, device='cpu')
        reranker_load_ms = (perf_counter()-loaded_at)*1000
        retriever = create_live_retriever(home, spec['campaign'], family='structured',
            embedding_url=args.embedding_url, embedding_identity_path=args.embedding_identity,
            reranker=reranker)
        loaded_at = perf_counter()
        routers = {name: load_router(name, args.routing_dir) for name in frozen['selected'].values()}
        retriever.load_timings_ms.update(reranker_load_ms=reranker_load_ms,
                                         router_load_ms=(perf_counter()-loaded_at)*1000)
        measured = MeasuredClient(GuardedClient(common[0], receipt, lease), receipt, gpu_device=0, interval=1.0)
        result = run_online(args.campaign, args.routing_dir, output, measured,
            GenerationBudget(rootrun/'generation_budget.json'), retriever, routers,
            common_clients=common, shared_context=shared_context, evidence_units=units, max_tasks=args.max_tasks)
    return 0 if result['status'] == 'COMPLETE' else 2


if __name__ == '__main__':
    raise SystemExit(main())
