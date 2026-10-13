"""Verify local UI packages without starting a service or generating text."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import json

from freeze_generation_v2 import model_client_factory, RUNTIME_POLICY
from ragscope.experiments.runtime import configure, runtime_home
from ragscope.indexing.identity import artifact, atomic_json, verify_artifacts
from ui.model_identity import verified_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    configure(cpu_only=True)
    frozen = json.loads(args.source.read_text(encoding='utf-8'))
    corpus = Path(frozen['provenance']['spec']['corpus_dir'])
    dev = {json.loads(line)['query_id'] for line in (corpus/'queries_dev.jsonl').read_text(encoding='utf-8').splitlines() if line}
    root = runtime_home()/'models/ollama'
    entries = []
    for key, spec in frozen['provenance']['spec']['models'].items():
        # The E3 shard did not use LFM. Apply the same verified runtime policy
        # that freeze_all applies to every model, without changing that shard.
        spec = {**spec, 'client_options': {**spec.get('client_options', {}), 'backend_runtime_config': RUNTIME_POLICY}}
        client, _ = model_client_factory(spec, dev, verify_weights=False)
        name, tag = client.model.split(':')
        namespace, name = name.split('/') if '/' in name else ('library', name)
        manifest = root/'manifests/registry.ollama.ai'/namespace/name/tag
        entry = {'id': key, 'model': client.model, 'provider': client.provider,
                 'base_url': client.base_url, 'identity': client.identity, 'ollama_manifest': artifact(manifest)}
        _, layer = verified_manifest(entry)
        weight = {'path': str(root/'blobs'/layer['digest'].replace(':', '-')),
                  'bytes': layer['size'], 'sha256': client.identity['weight_sha256']}
        if not verify_artifacts([weight]):
            raise ValueError('Actual local weight bytes differ')
        entry.update(weight=weight, counter_kind=spec.get('counter_kind', 'qwen35'),
            renderer_source=spec['renderer_source'], tokenizer_path=spec['tokenizer_dir'],
            num_ctx=client.num_ctx, num_gpu=999, service_device_verified='gpu', think=client.think,
            timeout=client.timeout, ready=False,
            reason='Local package verified; project service currently released, final application QA pending')
        entries.append(entry)
        print(key, 'PASS_LOCAL_WEIGHT_AND_MANIFEST', flush=True)
    body = {'status': 'PASS_LOCAL_PACKAGES_NOT_SERVICE_READINESS', 'source': artifact(args.source),
            'entries': entries, 'generation_calls': 0, 'gpu_used': False}
    if args.output.exists():
        if json.loads(args.output.read_text(encoding='utf-8')) != body:
            raise ValueError('Frozen UI model pins changed; choose a new output')
    else:
        atomic_json(args.output, body)


if __name__ == '__main__':
    main()
