"""Derive text-only Ollama aliases without changing any official language weights.

The scientific-text task does not use the vision projector. Removing that layer
avoids loading/warming an unused image encoder on a 16GB host.
"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import hashlib
import json
from ragscope.experiments.runtime import configure,runtime_home
from ragscope.indexing.identity import atomic_json,artifact,verify_artifacts
configure(cpu_only=True); root=runtime_home()
for size in ('2b','4b','9b'):
    sourcepath=root/'models/generation'/f'qwen3.5_{size}-q4_K_M.json'
    if not sourcepath.exists(): continue
    source=json.loads(sourcepath.read_text())
    if not verify_artifacts(source['artifacts']): raise ValueError('Official source artifact mismatch')
    manifest=json.loads(Path(source['manifest']['path']).read_text())
    removed=[layer for layer in manifest['layers'] if layer['mediaType']=='application/vnd.ollama.image.projector']
    manifest['layers']=[layer for layer in manifest['layers'] if layer not in removed]
    raw=json.dumps(manifest,separators=(',',':')).encode()
    tag=f'{size}-q4_K_M'; destination=root/'models/ollama/manifests/registry.ollama.ai/library/ragscope-qwen3.5'/tag
    destination.parent.mkdir(parents=True,exist_ok=True)
    if destination.exists() and destination.read_bytes()!=raw: raise ValueError('Text-only alias changed')
    if not destination.exists(): destination.write_bytes(raw)
    output={**source,'model':'ragscope-qwen3.5:'+tag,'manifest_sha256':hashlib.sha256(raw).hexdigest(),
            'manifest':artifact(destination),'upstream_manifest_sha256':source['manifest_sha256'],
            'upstream_model':source['model'],'removed_unused_vision_projectors':removed,
            'artifacts':[a for a in source['artifacts'] if a['media_type']!='application/vnd.ollama.image.projector'],
            'derivation':'Official language weights/config/template unchanged; unused vision layer omitted for text-only task.'}
    atomic_json(root/'models/generation/text_only'/sourcepath.name,output)
    print(json.dumps({'model':output['model'],'revision':output['manifest_sha256'],'removed_bytes':sum(x['size'] for x in removed)}),flush=True)
