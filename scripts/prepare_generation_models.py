"""Download immutable official Ollama model artifacts to D, without loading a model."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
import hashlib
import json
import urllib.request
from ragscope.experiments.assets import download
from ragscope.experiments.runtime import configure,runtime_home
from ragscope.indexing.identity import atomic_json,artifact

configure(cpu_only=True)
parser=argparse.ArgumentParser(); parser.add_argument('--tags',nargs='+',default=['2b-q4_K_M','4b-q4_K_M','9b-q4_K_M'])
args=parser.parse_args(); root=runtime_home(); store=root/'models/ollama'
for tag in args.tags:
    manifest_path=store/'manifests/registry.ollama.ai/library/qwen3.5'/tag
    source_path=root/'models/generation'/f'qwen3.5_{tag}.json'
    if source_path.exists():
        source=json.loads(source_path.read_text()); raw=manifest_path.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=source['manifest_sha256']: raise ValueError('Existing manifest changed')
    else:
        url='https://registry.ollama.ai/v2/library/qwen3.5/manifests/'+tag
        with urllib.request.urlopen(url,timeout=30) as response: raw=response.read()
        source={'model':'qwen3.5:'+tag,'source':url,'manifest_sha256':hashlib.sha256(raw).hexdigest()}
    manifest=json.loads(raw)
    source['artifacts']=[]
    for layer in [manifest['config']]+manifest['layers']:
        digest=layer['digest']; path=store/'blobs'/digest.replace(':','-')
        item=download('https://registry.ollama.ai/v2/library/qwen3.5/blobs/'+digest,path,digest.split(':')[1],layer['size'])
        source['artifacts'].append(dict(item,media_type=layer['mediaType']))
    manifest_path.parent.mkdir(parents=True,exist_ok=True)
    if not manifest_path.exists(): manifest_path.write_bytes(raw)
    source.update(manifest=artifact(manifest_path),config=json.loads((store/'blobs'/manifest['config']['digest'].replace(':','-')).read_text()),model_loaded=False)
    atomic_json(source_path,source)
    print(json.dumps({'model':source['model'],'revision':source['manifest_sha256'],'config':source['config'],'status':'VERIFIED_DOWNLOADED'}),flush=True)
