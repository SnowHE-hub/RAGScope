"""Official LiquidAI Ollama artifact, cross-checked against official HF GGUF SHA."""
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
parser=argparse.ArgumentParser(); parser.add_argument('--metadata-only',action='store_true'); args=parser.parse_args()
configure(cpu_only=True); root=runtime_home(); name='liquidai/lfm2.5-1.2b-instruct'; tag='q4_k_m'
url=f'https://registry.ollama.ai/v2/{name}/manifests/{tag}'
with urllib.request.urlopen(url,timeout=30) as response: raw=response.read()
manifest=json.loads(raw)
weight=next(layer for layer in manifest['layers'] if layer['mediaType']=='application/vnd.ollama.image.model')
assert weight['digest']=='sha256:b1b3de114215d9507409a662a501a631095a479a419584e8a2ded6304b19b4f5' and weight['size']==730895168
artifacts=[]
for layer in [manifest['config']]+manifest['layers']:
    if args.metadata_only and layer is weight: continue
    digest=layer['digest']; path=root/'models/ollama/blobs'/digest.replace(':','-')
    item=download(f'https://registry.ollama.ai/v2/{name}/blobs/{digest}',path,digest.split(':')[1],layer['size'])
    artifacts.append(dict(item,media_type=layer['mediaType']))
tokenizer_model='LiquidAI/LFM2.5-1.2B-Instruct'; revision='0f604ada3f766f9f257460c4c9f0b5d6f69d431b'
directory=root/'models/tokenizers/LFM2.5-1.2B-Instruct'; tokenizer_artifacts=[]
for filename in ('tokenizer.json','tokenizer_config.json','chat_template.jinja','config.json','LICENSE'):
    tokenizer_artifacts.append(download(f'https://huggingface.co/{tokenizer_model}/resolve/{revision}/{filename}',directory/filename))
atomic_json(directory/'manifest.json',dict(model=tokenizer_model,revision=revision,artifacts=tokenizer_artifacts))
source=dict(model=name+':'+tag,source=url,manifest_sha256=hashlib.sha256(raw).hexdigest(),artifacts=artifacts,
            hf_model='LiquidAI/LFM2.5-1.2B-Instruct-GGUF',hf_revision='8ed288026e23958ad9dfa92d53ed773a8eee7125',
            hf_weight_sha256=weight['digest'].split(':')[1],license='LFM Open License v1.0; local non-commercial research',
            config=json.loads((root/'models/ollama/blobs'/manifest['config']['digest'].replace(':','-')).read_text()))
if not args.metadata_only:
    destination=root/'models/ollama/manifests/registry.ollama.ai'/name/tag
    destination.parent.mkdir(parents=True,exist_ok=True)
    if destination.exists() and destination.read_bytes()!=raw: raise ValueError('Existing LFM manifest changed')
    if not destination.exists(): destination.write_bytes(raw)
    source['manifest']=artifact(destination)
atomic_json(root/'models/generation'/('lfm2.5_metadata.json' if args.metadata_only else 'lfm2.5_q4_k_m.json'),source)
print(json.dumps(source),flush=True)
