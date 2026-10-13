"""Fetch pinned official tokenizers and installed-backend renderer source for audit."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json
from ragscope.experiments.runtime import configure,runtime_home
from ragscope.experiments.assets import download
from ragscope.indexing.identity import atomic_json

configure(cpu_only=True); root=runtime_home()
source=json.loads((root/'runs/20261014_v2/model_sources.json').read_text())
for item in source['observations']:
    model=item['model']
    if not model.startswith('Qwen/Qwen3.5-'): continue
    revision=item['revision']; directory=root/'models/tokenizers'/model.split('/')[-1]
    artifacts=[]
    for name in ('tokenizer.json','tokenizer_config.json','chat_template.jinja','config.json','LICENSE'):
        artifacts.append(download(f'https://huggingface.co/{model}/resolve/{revision}/{name}',directory/name))
    atomic_json(directory/'manifest.json',dict(model=model,revision=revision,artifacts=artifacts))
    print(json.dumps(dict(model=model,revision=revision,status='TOKENIZER_READY')),flush=True)
for name in ('qwen35.go','renderer.go'):
    print(download('https://raw.githubusercontent.com/ollama/ollama/v0.34.4/model/renderers/'+name,root/'downloads/ollama_renderer_0.34.4'/name))
