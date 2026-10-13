"""Bind a UI Ollama tag to its frozen manifest and actual local weight bytes."""
from __future__ import annotations

import json
from pathlib import Path
import urllib.request


def verified_manifest(entry):
    from ragscope.indexing.identity import verify_artifacts
    frozen = entry['ollama_manifest']
    if not verify_artifacts([frozen]):
        raise ValueError('生成模型 manifest 已改变或缺失')
    body = json.loads(Path(frozen['path']).read_text(encoding='utf-8'))
    layers = [layer for layer in body.get('layers', [])
              if layer.get('mediaType') == 'application/vnd.ollama.image.model']
    expected = entry['identity']['weight_sha256']
    if len(layers) != 1 or layers[0]['digest'] != 'sha256:'+expected:
        raise ValueError('生成模型 manifest 与冻结权重不一致')
    return body, layers[0]


def tag_matches(entry, observed_models):
    verified_manifest(entry)
    found = [row for row in observed_models if row.get('name', row.get('model')) == entry['model']]
    return len(found) == 1 and found[0].get('digest') == entry['ollama_manifest']['sha256']


def verify_model_package(entry, service_path):
    from ragscope.generation.llm_client import _local_url, _NoRedirect
    from ragscope.indexing.identity import verify_artifacts
    if entry['provider'] != 'ollama':
        raise ValueError('交互生成尚未核验该后端的实际权重；不能仅采用配置身份')
    _, layer = verified_manifest(entry)
    service = json.loads(Path(service_path).read_text(encoding='utf-8'))
    model_root = Path(service['models_path']).resolve()
    if Path(service['environment']['OLLAMA_MODELS']).resolve() != model_root:
        raise ValueError('服务实际模型目录与回执不一致')
    name, separator, tag = entry['model'].rpartition(':')
    if not separator:
        name, tag = entry['model'], 'latest'
    parts = name.split('/')
    if len(parts) == 1:
        parts.insert(0, 'library')
    if len(parts) != 2 or any(part in ('', '.', '..') or '\\' in part or ':' in part for part in [*parts, tag]):
        raise ValueError('交互模型名称必须明确指向已核验的本地 registry tag')
    expected_manifest = model_root/'manifests'/'registry.ollama.ai'/parts[0]/parts[1]/tag
    if Path(entry['ollama_manifest']['path']).resolve() != expected_manifest.resolve():
        raise ValueError('冻结 manifest 不属于当前服务的同名模型')
    weight = model_root/'blobs'/layer['digest'].replace(':', '-')
    if not verify_artifacts([{'path': str(weight), 'bytes': layer['size'], 'sha256': entry['identity']['weight_sha256']}]):
        raise ValueError('生成权重实际字节已改变或缺失')
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())
    with opener.open(_local_url(entry['base_url'])+'/api/tags', timeout=2) as response:
        models = json.load(response).get('models', [])
    if not tag_matches(entry, models):
        raise ValueError('服务同名 tag 的实际 digest 与冻结模型不一致')


class PinnedUIClient:
    def __init__(self, client, entry, service_path):
        self.client, self.entry, self.service_path = client, entry, service_path

    def __getattr__(self, name):
        return getattr(self.client, name)

    def chat(self, messages, **kwargs):
        # Runs while the shared generation-client lock is held, immediately
        # before the transport call. An unchanged model name is insufficient.
        verify_model_package(self.entry, self.service_path)
        return self.client.chat(messages, **kwargs)
