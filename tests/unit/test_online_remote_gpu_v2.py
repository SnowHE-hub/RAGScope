"""Transport contracts only; no model, GPU or actual HTTP request is made."""
import io
import json
import os
from pathlib import Path
import sys
import time

import psutil
import pytest

from ragscope.generation.llm_client import GenerationError, LocalLLMClient
from ragscope.indexing.identity import atomic_json
from ragscope.routing.online_campaign import authorize_remote_gpu


@pytest.fixture
def owned(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2]/'scripts'))
    from run_generation_campaign import verify_owned_service
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    lease, service_path = tmp_path/'lease.json', tmp_path/'service.json'
    lease_body = {'status': 'ACQUIRED', 'owner': 'ragscope', 'safety_verified': True, 'gpu_device': 0,
                  'host_platform': sys.platform, 'host_boot_time_unix': psutil.boot_time(), 'validated_at': time.time()}
    service = {'status': 'READY', 'base_url': 'http://127.0.0.1:11435', 'backend_version': 'fixture',
               'pid': 123, 'creation_time': 100, 'binary': str(tmp_path/'owned-ollama.exe'),
               'environment': {'OLLAMA_MAX_LOADED_MODELS': '1', 'OLLAMA_NUM_PARALLEL': '1',
                               'OLLAMA_NO_CLOUD': '1', 'LLAMA_ARG_CACHE_RAM': '0'}}
    atomic_json(lease, lease_body)
    atomic_json(service_path, service)

    class Process:
        def create_time(self): return 100
        def exe(self): return str(tmp_path/'owned-ollama.exe')

    monkeypatch.setattr(psutil, 'Process', lambda pid: Process())
    requests = []

    class Opener:
        def open(self, request, timeout):
            requests.append(json.loads(request.data))
            return io.BytesIO(b'{"message":{"content":"fixture"}}')

    monkeypatch.setattr('urllib.request.build_opener', lambda *args: Opener())
    client = LocalLLMClient('fixture', num_gpu=999, service_device_verified='gpu',
        backend_version='fixture', backend_runtime_config={
            'prompt_ram_cache_mib': 0, 'prompt_cache_policy': 'backend_history_disabled'})
    return {'client': client, 'lease': lease, 'lease_body': lease_body, 'service': service,
            'service_path': service_path, 'requests': requests,
            'validator': lambda identity: verify_owned_service(service_path, lease, identity)}


def test_default_cpu_policy_rejects_gpu_before_http(owned):
    with pytest.raises(GenerationError, match='prohibited'):
        owned['client'].chat([])
    assert owned['requests'] == [] and os.environ['CUDA_VISIBLE_DEVICES'] == ''


def test_verified_owned_remote_gpu_allows_http_without_exposing_local_cuda(owned):
    client = owned['client']
    original_identity = dict(client.identity)
    authorize_remote_gpu(client, owned['validator'])
    assert client.chat([]) == 'fixture'
    assert owned['requests'][0]['options']['num_gpu'] == 999
    assert client.identity == original_identity and os.environ['CUDA_VISIBLE_DEVICES'] == ''
    # Revoking the lease after authorization must block the very next request.
    atomic_json(owned['lease'], {**owned['lease_body'], 'status': 'RELEASED'})
    with pytest.raises(GenerationError, match='lease'):
        client.chat([])
    assert len(owned['requests']) == 1


@pytest.mark.parametrize('change', ['missing_lease', 'old_boot', 'wrong_process', 'unknown_endpoint', 'not_ready'])
def test_unverified_remote_service_never_authorizes_or_calls_http(owned, change):
    if change == 'missing_lease':
        owned['lease'].unlink()
    elif change == 'old_boot':
        atomic_json(owned['lease'], {**owned['lease_body'], 'host_boot_time_unix': psutil.boot_time()-1000})
    else:
        updates = {'wrong_process': {'creation_time': 101},
                   'unknown_endpoint': {'base_url': 'http://127.0.0.1:11434'},
                   'not_ready': {'status': 'EXITED'}}
        atomic_json(owned['service_path'], {**owned['service'], **updates[change]})
    with pytest.raises(GenerationError):
        authorize_remote_gpu(owned['client'], owned['validator'])
    assert owned['client'].remote_gpu_authorizer is None
    assert owned['requests'] == [] and os.environ['CUDA_VISIBLE_DEVICES'] == ''


def test_bare_boolean_is_not_remote_authority():
    with pytest.raises(ValueError, match='validator'):
        LocalLLMClient('fixture', num_gpu=999, remote_gpu_authorizer=True)


def test_cpu_client_cannot_be_relabeled_as_remote_gpu(owned):
    client = LocalLLMClient('fixture', num_gpu=0, service_device_verified='cpu')
    with pytest.raises(ValueError, match='verified GPU'):
        authorize_remote_gpu(client, owned['validator'])
    assert client.remote_gpu_authorizer is None and os.environ['CUDA_VISIBLE_DEVICES'] == ''
