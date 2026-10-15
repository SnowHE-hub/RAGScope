"""Private v3 Ollama lifecycle and durable counted local requests."""
from __future__ import annotations
import json
import os
from pathlib import Path
import socket
import subprocess
import time

import psutil
import requests

from ragscope.experiments.runtime import require_ram, validate_gpu_lease
from ragscope.generation.llm_client import LocalLLMClient
from ragscope.indexing.identity import artifact, atomic_json, file_hash
from ragscope.v3.budget import Budget
from ragscope.v3.gpu import GpuSession


class CapturedClient(LocalLLMClient):
    def _request(self, path, payload):
        self.raw_response = None
        response = super()._request(path, payload)
        self.raw_response = response
        return response


class ManagedOllama:
    """One root-owned, stateless backend. Never touches a v2 stop/lease file."""
    def __init__(self, run, binary, *, port=11435, num_ctx=8192, startup_ram_bytes=2*1024**3):
        self.run, self.binary = Path(run), Path(binary).resolve()
        self.runtime = self.run.parents[1]
        self.port, self.num_ctx = port, num_ctx
        self.startup_ram_bytes = startup_ram_bytes
        self.base_url = f'http://127.0.0.1:{port}'
        self.process = None
        self.session = requests.Session()
        self.session.trust_env = False

    def __enter__(self):
        require_ram(self.startup_ram_bytes)
        if not self.binary.is_relative_to(self.runtime / 'tools') or not self.binary.is_file():
            raise ValueError('Private executable must be an existing verified runtime tool')
        with socket.socket() as probe:
            probe.bind(('127.0.0.1', self.port))
        self.gpu = GpuSession(self.run)
        self.gpu.__enter__()
        self.directory = self.run / 'services' / f'{os.getpid()}_{time.time_ns()}'
        self.directory.mkdir(parents=True)
        self.path = self.directory / 'service.json'
        self.log = (self.directory / 'service.log').open('a', encoding='utf-8')
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES='0', OLLAMA_HOST=f'127.0.0.1:{self.port}',
            OLLAMA_MODELS=str(self.runtime / 'models/ollama'), OLLAMA_NO_CLOUD='1',
            OLLAMA_MAX_LOADED_MODELS='1', OLLAMA_NUM_PARALLEL='1',
            OLLAMA_CONTEXT_LENGTH=str(self.num_ctx), OLLAMA_GPU_OVERHEAD=str(512 * 1024**2),
            OLLAMA_KEEP_ALIVE='5m', OLLAMA_DEBUG='1', LLAMA_ARG_CACHE_RAM='0')
        try:
            self.process = subprocess.Popen([str(self.binary), 'serve'], stdout=self.log,
                stderr=subprocess.STDOUT, env=env, cwd=self.run,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
            owned = psutil.Process(self.process.pid)
            self.record = {'status': 'STARTING', 'pid': owned.pid, 'creation_time': owned.create_time(),
                'binary': str(self.binary), 'binary_sha256': file_hash(self.binary),
                'argv': owned.cmdline(), 'cwd': str(self.run), 'base_url': self.base_url,
                'lease': self.gpu.receipt,
                'environment': {k: env[k] for k in env if k.startswith(('OLLAMA_', 'LLAMA_ARG_')) or k == 'CUDA_VISIBLE_DEVICES'},
                'original_jobs_policy': 'USER_MANUAL_RESUME'}
            atomic_json(self.path, self.record)
            deadline = time.monotonic() + 60
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    raise RuntimeError('Private service exited during startup')
                try:
                    response = self.session.get(self.base_url + '/api/version', timeout=2)
                    response.raise_for_status()
                    self.record.update(status='READY', backend_version=response.json()['version'])
                    atomic_json(self.path, self.record)
                    return self
                except requests.RequestException:
                    time.sleep(0.5)
            raise RuntimeError('Bounded private service startup timed out')
        except BaseException as exc:
            self.__exit__(type(exc), exc, exc.__traceback__)
            raise

    def owned_process(self):
        process = psutil.Process(self.process.pid)
        if (abs(process.create_time() - self.record['creation_time']) > .01 or
                Path(process.exe()).resolve() != self.binary or process.cmdline() != self.record['argv']):
            raise RuntimeError('Owned backend process identity changed')
        return process

    def verify(self):
        validate_gpu_lease(self.gpu.lease_path)
        self.owned_process()
        if self.record['status'] != 'READY':
            raise RuntimeError('Owned backend is not ready')
        self.gpu.check()
        return self.record

    def client(self, original, *, num_ctx=None):
        identity = dict(original.identity)
        options = {k: identity[k] for k in ['revision','weight_sha256','quantization','chat_template_hash','timeout','seed','think','json_constraint','backend_runtime_config'] if k in identity}
        return CapturedClient(identity['model'], provider='ollama', base_url=self.base_url,
            backend_version=self.record['backend_version'], num_gpu=999,
            num_ctx=num_ctx or self.num_ctx, service_device_verified='gpu',
            prompt_token_counter=original.prompt_token_counter, **options)

    def __exit__(self, kind, value, traceback):
        failures = []
        try:
            if self.process is not None and self.process.poll() is None:
                self.owned_process()
                try:
                    response = self.session.get(self.base_url+'/api/ps', timeout=10)
                    response.raise_for_status()
                    for model in response.json().get('models', []):
                        response = self.session.post(self.base_url+'/api/generate',
                            json={'model': model['name'], 'keep_alive': 0}, timeout=30)
                        response.raise_for_status()
                    self.record['models_unloaded'] = True
                except Exception as exc:
                    failures.append('unload:' + type(exc).__name__)
                children = []
                for child in psutil.Process(self.process.pid).children(recursive=True):
                    try:
                        if Path(child.exe()).resolve().is_relative_to(self.binary.parent):
                            children.append((child.pid, child.create_time(), child.exe(), child.cmdline()))
                    except psutil.NoSuchProcess:
                        pass
                self.process.terminate()
                self.process.wait(timeout=30)
                for pid, created, exe, argv in children:
                    try:
                        child = psutil.Process(pid)
                        if (abs(child.create_time()-created)<.01 and child.exe()==exe and child.cmdline()==argv):
                            child.terminate()
                            child.wait(timeout=10)
                    except psutil.NoSuchProcess:
                        pass
            if hasattr(self, 'record'):
                self.record.update(status='EXITED', actual_backend_exit=self.process.poll() if self.process else None,
                    ended_unix=time.time(), shutdown_errors=failures, parent_exception=type(value).__name__ if value else None)
                atomic_json(self.path, self.record)
        finally:
            if hasattr(self, 'log'):
                self.log.close()
            self.session.close()
            self.gpu.__exit__(kind, value, traceback)


def counted_chat(run, service, client, *, request_id, stage, messages, max_tokens,
                 purpose, output_schema=None, temperature=0.0):
    """The reservation precedes every real request, even when the request fails."""
    budget = Budget(run)
    require_ram(2 * 1024**3)
    service.verify()
    budget.reserve(request_id=request_id, stage=stage, max_output_tokens=max_tokens,
                   purpose=purpose, model_identity=client.identity)
    client.last_metadata = {}
    client.raw_response = None
    try:
        started = time.monotonic()
        content = client.chat(messages, max_tokens=max_tokens, temperature=temperature,
                              output_schema=output_schema)
        elapsed = time.monotonic()-started
    except BaseException as exc:
        budget.finalize(request_id, output_tokens=client.last_metadata.get('eval_count'),
                        outcome='FAILED_' + type(exc).__name__)
        raise
    budget.finalize(request_id, output_tokens=client.last_metadata.get('eval_count'), outcome='BACKEND_RESPONSE')
    return {'content': content, 'backend_metadata': client.last_metadata,
            'raw_backend_response': client.raw_response, 'request_wall_seconds': elapsed,
            'model_identity': client.identity, 'request_id': request_id,
            'service_receipt': str(service.path), 'gpu_lease': service.gpu.receipt}
