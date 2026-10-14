"""Current-process GPU ownership without stopping or restoring other jobs."""
from __future__ import annotations
import csv
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from filelock import FileLock
import psutil
from ragscope.experiments.runtime import validate_gpu_lease
from ragscope.indexing.identity import artifact, atomic_json


def gpu_observation(allowed_pids=()):
    result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name',
                             '--format=csv,noheader'], capture_output=True, text=True, check=True, timeout=20)
    apps = []
    for row in csv.reader(io.StringIO(result.stdout)):
        if not row:
            continue
        pid, name = int(row[0].strip()), row[1].strip()
        process = psutil.Process(pid)
        permitted = pid in allowed_pids or (process.name().casefold() in {'codex.exe', 'chatgpt.exe'}
            and 'WindowsApps' in process.exe() and 'OpenAI.Codex' in process.exe())
        apps.append({'pid': pid, 'name': name, 'exe': process.exe(),
                     'create_time': process.create_time(), 'permitted': permitted})
    if any(not app['permitted'] for app in apps):
        raise RuntimeError('Unknown GPU consumer retained; do not terminate it')
    device = subprocess.run(['nvidia-smi', '--query-gpu=index,name,memory.used,memory.total,utilization.gpu',
                             '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True, timeout=20)
    return {'compute_apps': apps, 'device_csv': device.stdout.strip(), 'observed_unix': time.time()}


class GpuSession:
    def __init__(self, run):
        self.run = Path(run)
        self.runtime = self.run.parents[1]
        self.lock = FileLock(str(self.runtime / 'runs/gpu.lock'), timeout=0)

    def __enter__(self):
        self.lock.acquire()
        try:
            observation = gpu_observation([os.getpid()])
            self.lease_path = self.run / 'gpu_leases' / f'{os.getpid()}_{time.time_ns()}.json'
            lease = {'status': 'ACQUIRED', 'owner': 'ragscope', 'safety_verified': True, 'gpu_device': 0,
                'host_platform': sys.platform, 'host_boot_time_unix': psutil.boot_time(), 'validated_at': time.time(),
                'pid': os.getpid(), 'process_creation_time': psutil.Process().create_time(),
                'argv': psutil.Process().cmdline(), 'restoration_policy': 'USER_MANUAL_RESUME',
                'baseline_observation': observation, 'unknown_processes_terminated': False,
                'gpu_guard_source': artifact(__file__)}
            atomic_json(self.lease_path, lease)
            validate_gpu_lease(self.lease_path)
            os.environ['CUDA_VISIBLE_DEVICES'] = '0'
            self.receipt = artifact(self.lease_path)
            return self
        except BaseException:
            self.lock.release()
            raise

    def check(self):
        validate_gpu_lease(self.lease_path)
        return gpu_observation([os.getpid()] + [p.pid for p in psutil.Process().children(recursive=True)])

    def __exit__(self, kind, value, traceback):
        try:
            atomic_json(self.lease_path.with_suffix('.release.json'), {'pid': os.getpid(),
                'original_lease': self.receipt, 'context_exited_unix': time.time(),
                'status': 'CONTEXT_EXITED_PROCESS_MUST_BE_OBSERVED_GONE',
                'model_memory_release': 'caller deletes model and empties own allocator; process exit is final proof',
                'original_training_policy': 'USER_MANUAL_RESUME', 'original_training_restarted': False,
                'exception': type(value).__name__ if value else None})
        finally:
            self.lock.release()
