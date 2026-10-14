"""Project-only Ollama lifecycle with a single GPU lease and D model storage."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json
import argparse
import os
import shutil
import signal
import socket
import subprocess
import time
import psutil
import requests
from filelock import FileLock
from ragscope.experiments.runtime import configure,runtime_home,validate_gpu_lease
from ragscope.indexing.identity import atomic_json,file_hash

def main(argv=None):
    configure(cpu_only=False); root=runtime_home(); run=root/'runs/20261014_v2'
    validate_gpu_lease(run/'gpu_lease.json')
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary',type=Path,help='Explicit private portable Ollama executable; preserves its adjacent libraries')
    args=parser.parse_args(argv)
    binary=str(args.binary.resolve()) if args.binary else shutil.which('ollama')
    if not binary: raise RuntimeError('Installed Ollama not found')
    if not Path(binary).is_file(): raise FileNotFoundError(binary)
    with socket.socket() as probe: probe.bind(('127.0.0.1',11435))
    environment=os.environ.copy()
    environment.update(CUDA_VISIBLE_DEVICES='0',OLLAMA_HOST='127.0.0.1:11435',OLLAMA_MODELS=str(root/'models/ollama'),
                       OLLAMA_NO_CLOUD='1',OLLAMA_MAX_LOADED_MODELS='1',OLLAMA_NUM_PARALLEL='1',OLLAMA_CONTEXT_LENGTH='4096',
                       OLLAMA_GPU_OVERHEAD=str(512*1024**2),OLLAMA_KEEP_ALIVE='5m',OLLAMA_DEBUG='1',
                       LLAMA_ARG_CACHE_RAM='0')
    stop=run/'generation.stop'
    if stop.exists(): raise RuntimeError('Previous stop marker must be explicitly archived before restart')
    session=requests.Session(); session.trust_env=False
    with FileLock(str(root/'runs/gpu.lock'),timeout=0), (root/'logs/generation_service.log').open('a',encoding='utf-8') as log:
        process=subprocess.Popen([binary,'serve'],stdout=log,stderr=subprocess.STDOUT,env=environment)
        record=dict(pid=process.pid,creation_time=psutil.Process(process.pid).create_time(),binary=binary,binary_sha256=file_hash(binary),
                    status='STARTING',base_url='http://127.0.0.1:11435',models_path=environment['OLLAMA_MODELS'],
                    environment={key:value for key,value in environment.items() if key.startswith(('OLLAMA_','LLAMA_ARG_')) or key=='CUDA_VISIBLE_DEVICES'})
        path=run/'generation_service.json'; atomic_json(path,record)
        def request_stop(*_): stop.write_text('Own service shutdown requested',encoding='utf-8')
        signal.signal(signal.SIGINT,request_stop); signal.signal(signal.SIGTERM,request_stop)
        low=0
        try:
            while process.poll() is None:
                low=low+1 if psutil.virtual_memory().available<512*1024**2 else 0
                if stop.exists() or low>=3:
                    record['stop_reason']='request' if stop.exists() else 'host_ram_below_512MiB'
                    break
                if record['status']=='STARTING':
                    try:
                        response=session.get(record['base_url']+'/api/version',timeout=2); response.raise_for_status()
                        record.update(status='READY',backend_version=response.json()['version']); atomic_json(path,record)
                    except requests.RequestException: pass
                time.sleep(1)
        finally:
            if process.poll() is None:
                try:
                    for model in session.get(record['base_url']+'/api/ps',timeout=10).json().get('models',[]):
                        session.post(record['base_url']+'/api/generate',json={'model':model['name'],'keep_alive':0},timeout=30).raise_for_status()
                    record['unloaded']=True
                except Exception as exc: record['unload_error']=str(exc)
                parent=psutil.Process(process.pid)
                children=parent.children(recursive=True)
                process.terminate(); process.wait(timeout=30)
                # Owned stateless runner children only, with captured creation identities.
                for child in children:
                    try:
                        if child.is_running(): child.terminate()
                    except psutil.NoSuchProcess: pass
            record.update(status='EXITED',exit_code=process.poll(),ended_at=time.time()); atomic_json(path,record)
    return process.returncode or 0

if __name__=='__main__': raise SystemExit(main())
