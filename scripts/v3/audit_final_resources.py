"""Observe exact owned process identities, ports and locks; never terminate user workloads."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import psutil
PROJECT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(PROJECT))
from filelock import FileLock,Timeout
from ragscope.indexing.identity import artifact,atomic_json,canonical_json


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign',required=True);a=p.parse_args();run=Path(a.campaign).resolve()
    runtime=Path('D:/RAGScope_Runtime').resolve()
    if run.parent!=runtime/'runs':raise ValueError('Wrong campaign')
    observer=psutil.Process()
    observer_pids={observer.pid}
    # Windows venv python.exe may be a forwarding launcher for this same argv.
    # Exclude only a verified ancestor of this observer with exactly its command.
    for parent in observer.parents():
        try:
            if (parent.name().casefold()=='python.exe' and
                    Path(parent.exe()).resolve().is_relative_to(runtime/'envs') and
                    parent.cmdline()[1:]==observer.cmdline()[1:]):observer_pids.add(parent.pid)
        except (psutil.AccessDenied,psutil.NoSuchProcess):pass
    own=[];observation_errors=[]
    for process in psutil.process_iter(['pid','name','exe','cmdline','create_time']):
        try:
            info=process.info;argv=info['cmdline'] or []
            if info['pid'] in observer_pids:continue
            normalized=[s.replace('\\','/').casefold() for s in argv]
            if info['name'].casefold()=='python.exe' and any('/scripts/v3/' in s or s.startswith('scripts/v3/') or 'ui/chat_app_v3.py' in s for s in normalized):
                if any(str(run).replace('\\','/').casefold()==s for s in normalized) or any('ui/chat_app_v3.py' in s for s in normalized):own.append(info)
        except (psutil.AccessDenied,psutil.NoSuchProcess):continue
    services=[]
    for path in (run/'services').glob('*/service.json'):
        row=json.loads(path.read_text('utf8'));same_alive=False
        try:
            process=psutil.Process(row['pid'])
            same_alive=abs(process.create_time()-row['creation_time'])<.01 and process.cmdline()==row['argv'] and Path(process.exe()).resolve()==Path(row['binary']).resolve()
        except psutil.NoSuchProcess:pass
        except psutil.AccessDenied:observation_errors.append(str(path)+':identity observation denied')
        services.append({'receipt':artifact(path),'recorded_status':row['status'],'same_owned_process_alive':same_alive,
                         'models_unloaded':row.get('models_unloaded'),'shutdown_errors':row.get('shutdown_errors',[])})
    ports={}
    for port in (11435,11437,8502):
        with socket.socket() as probe:
            try:probe.bind(('127.0.0.1',port));ports[str(port)]='AVAILABLE'
            except OSError:ports[str(port)]='IN_USE_PRESERVE_PROCESS'
    locks={}
    for path in (runtime/'runs/gpu.lock',run/'writer.lock',run/'orchestrator.lock'):
        lock=FileLock(str(path),timeout=0)
        try:lock.acquire();locks[str(path)]='AVAILABLE';lock.release()
        except Timeout:locks[str(path)]='IN_USE'
    gpu=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],capture_output=True,text=True,timeout=20)
    device=subprocess.run(['nvidia-smi','--query-gpu=memory.used,memory.free,utilization.gpu','--format=csv,noheader'],capture_output=True,text=True,timeout=20)
    passed=not own and not observation_errors and all(not s['same_owned_process_alive'] for s in services) and all(v=='AVAILABLE' for v in [*ports.values(),*locks.values()])
    result={'status':'PASS_OWNED_RESOURCES_RELEASED' if passed else 'FAIL_OR_UNVERIFIED_OWNED_RESOURCE_RELEASE',
        'observed_utc':datetime.now(timezone.utc).isoformat(),'owned_live_workers':own,'service_observations':services,
        'observer_exact_pids_excluded':sorted(observer_pids),
        'ports':ports,'locks':locks,'observation_errors':observation_errors,
        'gpu_compute_apps_csv':gpu.stdout,'gpu_observation_exit':gpu.returncode,
        'gpu_device_csv':device.stdout,'gpu_device_observation_exit':device.returncode,
        'memory_available_bytes':psutil.virtual_memory().available,'original_training_policy':'USER_MANUAL_RESUME',
        'original_training_restarted':False,'scheduled_tasks_recreated':False,'user_processes_stopped':False,
        'other_processes_policy':'Observed only; graphics/app consumers may remain. No GPU-zero claim.',
        'wsl_restarted':False,'wsl_memory_cap_changed':False,'driver_security_changes':False,'runner':artifact(__file__)}
    atomic_json(run/'FINAL_RESOURCE_STATE.json',result)
    atomic_json(PROJECT/'docs/execution/v3/FINAL_RESOURCE_STATE.json',result)
    print(canonical_json({k:result[k] for k in ('status','ports','locks','owned_live_workers','memory_available_bytes')}));return 0 if passed else 3


if __name__=='__main__':raise SystemExit(main())
