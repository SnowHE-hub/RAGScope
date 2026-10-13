"""Bounded, identity-checked interruption/restoration of the two audited research queues.

Run in Ubuntu via --exec python3. No process-group or force kill is used.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import struct
import subprocess
import time

ROOT = Path('/mnt/d/RAGScope_Runtime/runs/20261014_v2')
REPO = Path('/home/lenovo/projects/qm-t030')
DATA = Path('/home/lenovo/projects/quantmind/data/gpu_v1')
LEASE = ROOT / 'gpu_lease.json'
PYTHON = '/home/lenovo/miniforge3/envs/qm-gpu/bin/python'

def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    with temporary.open('w') as f:
        json.dump(value, f, indent=2); f.flush(); os.fsync(f.fileno())
    os.replace(temporary, path)

def artifact(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''): h.update(block)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=h.hexdigest())

def proc(pid):
    p = Path('/proc') / str(pid)
    try:
        stat = p.joinpath('stat').read_text().rsplit(')', 1)[1].split()
        argv = p.joinpath('cmdline').read_bytes().decode().strip('\0').split('\0')
        return dict(pid=int(pid), start_ticks=stat[19], state=stat[0], argv=argv,
                    cwd=os.readlink(p/'cwd'), stdout=os.readlink(p/'fd/1'),
                    children=[int(x) for x in (p/f'task/{pid}/children').read_text().split()])
    except (FileNotFoundError, ProcessLookupError): return None

def candidates():
    result = []
    for p in Path('/proc').iterdir():
        if not p.name.isdigit(): continue
        try:
            argv = p.joinpath('cmdline').read_bytes().decode().strip('\0').split('\0')
            if len(argv)>1 and argv[1]=='docs/work/evidence/T-030/run_queue.sh':
                result.append(proc(p.name))
        except (OSError, UnicodeError): pass
    return result

def validate_checkpoint(child):
    argv = child['argv']
    if argv[:2] != [PYTHON, 'docs/work/evidence/T-030/run_trees.py']:
        raise ValueError('Only the audited tree-training entry point is eligible')
    if len(argv)!=5 or argv[2]!='/home/lenovo/projects/quantmind' or argv[3] not in ('xgb','lgbm') or argv[4] not in ('20d','63d'):
        raise ValueError('Unexpected research task arguments')
    tag = '_'.join(argv[3:]); run = DATA/'runs'/tag
    records, assets = [], []
    for path in sorted((run/'ledger').glob('*.json')):
        row = json.loads(path.read_text())
        records.append(dict(name=path.name, **artifact(path)))
        match = re.fullmatch(r'k(\d+)_final_s(\d+)\.json',path.name)
        if match:
            stem=f'k{int(match[1]):02d}_s{match[2]}'
            prediction=run/(stem+'.npy')
            with prediction.open('rb') as f:
                if f.read(6)!=b'\x93NUMPY': raise ValueError('Invalid prediction header')
                version=f.read(2); n=struct.unpack('<H' if version[0]==1 else '<I', f.read(2 if version[0]==1 else 4))[0]
                header=ast.literal_eval(f.read(n).decode().strip())
                if header['descr'] not in ('<f4','<f8') or len(header['shape'])!=1: raise ValueError('Unexpected prediction dtype/shape')
                expected=f.tell()+header['shape'][0]*(4 if header['descr']=='<f4' else 8)
                if expected!=prediction.stat().st_size: raise ValueError('Incomplete prediction payload')
            assets.append(artifact(prediction))
            if argv[3]=='xgb':
                model=run/(stem+'.ubj')
                with model.open('rb') as f:
                    if f.read(1)!=b'{': raise ValueError('Invalid UBJSON model header')
                    f.seek(-1,2)
                    if f.read(1)!=b'}': raise ValueError('Incomplete UBJSON model')
                assets.append(artifact(model))
    for path in sorted((DATA/'preds'/tag).glob('fold_*.parquet')):
        with path.open('rb') as f:
            if f.read(4)!=b'PAR1': raise ValueError('Invalid parquet header')
            f.seek(-4,2)
            if f.read()!=b'PAR1': raise ValueError('Invalid parquet footer')
        assets.append(artifact(path))
    if not records: raise ValueError('No durable completed training units')
    return dict(tag=tag, ledger=records, artifacts=assets,
                semantics='Completed inner configurations/seeds/folds resume; unfinished unit recomputes. No mid-unit checkpoint claimed.')

def audit():
    previous=None
    if LEASE.exists():
        previous=json.loads(LEASE.read_text())
        history=ROOT/'lease_history'/f"lease_{time.time_ns()}.json"
        save(history,previous)
    result=[]
    for parent in candidates():
        if parent['cwd']!=str(REPO) or len(parent['children'])!=1: raise ValueError('Unexpected queue ownership/tree')
        child=proc(parent['children'][0])
        if child is None or child['children']: raise ValueError('Unexpected child process tree')
        checkpoint=validate_checkpoint(child)
        queue=Path(parent['argv'][2]); queue=queue if queue.is_absolute() else REPO/queue
        logdir=Path(parent['argv'][3]); logdir=logdir if logdir.is_absolute() else REPO/logdir
        result.append(dict(parent=parent, child=child, checkpoint=checkpoint, queue=artifact(queue),
                           queue_script=artifact(REPO/'docs/work/evidence/T-030/run_queue.sh'),
                           resume_command=['bash', 'docs/work/evidence/T-030/run_queue.sh', str(queue), str(logdir)],
                           resume_cwd=str(REPO), resume_stdout=str(logdir/'ragscope_resume.out'),
                           log_before=artifact(Path(child['stdout']))))
    if not result: raise ValueError('No matching original queues; do not infer that restoration is needed')
    record=dict(status='AUDITED', owner='ragscope', gpu_device=0, safety_verified=True,
                creation_time=time.time(), original_task_state='RUNNING', jobs=result,
                authorization='User explicitly permits interruption of CPU/GPU processes and requests restoration.',
                stop_method='Identity-check parent SIGTERM to prevent queue advance, then child SIGINT; never SIGKILL. Verify all checkpoint hashes after exit.')
    if previous: record['previous_lease_history']=str(history)
    save(LEASE,record)
    print(json.dumps(dict(status=record['status'], jobs=[dict(pid=j['child']['pid'],tag=j['checkpoint']['tag'],units=len(j['checkpoint']['ledger'])) for j in result])))

def stop():
    record=json.loads(LEASE.read_text())
    if record['status']!='AUDITED': raise ValueError('Stop requires a fresh audit')
    for job in record['jobs']:
        for kind in ('parent','child'):
            expected=job[kind]; actual=proc(expected['pid'])
            if not actual or actual['start_ticks']!=expected['start_ticks'] or actual['argv']!=expected['argv']: raise ValueError('PID identity changed; nothing stopped')
    record['status']='STOPPING'; save(LEASE,record)
    for job in record['jobs']:
        os.kill(job['parent']['pid'],signal.SIGTERM)
        os.kill(job['child']['pid'],signal.SIGINT)
        job['signals_sent_at']=time.time(); save(LEASE,record)
    deadline=time.monotonic()+55
    while time.monotonic()<deadline:
        alive=[p for j in record['jobs'] for k in ('parent','child') if (p:=proc(j[k]['pid'])) and p['state']!='Z']
        if not alive: break
        time.sleep(1)
    if alive:
        record['status']='STOP_PENDING'; record['remaining_pids']=[p['pid'] for p in alive]; save(LEASE,record)
        raise RuntimeError('Graceful interrupt pending; no escalation performed')
    for job in record['jobs']:
        for a in job['checkpoint']['ledger']+job['checkpoint']['artifacts']:
            if artifact(Path(a['path']))!={k:a[k] for k in ('path','bytes','sha256')}: raise ValueError('Previously completed checkpoint changed')
        job['stopped']=True; job['stopped_at']=time.time()
    record.update(status='ACQUIRED', original_task_state='STOPPED', interrupted_at=time.time())
    save(LEASE,record); print('ACQUIRED: audited queues stopped; all prior checkpoint hashes preserved')

def restore():
    record=json.loads(LEASE.read_text())
    if record.get('restoration_policy')=='USER_MANUAL_RESUME':
        raise ValueError('Latest user instruction reserves restoration for the user; release own resources and notify only')
    if record['status']=='RESTORED': print('Already restored; no duplicate created'); return
    if record['status']!='ACQUIRED': raise ValueError('Only an acquired lease can restore tasks')
    active=candidates()
    for job in record['jobs']:
        if any(p['argv'][2:]==job['resume_command'][2:] for p in active): raise ValueError('Matching queue already running; manual reconciliation required')
        for key in ('queue','queue_script'):
            if artifact(Path(job[key]['path']))!=job[key]: raise ValueError('Resume script/queue changed')
        if not job.get('stopped'): raise ValueError('Task not stopped by this lease')
    for job in record['jobs']:
        environment=os.environ.copy(); environment.update(PY=PYTHON,PYTHONPATH='.',WRITE_MODE='parquet_only')
        with open(job['resume_stdout'],'ab') as output:
            process=subprocess.Popen(job['resume_command'],cwd=job['resume_cwd'],env=environment,stdin=subprocess.DEVNULL,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
        job['restored_pid']=process.pid; job['restored_at']=time.time(); save(LEASE,record)
    record['status']='RESTORED_PENDING_PROGRESS'; save(LEASE,record)
    print('Queues resumed; verify new log progress before marking RESTORED')

def settle():
    """Finish an earlier bounded SIGINT wait, without issuing another signal."""
    record=json.loads(LEASE.read_text())
    if record['status'] not in ('STOP_PENDING','STOPPING'): raise ValueError('No pending stop')
    alive=[]
    for job in record['jobs']:
        for kind in ('parent','child'):
            actual=proc(job[kind]['pid'])
            if actual and actual['state']!='Z' and actual['start_ticks']==job[kind]['start_ticks']: alive.append(actual['pid'])
        if not any(pid in alive for pid in (job['parent']['pid'],job['child']['pid'])):
            for a in job['checkpoint']['ledger']+job['checkpoint']['artifacts']:
                if artifact(Path(a['path']))!={k:a[k] for k in ('path','bytes','sha256')}: raise ValueError('Checkpoint hash changed: '+a['path'])
            job['stopped']=True; job.setdefault('stopped_at',time.time())
    record['remaining_pids']=alive
    if not alive: record.update(status='ACQUIRED', original_task_state='STOPPED', interrupted_at=time.time())
    save(LEASE,record); print(json.dumps({'status':record['status'],'remaining_pids':alive}))

def terminate_pending():
    """User-authorized SIGTERM only for the audited child that ignores SIGINT.

    Completed units are durable; the current unfinished unit is recomputed.
    This is not a mid-unit graceful save and is explicitly recorded as such.
    """
    record=json.loads(LEASE.read_text())
    if record['status']!='STOP_PENDING': raise ValueError('Expected previous bounded SIGINT attempt')
    for job in record['jobs']:
        if job.get('stopped'): continue
        expected=job['child']; actual=proc(expected['pid'])
        if not actual or actual['start_ticks']!=expected['start_ticks'] or actual['argv']!=expected['argv']:
            raise ValueError('Pending child identity changed')
        status=(Path('/proc')/str(actual['pid'])/'status').read_text()
        ignored=int(next(x.split()[1] for x in status.splitlines() if x.startswith('SigIgn:')),16)
        if not ignored & (1 << (signal.SIGINT-1)): raise ValueError('Child does not ignore SIGINT; allow graceful return')
        for a in job['checkpoint']['ledger']+job['checkpoint']['artifacts']:
            if artifact(Path(a['path']))!={k:a[k] for k in ('path','bytes','sha256')}: raise ValueError('Checkpoint changed')
        job['termination_reason']='SIGINT disposition is ignored (observed /proc); explicit user termination authorization. Completed unit hashes verified. Unfinished unit restarts.'
        job['sigterm_at']=time.time(); save(LEASE,record)
        os.kill(actual['pid'],signal.SIGTERM)
    time.sleep(2)
    settle()

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('action', choices=['audit','stop','settle','terminate_pending','restore'])
    globals()[parser.parse_args().action]()
