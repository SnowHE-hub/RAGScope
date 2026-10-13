"""One conditional CPU load attempt; only an identified, result-free child may be terminated."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid
from urllib.request import urlopen

from ragscope.indexing.identity import artifact, atomic_json as _atomic_json, signature

GIB=1024**3
RESERVE=int(1.25*GIB)


def atomic_json(path, value):
    """Windows readers can briefly deny replacement; never turn this into an unbounded retry."""
    for attempt in range(5):
        try:
            return _atomic_json(path,value)
        except PermissionError:
            if attempt==4:raise
            time.sleep(.02*(2**attempt))


def read(path):
    return json.loads(Path(path).read_text('utf-8'))


def process_receipt(process):
    return {'pid':process.pid,'creation_time':process.create_time(),'command':process.cmdline(),'cwd':process.cwd()}


def matches_process(receipt, process):
    """Never infer ownership from PID or executable basename alone."""
    return (process.pid==receipt.get('pid') and abs(process.create_time()-receipt.get('creation_time',0))<.001
            and process.cmdline()==receipt.get('command') and Path(process.cwd()).resolve()==Path(receipt.get('cwd','')).resolve())


def other_owned_models(home, campaign, *, excluded_pids=()):
    """Observe only recognizable RAGScope model commands and its known service receipt."""
    import psutil
    home=Path(home).resolve()
    exclude=set(excluded_pids)|{os.getpid()}
    current=psutil.Process(os.getpid())
    # Native Windows venv Python has a same-command launcher parent. It is our
    # invocation, not a second resident model; unrelated ancestors stay visible.
    for ancestor in current.parents():
        try:
            if ancestor.cmdline()[1:]==current.cmdline()[1:] and ancestor.cwd()==current.cwd():
                exclude.add(ancestor.pid)
        except (psutil.NoSuchProcess,psutil.AccessDenied):pass
    blockers=[]
    names={'python.exe','pythonw.exe','python','python3','llama-server.exe','ollama.exe','ollama'}
    markers=('run_rerank_v2.py','run_laya_v2.py','ragscope.routing.laya_campaign',
             'ragscope.training.run','evaluate_lora_v2.py','run_online_routing_v2.py',
             'ragscope.experiments.legacy_reconstruction')
    project=str(Path.cwd().resolve()).replace('\\','/').lower()
    home_text=str(home).replace('\\','/').lower()
    for process in psutil.process_iter(['pid','name']):
        if process.pid in exclude or str(process.info['name']).lower() not in names:continue
        try:
            command=process.cmdline()
            text=' '.join(command).replace('\\','/').lower()
            cwd=str(Path(process.cwd()).resolve()).replace('\\','/').lower()
            owned=(home_text in text or cwd==project)
            is_model=any(marker in text for marker in markers) or ('llama-server' in text and home_text in text)
            if owned and is_model:blockers.append(process_receipt(process))
        except (psutil.NoSuchProcess,psutil.AccessDenied):
            continue
    service_path=home/'runs'/campaign/'generation_service.json'
    if service_path.exists():
        service=read(service_path)
        try:
            process=psutil.Process(service['pid'])
            if abs(process.create_time()-service.get('creation_time',0))<.001:
                # Only the pre-existing owned loopback endpoint; never download/start/unload.
                if service.get('base_url')!='http://127.0.0.1:11435':
                    raise ValueError('Unrecognized owned generation endpoint')
                with urlopen(service['base_url']+'/api/ps',timeout=2) as response:
                    resident=json.load(response).get('models')
                if resident!=[]:blockers.append({'service':artifact(service_path),'reason':'owned generation models resident or unverified'})
        except psutil.NoSuchProcess:
            pass
        except (KeyError,ValueError,OSError,psutil.AccessDenied) as error:
            blockers.append({'service':str(service_path),'reason':'Cannot verify generation service is empty','error':str(error)})
    return blockers


class ChildControl:
    """The child cannot load or warm up until its exact identity has been acknowledged."""
    def __init__(self, directory, home, campaign):
        import psutil
        self.directory=Path(directory).resolve()
        expected=Path(home).resolve()/'runs'/(campaign+'_laya')/'load_fallbacks'
        if self.directory.parent!=expected or len(self.directory.name)!=32:
            raise ValueError('Supervised directory outside the owned Laya run')
        self.request=read(self.directory/'request.json')
        if self.request['token']!=self.directory.name or self.request['campaign']!=campaign:
            raise ValueError('Supervised request identity differs')
        self.process=psutil.Process(os.getpid())
        parent=psutil.Process(self.request['supervisor']['pid'])
        if not matches_process(self.request['supervisor'],parent):raise ValueError('Supervisor identity changed')
        if parent.pid not in {p.pid for p in self.process.parents()}:
            raise ValueError('Supervisor is not an ancestor of this child')
        if self.process.cmdline()[1:]!=self.request['command'][1:]:raise ValueError('Child command differs')
        if Path(self.process.cwd()).resolve()!=Path(self.request['cwd']).resolve():raise ValueError('Child cwd differs')
        self.receipt=process_receipt(self.process)|{'token':self.request['token'],'request_hash':signature(self.request)}
        self.wait('READY','LOAD')

    def checkpoint(self, phase, **extra):
        import psutil
        atomic_json(self.directory/'child.json',self.receipt|{'phase':phase,'time':time.time(),**extra})
        parent=psutil.Process(self.request['supervisor']['pid'])
        if not matches_process(self.request['supervisor'],parent):raise MemoryError('Supervisor exited/changed')
        control=self.directory/'control.json'
        if control.exists():
            message=read(control)
            if message.get('token')!=self.request['token']:raise ValueError('Control identity changed')
            if message.get('command')=='STOP':raise MemoryError('Supervised resource stop requested')
        if psutil.virtual_memory().available<RESERVE:raise MemoryError('Laya supervised RAM reserve')

    def wait(self, phase, command, **extra):
        deadline=time.monotonic()+60
        while time.monotonic()<deadline:
            self.checkpoint(phase,**extra)
            path=self.directory/'control.json'
            if path.exists() and read(path).get('command')==command:return
            time.sleep(.1)
        raise MemoryError('Supervisor did not authorize next model phase within60s')

    def close(self):
        atomic_json(self.directory/'child.json',self.receipt|{'phase':'EXITING','time':time.time()})


def verified_child(directory, request):
    import psutil
    path=Path(directory)/'child.json'
    if not path.exists():return None,None
    receipt=read(path)
    if receipt.get('token')!=request['token'] or receipt.get('request_hash')!=signature(request):
        raise ValueError('Child receipt token/request changed')
    process=psutil.Process(receipt['pid'])
    if not matches_process(receipt,process):raise ValueError('Child process identity changed')
    if process.cmdline()[1:]!=request['command'][1:]:raise ValueError('Observed child argv differs')
    if request['supervisor']['pid'] not in {p.pid for p in process.parents()}:
        raise ValueError('Observed process is no longer the owned child')
    return receipt,process


def monitor_child(process, directory, request, *, available=None, load_timeout=180):
    """Hard stop is prohibited after warmup authorization, including any formal inference."""
    import psutil
    available=available or (lambda:psutil.virtual_memory().available)
    directory=Path(directory)
    stage='STARTING';warmup_authorized=False;load_authorized=False;formal_authorized=False
    started=time.monotonic();load_started=None;stopped=False;hard_stopped=False;peak=0;minimum=None
    def control(command):
        atomic_json(directory/'control.json',{'token':request['token'],'command':command,'time':time.time()})
    with (directory/'samples.jsonl').open('a',encoding='utf-8') as writer:
        while process.poll() is None:
            free=int(available());minimum=free if minimum is None else min(minimum,free)
            receipt,child=None,None
            try:receipt,child=verified_child(directory,request)
            except psutil.NoSuchProcess:pass
            if receipt:stage=receipt['phase']
            try:rss=child.memory_info().rss if child else 0
            except psutil.NoSuchProcess:child=None;rss=0
            peak=max(peak,rss)
            writer.write(json.dumps({'time':time.time(),'pid':child.pid if child else None,'phase':stage,
                                    'rss':rss,'host_available':free})+'\n');writer.flush()
            timed_out=(not load_authorized and time.monotonic()-started>60) or (
                load_authorized and not warmup_authorized and time.monotonic()-load_started>load_timeout)
            if free<RESERVE or timed_out:
                stopped=True;control('STOP')
                if child is not None and not warmup_authorized and stage in {'READY','LOADING','LOADED_WAITING_PERMISSION'}:
                    # No WARMUP command has ever been sent, so no result-producing call can race this stop.
                    fresh,owned=verified_child(directory,request)
                    if fresh['phase'] in {'READY','LOADING','LOADED_WAITING_PERMISSION'}:
                        atomic_json(directory/'TERMINATION.json',{'reason':'RAM reserve' if free<RESERVE else 'bounded load timeout',
                            'observed_identity':fresh,'available_bytes':free,'no_warmup_authorized':True,
                            'results_can_have_started':False,'time':time.time()})
                        owned.terminate();hard_stopped=True
            if not stopped and receipt:
                if stage=='READY' and not load_authorized:
                    if free<int(4.5*GIB):stopped=True;control('STOP')
                    else:control('LOAD');load_authorized=True;load_started=time.monotonic()
                elif stage=='LOADED_WAITING_PERMISSION' and not warmup_authorized:
                    control('WARMUP');warmup_authorized=True
                elif stage=='PREFLIGHT_PASSED' and not formal_authorized:
                    if receipt.get('warmup_count')!=3 or receipt.get('warmup_errors')!=0:
                        stopped=True;control('STOP')
                    else:control('CONTINUE');formal_authorized=True
            time.sleep(.25)
    code=process.wait()
    result={'exit_code':code,'stopped_for_resource_or_timeout':stopped,'hard_stopped_unsaved_loader':hard_stopped,
            'warmup_authorized':warmup_authorized,'formal_authorized':formal_authorized,'peak_child_rss':peak,
            'minimum_host_available':minimum,'elapsed_seconds':time.monotonic()-started,'final_phase':stage}
    atomic_json(directory/'MONITOR_RESULT.json',result)
    return result


def conditional_run(home, campaign):
    """Use >=5GiB unchanged; permit exactly one monitored 4.5–5GiB attempt."""
    import psutil
    from .laya_campaign import run_raw
    home=Path(home)
    folder=home/'runs'/(campaign+'_laya')
    summary_path=folder/'SUMMARY.json'
    if summary_path.exists() and read(summary_path).get('status')=='COMPLETE':
        return run_raw(home,campaign)  # Existing artifact validation, without a model load.
    free=psutil.virtual_memory().available
    if free<int(4.5*GIB):raise MemoryError('Laya requires >=4.5GiB even for its single supervised fallback')
    blockers=other_owned_models(home,campaign)
    if blockers:raise MemoryError('Other owned model processes are active; no Laya fallback attempt made')
    if free>=5*GIB:return run_raw(home,campaign)
    state_path=folder/'LOAD_FALLBACK_STATE.json'
    if state_path.exists():raise RuntimeError('The single conditional Laya loading attempt has already been used')
    token=uuid.uuid4().hex
    directory=folder/'load_fallbacks'/token;directory.mkdir(parents=True)
    command=[sys.executable,'-u','-m','ragscope.routing.laya_campaign','--home',str(home),'--campaign',campaign,
             '--run','--supervised-worker',str(directory)]
    request={'token':token,'campaign':campaign,'home':str(home),'command':command,'cwd':str(Path.cwd().resolve()),
             'supervisor':process_receipt(psutil.Process(os.getpid())),'available_at_request':free,
             'minimum_load_gib':4.5,'reserve_gib':1.25,'max_attempts':1,'model_or_algorithm_changes':False}
    atomic_json(directory/'request.json',request)
    state={'status':'RUNNING','attempts':1,'request':artifact(directory/'request.json'),
           'interpretation':'2.354GiB is tensor storage address space, not measured RSS lower bound'}
    # Exclusive creation means concurrent callers cannot each consume the one attempt.
    with state_path.open('x',encoding='utf-8') as writer:json.dump(state,writer)
    process=None
    try:
        with (directory/'child.log').open('w',encoding='utf-8') as log:
            process=subprocess.Popen(command,cwd=request['cwd'],stdout=log,stderr=subprocess.STDOUT,
                                     creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            result=monitor_child(process,directory,request)
    except Exception as error:
        # A malformed/unverifiable identity never authorizes a termination. The child checks
        # this stop before every next phase; a lost supervisor also prevents further calls.
        atomic_json(directory/'control.json',{'token':token,'command':'STOP','time':time.time()})
        state.update(status='FAILED_SUPERVISION',error={'type':type(error).__name__,'message':str(error)},
                     launcher_pid=process.pid if process else None,
                     launcher_exited=process is None or process.poll() is not None,
                     automatic_retry_permitted=False)
        atomic_json(state_path,state)
        raise
    state.update(status='COMPLETE' if result['exit_code']==0 and result['formal_authorized'] else 'FAILED_STOPPED',
                 result=result,monitor=artifact(directory/'MONITOR_RESULT.json'),log=artifact(directory/'child.log'))
    atomic_json(state_path,state)
    if state['status']!='COMPLETE':raise MemoryError('Single supervised Laya attempt stopped/failed; no retry or lower threshold')
    if read(summary_path).get('status')!='COMPLETE':raise ValueError('Supervised child did not complete its fixed raw population')
    return run_raw(home,campaign)  # Verify the complete child's immutable artifacts, no reload.
