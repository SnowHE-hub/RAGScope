"""Only tiny, model-free child processes; never evidence of actual Laya feasibility."""
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import uuid

import psutil
import pytest

from ragscope.indexing.identity import atomic_json
from ragscope.routing import laya_supervision as module


def launch_fixture(tmp_path, mode):
    script=tmp_path/'owned_fixture.py'
    script.write_text('''
import sys,time,json
from pathlib import Path
from types import SimpleNamespace
import psutil
from ragscope.routing.laya_supervision import ChildControl,GIB
psutil.virtual_memory=lambda:SimpleNamespace(available=5*GIB)
control=ChildControl(sys.argv[1],sys.argv[2],'fixture')
control.checkpoint('LOADING')
if sys.argv[3]=='loading':time.sleep(15)
control.wait('LOADED_WAITING_PERMISSION','WARMUP')
control.checkpoint('WARMUP_RUNNING')
time.sleep(.6)
(Path(sys.argv[1])/'fixture_result.json').write_text(json.dumps({'offline_fixture':True,'saved':True}))
control.checkpoint('WARMUP_COMMITTED')
control.wait('PREFLIGHT_PASSED','CONTINUE',warmup_count=3,warmup_errors=0)
control.checkpoint('FORMAL_COMMITTED')
control.close()
''',encoding='utf-8')
    token=uuid.uuid4().hex
    directory=tmp_path/'runs/fixture_laya/load_fallbacks'/token;directory.mkdir(parents=True)
    command=[sys.executable,'-u',str(script),str(directory),str(tmp_path),mode]
    request={'token':token,'campaign':'fixture','command':command,'cwd':str(Path.cwd().resolve()),
             'supervisor':module.process_receipt(psutil.Process(os.getpid()))}
    atomic_json(directory/'request.json',request)
    log=(directory/'child.log').open('w',encoding='utf-8')
    env=dict(os.environ)
    env['PYTHONPATH']=str(Path.cwd().resolve())+(os.pathsep+env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    process=subprocess.Popen(command,cwd=Path.cwd(),stdout=log,stderr=subprocess.STDOUT,
                             env=env,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    return process,directory,request,log


def fixture_phase(directory):
    path=directory/'child.json'
    return module.read(path).get('phase') if path.exists() else 'STARTING'


def test_supervised_protocol_runs_only_after_two_explicit_permissions(tmp_path):
    process,directory,request,log=launch_fixture(tmp_path,'success')
    try:result=module.monitor_child(process,directory,request,available=lambda:int(4.75*module.GIB))
    finally:log.close()
    assert result['exit_code']==0,result|{'log':(directory/'child.log').read_text('utf-8')}
    assert result['warmup_authorized'] and result['formal_authorized']
    assert not result['hard_stopped_unsaved_loader']
    assert module.read(directory/'fixture_result.json')['saved']


def test_low_ram_loading_stops_only_exact_owned_result_free_process(tmp_path):
    process,directory,request,log=launch_fixture(tmp_path,'loading')
    def available():
        return module.RESERVE-1 if fixture_phase(directory)=='LOADING' else int(4.75*module.GIB)
    try:result=module.monitor_child(process,directory,request,available=available)
    finally:log.close()
    assert result['exit_code']!=0
    assert result['hard_stopped_unsaved_loader'] and not result['warmup_authorized']
    receipt=module.read(directory/'TERMINATION.json')
    assert receipt['results_can_have_started'] is False
    assert not (directory/'fixture_result.json').exists()
    assert not psutil.pid_exists(receipt['observed_identity']['pid'])


def test_low_ram_after_warmup_permission_waits_for_durable_result(tmp_path):
    process,directory,request,log=launch_fixture(tmp_path,'warmup')
    def available():
        return module.RESERVE-1 if fixture_phase(directory)=='WARMUP_RUNNING' else int(4.75*module.GIB)
    try:result=module.monitor_child(process,directory,request,available=available)
    finally:log.close()
    assert result['exit_code']!=0 and result['stopped_for_resource_or_timeout']
    assert result['warmup_authorized'] and not result['formal_authorized']
    assert not result['hard_stopped_unsaved_loader'] and not (directory/'TERMINATION.json').exists()
    assert module.read(directory/'fixture_result.json')['saved']


def test_changed_pid_creation_time_and_command_do_not_establish_ownership():
    p=psutil.Process(os.getpid());receipt=module.process_receipt(p)
    assert module.matches_process(receipt,p)
    assert not module.matches_process(receipt|{'creation_time':receipt['creation_time']-1},p)
    assert not module.matches_process(receipt|{'command':['unrelated']},p)


@pytest.mark.parametrize('failures',[2,5])
def test_windows_atomic_state_replacement_has_bounded_retry(tmp_path,monkeypatch,failures):
    calls=[];delays=[]
    def write(path,value):
        calls.append((path,value))
        if len(calls)<=failures:raise PermissionError('Windows concurrent reader')
        atomic_json(path,value)
    monkeypatch.setattr(module,'_atomic_json',write)
    monkeypatch.setattr(module.time,'sleep',delays.append)
    path=tmp_path/'control.json'
    if failures==5:
        with pytest.raises(PermissionError):module.atomic_json(path,{'command':'STOP'})
        assert len(calls)==5 and not path.exists()
    else:
        module.atomic_json(path,{'command':'STOP'})
        assert len(calls)==3 and module.read(path)['command']=='STOP'
    assert sum(delays)<=.3


def test_own_windows_launcher_is_not_another_model(tmp_path,monkeypatch):
    class Process:
        def __init__(self,pid,command):
            self.pid=pid;self.command=command;self.info={'pid':pid,'name':'python.exe'}
        def cmdline(self):return self.command
        def cwd(self):return str(Path.cwd())
        def parents(self):return [launcher]
        def create_time(self):return 1
    current=Process(90,['base/python.exe','scripts/run_laya_v2.py','--run'])
    launcher=Process(89,['env/Scripts/python.exe','scripts/run_laya_v2.py','--run'])
    other=Process(88,['env/Scripts/python.exe','scripts/run_rerank_v2.py','--run'])
    monkeypatch.setattr(module.os,'getpid',lambda:90)
    monkeypatch.setattr(psutil,'Process',lambda pid:current)
    monkeypatch.setattr(psutil,'process_iter',lambda _:iter([current,launcher,other]))
    assert [x['pid'] for x in module.other_owned_models(tmp_path,'fixture')]==[88]


def test_forged_receipt_is_rejected_before_any_process_action(tmp_path):
    request={'token':'actual','command':sys.argv,'supervisor':module.process_receipt(psutil.Process())}
    atomic_json(tmp_path/'child.json',{'token':'forged','pid':os.getpid()})
    with pytest.raises(ValueError,match='token'):module.verified_child(tmp_path,request)


def test_parent_scope_path_is_enforced_before_any_model_import(tmp_path):
    before='torch' in sys.modules
    with pytest.raises(ValueError,match='outside'):module.ChildControl(tmp_path,tmp_path,'fixture')
    assert ('torch' in sys.modules)==before


@pytest.mark.parametrize('gib',[0,4.49])
def test_below_conditional_threshold_never_consumes_attempt(tmp_path,monkeypatch,gib):
    monkeypatch.setattr(psutil,'virtual_memory',lambda:SimpleNamespace(available=int(gib*module.GIB)))
    with pytest.raises(MemoryError,match='>=4.5'):module.conditional_run(tmp_path,'fixture')
    assert not (tmp_path/'runs/fixture_laya/LOAD_FALLBACK_STATE.json').exists()


def test_original_five_gib_path_remains_unchanged(tmp_path,monkeypatch):
    from ragscope.routing import laya_campaign
    monkeypatch.setattr(psutil,'virtual_memory',lambda:SimpleNamespace(available=5*module.GIB))
    monkeypatch.setattr(module,'other_owned_models',lambda *_:[])
    calls=[]
    monkeypatch.setattr(laya_campaign,'run_raw',lambda *args: calls.append(args) or {'offline_fixture':True})
    assert module.conditional_run(tmp_path,'fixture')['offline_fixture']
    assert calls==[(tmp_path,'fixture')]


def test_other_models_and_used_attempt_refuse_fallback(tmp_path,monkeypatch):
    monkeypatch.setattr(psutil,'virtual_memory',lambda:SimpleNamespace(available=int(4.75*module.GIB)))
    monkeypatch.setattr(module,'other_owned_models',lambda *_:[{'pid':123,'offline_fixture':True}])
    with pytest.raises(MemoryError,match='Other owned'):module.conditional_run(tmp_path,'fixture')
    monkeypatch.setattr(module,'other_owned_models',lambda *_:[])
    atomic_json(tmp_path/'runs/fixture_laya/LOAD_FALLBACK_STATE.json',{'attempts':1,'status':'FAILED_STOPPED'})
    with pytest.raises(RuntimeError,match='already been used'):module.conditional_run(tmp_path,'fixture')
