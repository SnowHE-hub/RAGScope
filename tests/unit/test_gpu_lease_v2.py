"""CPU-only lease contracts; fixtures never authorize or load a real model."""
import json
import subprocess
import sys
import time
from pathlib import Path

import psutil
import pytest

from ragscope.experiments import runtime

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'scripts'))


@pytest.fixture
def lease(tmp_path, monkeypatch):
    monkeypatch.setattr(psutil, 'boot_time', lambda: 1000.0)
    monkeypatch.setattr(runtime.time, 'time', lambda: 2000.0)
    row = {'status': 'ACQUIRED', 'owner': 'ragscope', 'safety_verified': True, 'gpu_device': 0,
           'host_platform': sys.platform, 'host_boot_time_unix': 1000.0, 'validated_at': 1500.0}
    path = tmp_path/'gpu_lease.json'
    path.write_text(json.dumps(row), encoding='utf-8')
    return path, row


def test_valid_receipt_is_read_only_and_tolerates_two_seconds(lease):
    path, row = lease
    before = path.read_bytes()
    assert runtime.validate_gpu_lease(path) == row
    assert path.read_bytes() == before
    for offset in (-2, 2):
        path.write_text(json.dumps({**row, 'host_boot_time_unix': 1000+offset}))
        assert runtime.validate_gpu_lease(path)['host_boot_time_unix'] == 1000+offset


@pytest.mark.parametrize('field,value', [
    ('status','RELEASED'), ('owner','other'), ('safety_verified',1), ('gpu_device',None),
    ('host_platform','different-platform'), ('host_boot_time_unix',900), ('host_boot_time_unix',1002.01),
    ('host_boot_time_unix',float('nan')), ('host_boot_time_unix',float('inf')),
    ('host_boot_time_unix',True), ('host_boot_time_unix','1000'),
    ('host_boot_time_unix',10**400),
    ('validated_at',float('nan')), ('validated_at',float('-inf')), ('validated_at',False),
    ('validated_at','1500'), ('validated_at',999.99), ('validated_at',2000.01),
    ('validated_at',10**400),
])
def test_invalid_receipts_fail_closed(lease, field, value):
    path, row = lease
    path.write_text(json.dumps({**row,field:value}))
    with pytest.raises(ValueError):
        runtime.validate_gpu_lease(path)


@pytest.mark.parametrize('field',['host_platform','host_boot_time_unix','validated_at'])
def test_old_receipts_missing_boot_fields_are_never_backfilled(lease,field):
    path, row = lease
    del row[field]
    path.write_text(json.dumps(row))
    before=path.read_bytes()
    with pytest.raises(ValueError):runtime.validate_gpu_lease(path)
    assert path.read_bytes()==before


def test_training_and_http_ui_entry_points_reject_old_boot_before_process_access(lease,monkeypatch):
    from ragscope.training.run import verify_lease
    from run_generation_campaign import verify_owned_service
    from ragscope.generation.llm_client import GenerationError
    path,row=lease
    row['host_boot_time_unix']=900
    path.write_text(json.dumps(row))
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0')
    with pytest.raises(ValueError,match='different host boot'):verify_lease(path)
    def forbidden(*args,**kwargs):raise AssertionError('Stale lease reached process inspection')
    monkeypatch.setattr(psutil,'Process',forbidden)
    with pytest.raises(GenerationError,match='different host boot'):
        verify_owned_service(path.parent/'absent_service.json',path,{})


def test_service_rejects_old_boot_before_starting_any_child(lease,monkeypatch):
    import serve_generation
    path,row=lease
    row.pop('host_boot_time_unix');path.write_text(json.dumps(row))
    home=path.parent
    owned=home/'runs/20261014_v2';owned.mkdir(parents=True)
    (owned/'gpu_lease.json').write_bytes(path.read_bytes())
    monkeypatch.setattr(serve_generation,'runtime_home',lambda:home)
    monkeypatch.setattr(serve_generation,'configure',lambda **kwargs:None)
    def forbidden(*args,**kwargs):raise AssertionError('Stale lease reached child startup')
    monkeypatch.setattr(serve_generation.subprocess,'Popen',forbidden)
    with pytest.raises(ValueError,match='host_boot_time_unix'):serve_generation.main()


def test_cli_uses_same_guard_and_outputs_only_device_on_success(tmp_path):
    path=tmp_path/'lease.json'
    row={'status':'ACQUIRED','owner':'ragscope','safety_verified':True,'gpu_device':0,
         'host_platform':sys.platform,'host_boot_time_unix':psutil.boot_time(),'validated_at':time.time()}
    path.write_text(json.dumps(row))
    command=[sys.executable,str(ROOT/'scripts/validate_gpu_lease.py'),'--lease',str(path),'--print-device']
    success=subprocess.run(command,capture_output=True,text=True,timeout=10)
    assert success.returncode==0 and success.stdout.strip()=='0'
    row.pop('validated_at');path.write_text(json.dumps(row));before=path.read_bytes()
    denied=subprocess.run(command,capture_output=True,text=True,timeout=10)
    assert denied.returncode==2 and not denied.stdout and 'finite numeric validated_at' in denied.stderr
    assert path.read_bytes()==before


def test_pilot_rejects_old_lease_before_cuda_enable_or_tokenizer_load(lease,monkeypatch):
    import runpy
    from ragscope.generation import token_budget
    path,row=lease
    row.pop('host_boot_time_unix')
    home=path.parent
    owned=home/'runs/20261014_v2';owned.mkdir(parents=True)
    (owned/'gpu_lease.json').write_text(json.dumps(row))
    monkeypatch.setattr(runtime,'runtime_home',lambda:home)
    monkeypatch.setattr(runtime,'configure',lambda cpu_only:monkeypatch.setenv('CUDA_VISIBLE_DEVICES','' if cpu_only else '0'))
    monkeypatch.setattr(sys,'argv',['pilot_generation.py','--size','2b'])
    def forbidden(*args,**kwargs):raise AssertionError('Old pilot lease reached tokenizer loading')
    monkeypatch.setattr(token_budget,'Qwen35PromptCounter',forbidden)
    with pytest.raises(ValueError,match='host_boot_time_unix'):
        runpy.run_path(str(ROOT/'scripts/pilot_generation.py'),run_name='__main__')
    import os
    assert os.environ['CUDA_VISIBLE_DEVICES']==''


def test_explicit_portable_binary_preserves_own_service_environment(tmp_path,monkeypatch):
    import serve_generation
    home=tmp_path/'runtime'; (home/'runs/20261014_v2').mkdir(parents=True); (home/'logs').mkdir()
    binary=home/'tools/ollama-0.34.4/ollama.exe'; binary.parent.mkdir(parents=True); binary.write_bytes(b'unit fixture only')
    monkeypatch.setattr(serve_generation,'runtime_home',lambda:home)
    monkeypatch.setattr(serve_generation,'configure',lambda **kwargs:None)
    monkeypatch.setattr(serve_generation,'validate_gpu_lease',lambda path:{})
    def forbidden_lookup(*args):raise AssertionError('Explicit binary fell back to system PATH')
    monkeypatch.setattr(serve_generation.shutil,'which',forbidden_lookup)
    class Probe:
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def bind(self,address):assert address==('127.0.0.1',11435)
    monkeypatch.setattr(serve_generation.socket,'socket',Probe)
    class CapturedStartup(Exception):pass
    def capture(command,**kwargs):
        assert command==[str(binary.resolve()),'serve']
        env=kwargs['env']
        assert env['OLLAMA_MODELS']==str(home/'models/ollama')
        assert env['LLAMA_ARG_CACHE_RAM']=='0'
        assert env['OLLAMA_MAX_LOADED_MODELS']==env['OLLAMA_NUM_PARALLEL']=='1'
        raise CapturedStartup
    monkeypatch.setattr(serve_generation.subprocess,'Popen',capture)
    with pytest.raises(CapturedStartup):serve_generation.main(['--binary',str(binary)])


def test_missing_explicit_binary_never_falls_back_or_opens_port(tmp_path,monkeypatch):
    import serve_generation
    monkeypatch.setattr(serve_generation,'configure',lambda **kwargs:None)
    monkeypatch.setattr(serve_generation,'validate_gpu_lease',lambda path:{})
    def forbidden(*args,**kwargs):raise AssertionError('Missing portable binary reached socket or PATH')
    monkeypatch.setattr(serve_generation.shutil,'which',forbidden)
    monkeypatch.setattr(serve_generation.socket,'socket',forbidden)
    with pytest.raises(FileNotFoundError):serve_generation.main(['--binary',str(tmp_path/'absent.exe')])
