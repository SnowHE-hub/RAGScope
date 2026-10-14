"""Offline migration/resource contracts only; no torch import or device allocation."""
import json
import sys
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest

from ragscope.experiments import rerank_gpu as gpu
from ragscope.experiments.ledger import RecordStore
from ragscope.indexing.identity import artifact, atomic_json
from ragscope.experiments.rerank_validation import validate_gpu_memory
from ragscope.retrieval.verified_reranker import CudaPairPredictor


@pytest.fixture
def migration(tmp_path, monkeypatch):
    run = tmp_path/'runs/demo_retrieval'
    jobs = []
    for prefix, n in [('E1',100),('TEST_CONTEXT',200),('E8',100),('ROUTE_CONTEXT_TRAIN',200),('ROUTE_CONTEXT_DEV',60)]:
        source=run/'records'/f'{prefix}_R512_hybrid_neighbor.jsonl'
        source.parent.mkdir(parents=True,exist_ok=True)
        source.write_text('offline source\n')
        atomic_json(source.with_suffix('.config.json'),{'offline_fixture':True})
        destination=source.with_stem(source.stem.replace('_hybrid_','_hybrid_rerank_'))
        job={'prefix':prefix,'source':source,'destination':destination,'queries':[{'query_id':f'{prefix}-{i}'} for i in range(n)],'provenance':[]}
        jobs.append(job)
        if prefix in ('E1','TEST_CONTEXT','E8'):
            count=29 if prefix=='E8' else n
            store=RecordStore(destination,{'offline_fixture':True})
            for i in range(count):store.add(str(i),{'device':'cpu','error':None,'offline_fixture':True})
            if prefix!='E8':
                atomic_json(run/f'R3_GROUP_{prefix}_VALIDATION.json',{'status':'PASS','n_total':count,'n_failed':0,'offline_fixture':True})
                atomic_json(run/'summaries'/f'{destination.stem}.json',{'offline_fixture':True})
    atomic_json(run/'R3_RUNTIME/LOAD_STATE.json',{'attempts':2,'backend_failures':0})
    atomic_json(run/'R3_RUNTIME/RESOURCE_PAUSE_PID30108.json',{'offline_fixture':True})
    cp=tmp_path/'checkpoint.json'; atomic_json(cp,{'offline_fixture':True})
    monkeypatch.setattr(gpu,'frozen_jobs',lambda *_:jobs)
    monkeypatch.setattr(gpu,'verified_source',lambda job:({},job['queries']))
    import ragscope.retrieval.verified_reranker as loader
    monkeypatch.setattr(loader,'verified_checkpoint',lambda _: (tmp_path,artifact(cp)))
    monkeypatch.setattr(gpu.importlib.metadata,'version',lambda n:{'torch':'2.9.1+cu128','transformers':'5.19.0','numpy':'2.5.4'}[n])
    return tmp_path,run,jobs


def test_gpu_jobs_keep_full_populations_and_disjoint_paths(migration):
    home,_,old=migration
    jobs=gpu.gpu_jobs(home,'demo')
    assert [len(j['queries']) for j in jobs]==[100,200,60]
    assert all(j['destination'].stem.endswith('_gpu') for j in jobs)
    assert not set(j['destination'] for j in jobs)&set(j['destination'] for j in old)
    assert jobs[0]['queries']==old[2]['queries']


def test_prepare_preserves329_and_is_idempotent_without_torch(migration):
    home,run,jobs=migration
    before={j['destination']:j['destination'].read_bytes() for j in jobs[:3]}
    imported='torch' in sys.modules
    value=gpu.prepare_migration(home,'demo')
    assert value['expected_gpu_rows']==360 and value['historical_extra_partial_calls']==29
    assert sum(g['n_saved'] for g in value['cpu_groups_preserved'])==329
    assert gpu.prepare_migration(home,'demo')==value
    assert all(p.read_bytes()==data for p,data in before.items())
    assert ('torch' in sys.modules)==imported


@pytest.mark.parametrize('target',['cpu_record','source','load_state','migration'])
def test_protected_inputs_cannot_change(migration,target):
    home,run,jobs=migration
    gpu.prepare_migration(home,'demo')
    path={'cpu_record':jobs[0]['destination'],'source':jobs[2]['source'],
          'load_state':run/'R3_RUNTIME/LOAD_STATE.json','migration':gpu.migration_path(home,'demo')}[target]
    if target=='migration':
        value=json.loads(path.read_text());value['dtype']='torch.float16';atomic_json(path,value)
    else:path.write_text('tampered fixture')
    with pytest.raises(ValueError):gpu.verify_migration(home,'demo')


def test_live_generation_service_rejected_before_model_import(tmp_path,monkeypatch):
    atomic_json(tmp_path/'runs/demo/generation_service.json',{'pid':123,'creation_time':456.0})
    import psutil
    monkeypatch.setattr(psutil,'Process',lambda _:SimpleNamespace(create_time=lambda:456.0))
    imported='torch' in sys.modules
    with pytest.raises(RuntimeError,match='actually exit'):gpu.require_service_exit(tmp_path,'demo')
    assert ('torch' in sys.modules)==imported


def test_resident_compute_pid_rejected(monkeypatch):
    monkeypatch.setattr(gpu.subprocess,'run',lambda a,**k:SimpleNamespace(stdout='12345\n' if 'compute-apps' in a[1] else '0,12,8000,8192\n'))
    with pytest.raises(RuntimeError,match='resident'):gpu.require_empty_gpu()


def test_unrecognized_gpu_observation_is_not_treated_as_empty(monkeypatch):
    monkeypatch.setattr(gpu.subprocess,'run',lambda a,**k:SimpleNamespace(
        stdout='unavailable' if 'compute-apps' in a[1] else '0,12,8000,8192\n'))
    with pytest.raises(RuntimeError,match='Unrecognized'):gpu.require_empty_gpu()


@pytest.mark.parametrize('changed',['none','pid','created','exe','subtype','used','free'])
def test_only_exact_idle_codex_utility_below64mib_is_allowed(monkeypatch,changed):
    import psutil
    pid=11277 if changed=='pid' else gpu.IDLE_CODEX_PID
    used=65 if changed=='used' else 12
    free=3000 if changed=='free' else 8000
    monkeypatch.setattr(gpu.subprocess,'run',lambda a,**k:SimpleNamespace(
        stdout=f'{pid}\n' if 'compute-apps' in a[1] else f'0,{used},{free},8192\n'))
    argv=['--type=utility','--utility-sub-type=on_device_model.mojom.OnDeviceModelService','--service-sandbox-type=on_device_model_execution']
    if changed=='subtype':argv[1]='--type=gpu-process'
    monkeypatch.setattr(psutil,'Process',lambda _:SimpleNamespace(
        create_time=lambda:gpu.IDLE_CODEX_CREATED+(1 if changed=='created' else 0),
        exe=lambda:'changed.exe' if changed=='exe' else gpu.IDLE_CODEX_EXE,cmdline=lambda:argv))
    if changed=='none':
        receipt=gpu.require_empty_gpu();assert receipt['compute_pids']==[gpu.IDLE_CODEX_PID]
        assert 'not a display-only' in receipt['allowed_idle_utility'][0]['interpretation']
    else:
        with pytest.raises((RuntimeError,MemoryError)):gpu.require_empty_gpu()


class Tensor:
    def __init__(self,array):self.array=np.asarray(array);self.shape=self.array.shape;self.ndim=self.array.ndim;self.dtype='float32'
    def to(self,_):return self
    def cpu(self):return self
    def tolist(self):return self.array.tolist()
    def __getitem__(self,item):return Tensor(self.array[item])


def test_gpu_predictor_sigmoid_no_truncation_batch2_contract():
    calls=[]
    def tokenizer(q,d,**kwargs):
        calls.append((q,d,kwargs));return {'input_ids':Tensor([[1,2,3]]*len(q))}
    model=lambda **kw:SimpleNamespace(logits=Tensor([[0.0]]*kw['input_ids'].shape[0]))
    torch=SimpleNamespace(cuda=SimpleNamespace(reset_peak_memory_stats=lambda _:None),
                          inference_mode=nullcontext,float32='float32',sigmoid=lambda x:Tensor(1/(1+np.exp(-x.array))))
    predictor=CudaPairPredictor(model,tokenizer,torch)
    assert predictor.predict([('q','a'),('q','b'),('q','c')],batch_size=2)==[.5,.5,.5]
    assert [len(c[0]) for c in calls]==[2,1]
    assert all(c[2]['truncation'] is False for c in calls)


def test_predictor_rejects_oversize_pair_without_calling_model():
    def model(**_):raise AssertionError('should reject before forward')
    tokenizer=lambda *a,**k:{'input_ids':Tensor([[0]*513])}
    torch=SimpleNamespace(cuda=SimpleNamespace(reset_peak_memory_stats=lambda _:None),inference_mode=nullcontext)
    with pytest.raises(ValueError,match='512-token'):CudaPairPredictor(model,tokenizer,torch).predict([('q','d')],batch_size=2)


@pytest.mark.parametrize('change',['missing','negative','peak','device'])
def test_invalid_gpu_memory_cannot_pass(change):
    row={'device':'cuda:0','allocated_bytes':100,'reserved_bytes':200,'peak_allocated_bytes':150,
         'peak_reserved_bytes':220,'free_bytes':500,'total_bytes':1000}
    validate_gpu_memory(row)
    if change=='missing':row.pop('peak_allocated_bytes')
    elif change=='negative':row['free_bytes']=-1
    elif change=='peak':row['peak_allocated_bytes']=90
    else:row['device']='cpu'
    with pytest.raises(ValueError):validate_gpu_memory(row)


def test_gpu_loader_uses_same_inherited_failure_budget(tmp_path,monkeypatch):
    from ragscope.experiments.rerank_campaign import load_reranker_with_receipt
    import ragscope.retrieval.verified_reranker as loader
    atomic_json(tmp_path/'LOAD_STATE.json',{'attempts':2,'backend_failures':0})
    def fail(_):raise MemoryError('offline resource wait')
    monkeypatch.setattr(loader,'load_verified_gpu_reranker',fail)
    with pytest.raises(MemoryError):load_reranker_with_receipt(tmp_path,tmp_path,device='cuda')
    state=json.loads((tmp_path/'LOAD_STATE.json').read_text())
    assert state['attempts']==3 and state['backend_failures']==0
