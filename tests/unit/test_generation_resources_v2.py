"""Instrument contract tests only; no hardware/model measurements are fabricated."""
import json
from types import SimpleNamespace

import pytest

from ragscope.generation.llm_client import GenerationError, LocalLLMClient
from ragscope.generation.resources import MeasuredClient, RequestResourceSampler


def test_sampler_records_device_total_and_owned_tree_rss(tmp_path,monkeypatch):
    import psutil
    import ragscope.generation.resources as resources
    receipt=tmp_path/'service.json'; receipt.write_text(json.dumps({'pid':1,'creation_time':10}))
    child=SimpleNamespace(memory_info=lambda:SimpleNamespace(rss=200))
    parent=SimpleNamespace(create_time=lambda:10,memory_info=lambda:SimpleNamespace(rss=100),children=lambda recursive:[child])
    monkeypatch.setattr(psutil,'Process',lambda pid:parent)
    monkeypatch.setattr(psutil,'virtual_memory',lambda:SimpleNamespace(available=900))
    monkeypatch.setattr(resources.subprocess,'run',lambda *a,**k:SimpleNamespace(stdout='12, 100'))
    sampler=RequestResourceSampler(receipt,'http://127.0.0.1:11435','offline-fixture')
    sampler.nvidia_smi='offline-mocked-executable'; sampler.sample_once()
    assert sampler.samples[0]['service_tree_rss_bytes']==300
    assert sampler.samples[0]['device_total_used_bytes']==12*1024**2
    assert sampler.samples[0]['host_available_bytes']==900
    monkeypatch.setattr(parent,'create_time',lambda:11)
    with pytest.raises(GenerationError,match='reused'):
        sampler._owned_process()


def test_sampler_finish_aggregates_observed_samples_without_allocator_peak_claim(tmp_path,monkeypatch):
    sampler=RequestResourceSampler(tmp_path/'receipt','http://127.0.0.1:11435','offline-fixture')
    def sample():
        sampler.samples.append({'device_total_used_bytes':50,'service_tree_rss_bytes':40,
                                'host_available_bytes':100,'device_capacity_bytes':1000})
    monkeypatch.setattr(sampler,'sample_once',sample)
    monkeypatch.setattr(sampler,'_resident',lambda:{'status':'OBSERVED','model_gpu_resident_bytes':25})
    sampler.start(); result=sampler.finish()
    assert result['sample_count']>=1 and result['peak_sampled_device_total_used_bytes']==50
    assert result['peak_gpu_allocated_bytes'] is None
    assert result['resident_after_request']['model_gpu_resident_bytes']==25
    with pytest.raises(ValueError):
        RequestResourceSampler(tmp_path/'receipt','http://127.0.0.1:11435','x',interval=10)


def test_wrapper_preserves_transport_failure_and_attaches_measurement_status(monkeypatch):
    import ragscope.generation.resources as resources
    class Sampler:
        def __init__(self,*args,**kwargs):pass
        def start(self):return self
        def finish(self):return {'status':'OFFLINE_TEST_FIXTURE'}
    class Client:
        model='offline'; base_url='http://127.0.0.1:11435'; last_metadata={}
        def chat(self,messages,**kwargs):
            self.last_metadata={'backend_field':1}
            raise TimeoutError('offline fixture')
    monkeypatch.setattr(resources,'RequestResourceSampler',Sampler)
    client=MeasuredClient(Client(),'unused')
    with pytest.raises(TimeoutError):client.chat([])
    assert client.last_metadata['backend_field']==1 and client.last_metadata['resources']=={'status':'OFFLINE_TEST_FIXTURE'}
    assert client.last_metadata['client_request_wall_ms']>=0 and client.last_metadata['telemetry_finalize_ms']>=0


def test_runtime_policy_participates_in_model_identity():
    policy={'prompt_ram_cache_mib':0,'prompt_cache_policy':'backend_history_disabled'}
    client=LocalLLMClient('offline-fixture',backend_runtime_config=policy)
    policy['prompt_ram_cache_mib']=5
    assert client.identity['backend_runtime_config']['prompt_ram_cache_mib']==0
