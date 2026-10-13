"""Lightweight per-request sampling; no torch import or allocator-peak claims."""
from __future__ import annotations
import json
from pathlib import Path
import shutil
import subprocess
import threading
import time
import urllib.request

from ragscope.generation.llm_client import GenerationError, _local_url, _NoRedirect


class RequestResourceSampler:
    """Sample device-total VRAM and owned service aggregate RSS every second.

    Device VRAM includes every process. RSS sums parent/children and can count
    shared resident pages more than once. Short-lived peaks can be missed.
    """
    def __init__(self, service_receipt, base_url, model, *, gpu_device=0, interval=1.0):
        if not .5<=interval<=1.0:
            raise ValueError('Sampling interval must be 0.5 to 1 second')
        self.service_receipt=Path(service_receipt)
        self.base_url=_local_url(base_url)
        self.model,self.gpu_device,self.interval=model,gpu_device,interval
        self.samples=[]; self.errors=[]; self.stop=threading.Event(); self.thread=None
        self.nvidia_smi=shutil.which('nvidia-smi')

    def _owned_process(self):
        import psutil
        receipt=json.loads(self.service_receipt.read_text(encoding='utf-8'))
        process=psutil.Process(receipt['pid'])
        if abs(process.create_time()-receipt['creation_time'])>.01:
            raise GenerationError('service_not_ready','Service PID reused before resource sampling')
        return process

    def sample_once(self):
        import psutil
        value={'monotonic_seconds':time.perf_counter(),'host_available_bytes':psutil.virtual_memory().available,
               'service_tree_rss_bytes':None,'device_total_used_bytes':None,'device_capacity_bytes':None}
        try:
            parent=self._owned_process()
            processes=[parent]+parent.children(recursive=True)
            sizes=[]
            for process in processes:
                try: sizes.append(process.memory_info().rss)
                except (psutil.NoSuchProcess,psutil.AccessDenied): pass
            value['service_tree_rss_bytes']=sum(sizes) if sizes else None
            value['service_process_count']=len(sizes)
        except (OSError,ValueError,KeyError,psutil.Error,GenerationError) as exc:
            self.errors.append({'instrument':'service_rss','error':str(exc)})
        if self.nvidia_smi:
            try:
                response=subprocess.run([self.nvidia_smi,f'--id={self.gpu_device}',
                    '--query-gpu=memory.used,memory.total','--format=csv,noheader,nounits'],
                    capture_output=True,text=True,timeout=5,check=True,
                    creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                used,total=[float(part.strip()) for part in response.stdout.strip().split(',')]
                value.update(device_total_used_bytes=int(used*1024**2),device_capacity_bytes=int(total*1024**2))
            except (OSError,ValueError,subprocess.SubprocessError) as exc:
                self.errors.append({'instrument':'nvidia_smi','error':str(exc)})
        elif not any(row['instrument']=='nvidia_smi' for row in self.errors):
            self.errors.append({'instrument':'nvidia_smi','error':'Executable unavailable'})
        self.samples.append(value)

    def _worker(self):
        while not self.stop.is_set():
            self.sample_once()
            if self.stop.wait(self.interval): break

    def start(self):
        self.started=time.perf_counter()
        self.thread=threading.Thread(target=self._worker,name='ragscope-resource-sampler',daemon=True)
        self.thread.start()
        return self

    def _resident(self):
        opener=urllib.request.build_opener(urllib.request.ProxyHandler({}),_NoRedirect())
        with opener.open(self.base_url+'/api/ps',timeout=3) as response:
            rows=json.load(response).get('models',[])
        matches=[row for row in rows if row.get('name')==self.model or row.get('model')==self.model]
        if len(matches)!=1:
            return {'status':'MODEL_NOT_RESIDENT_OR_AMBIGUOUS','matching_entries':len(matches)}
        row=matches[0]
        return {'status':'OBSERVED','endpoint':'/api/ps','model':row.get('name',row.get('model')),
                'model_resident_bytes':row.get('size'),'model_gpu_resident_bytes':row.get('size_vram'),
                'context_length':row.get('context_length'),'digest':row.get('digest'),
                'interpretation':'Backend-reported resident allocation after request; not request peak or layer offload count'}

    def finish(self):
        self.stop.set()
        self.thread.join(timeout=6)
        if self.thread.is_alive():
            self.errors.append({'instrument':'sampler','error':'Worker exceeded bounded shutdown'})
        samples=list(self.samples)
        def extrema(key, reducer):
            values=[row[key] for row in samples if row.get(key) is not None]
            return reducer(values) if values else None
        try: resident=self._resident()
        except Exception as exc:
            resident={'status':'UNAVAILABLE','error':str(exc)}
        return {'status':'MEASURED_SAMPLED' if samples else 'NO_SAMPLES','interval_seconds':self.interval,
                'sample_count':len(samples),'elapsed_seconds':time.perf_counter()-self.started,
                'gpu_device':self.gpu_device,'peak_sampled_device_total_used_bytes':extrema('device_total_used_bytes',max),
                'peak_sampled_service_tree_rss_bytes':extrema('service_tree_rss_bytes',max),
                'minimum_sampled_host_available_bytes':extrema('host_available_bytes',min),
                'device_capacity_bytes':extrema('device_capacity_bytes',max),
                'peak_gpu_allocated_bytes':None,'peak_gpu_reserved_bytes':None,'actual_layer_offload_count':None,
                'limitations':['Sampled extrema can miss peaks','GPU memory is whole-device total, not this process',
                               'RSS is summed owned service and descendants; shared pages may be counted repeatedly',
                               'Allocator peaks and layer offload count are not exposed by this backend'],
                'resident_after_request':resident,'samples':samples,'instrument_errors':self.errors}


class MeasuredClient:
    def __init__(self,client,service_receipt,*,gpu_device=0,interval=1.0):
        self.client,self.service_receipt,self.gpu_device,self.interval=client,service_receipt,gpu_device,interval
        self.last_metadata={}
    def __getattr__(self,key): return getattr(self.client,key)
    def chat(self,messages,**kwargs):
        self.last_metadata={}
        sampler=RequestResourceSampler(self.service_receipt,self.base_url,self.model,
                                       gpu_device=self.gpu_device,interval=self.interval).start()
        request_started=time.perf_counter()
        try:
            return self.client.chat(messages,**kwargs)
        finally:
            request_finished=time.perf_counter()
            measured=sampler.finish()
            self.last_metadata={**getattr(self.client,'last_metadata',{}),'resources':measured,
                'client_request_wall_ms':(request_finished-request_started)*1000,
                'telemetry_finalize_ms':(time.perf_counter()-request_finished)*1000,
                'timing_note':'Pipeline generation_ms includes telemetry finalization; backend compute durations remain separate'}
