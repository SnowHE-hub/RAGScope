"""Serial query encoder and one generative model, with owned hot residency."""
import os
import socket
import subprocess
import time
import psutil
import requests
import numpy as np
from ragscope.experiments.runtime import require_ram
from ragscope.experiments.ledger import append_jsonl
from ragscope.indexing.identity import atomic_json,file_hash,signature
from ragscope.v3.gpu import GpuSession
from ragscope.v3.local_runtime import ManagedOllama
from ragscope.v3.private_embeddings import PrivateEmbeddingClient,TAG
from ragscope.indexing.local_embedding import QUERY_INSTRUCTION


class OnlineOwnedOllama(ManagedOllama):
    """Two small resident weights: one encoder and one2B generator; all calls serial."""
    def __enter__(self):
        require_ram(self.startup_ram_bytes)
        if not self.binary.is_relative_to(self.runtime/'tools') or not self.binary.is_file():raise ValueError('Verified runtime binary required')
        with socket.socket() as probe:probe.bind(('127.0.0.1',self.port))
        self.gpu=GpuSession(self.run);self.gpu.__enter__()
        self.directory=self.run/'services'/f'online_{os.getpid()}_{time.time_ns()}';self.directory.mkdir(parents=True)
        self.path=self.directory/'service.json';self.log=(self.directory/'service.log').open('a',encoding='utf8')
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0',OLLAMA_HOST=f'127.0.0.1:{self.port}',
            OLLAMA_MODELS=str(self.runtime/'models/ollama'),OLLAMA_NO_CLOUD='1',OLLAMA_MAX_LOADED_MODELS='2',
            OLLAMA_NUM_PARALLEL='1',OLLAMA_CONTEXT_LENGTH=str(self.num_ctx),OLLAMA_GPU_OVERHEAD=str(512*1024**2),
            OLLAMA_KEEP_ALIVE='5m',OLLAMA_DEBUG='1',LLAMA_ARG_CACHE_RAM='0')
        try:
            self.process=subprocess.Popen([str(self.binary),'serve'],stdout=self.log,stderr=subprocess.STDOUT,env=env,cwd=self.run,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
            owned=psutil.Process(self.process.pid)
            self.record={'status':'STARTING','pid':owned.pid,'creation_time':owned.create_time(),'binary':str(self.binary),
                'binary_sha256':file_hash(self.binary),'argv':owned.cmdline(),'cwd':str(self.run),'base_url':self.base_url,
                'lease':self.gpu.receipt,'environment':{k:env[k] for k in env if k.startswith(('OLLAMA_','LLAMA_ARG_')) or k=='CUDA_VISIBLE_DEVICES'},
                'original_jobs_policy':'USER_MANUAL_RESUME','residency_policy':'one encoder plus one2B generative model, loaded and called serially'}
            atomic_json(self.path,self.record);deadline=time.monotonic()+60
            while time.monotonic()<deadline:
                if self.process.poll() is not None:raise RuntimeError('Owned online service exited')
                try:
                    response=self.session.get(self.base_url+'/api/version',timeout=2);response.raise_for_status()
                    self.record.update(status='READY',backend_version=response.json()['version']);atomic_json(self.path,self.record);return self
                except requests.RequestException:time.sleep(.5)
            raise RuntimeError('Bounded online startup timeout')
        except BaseException as exc:self.__exit__(type(exc),exc,exc.__traceback__);raise

    def residency(self,expected):
        self.verify();response=self.session.get(self.base_url+'/api/ps',timeout=10);response.raise_for_status()
        models=response.json().get('models',[])
        if {m['name'] for m in models}!=set(expected) or any(m.get('size_vram',0)<=0 for m in models):
            raise RuntimeError('Both permitted models must be GPU resident, without another generative model')
        return models


class UncachedOnlineEmbedding:
    def __init__(self,run,service):
        self.client=PrivateEmbeddingClient(run,service);self.service,self.run=service,run
        self.identity=self.client.identity

    def query(self,question,request_id):
        text=QUERY_INSTRUCTION+question
        if len(self.client.tokenizer.encode(text))+1>4096:raise ValueError('Online query exceeds embedding budget')
        event={'request_id':request_id,'input_hash':signature(text),'identity':self.identity,'embedding_cache':False,'new_generation_requests':0}
        append_jsonl(self.run/'ONLINE_EMBEDDING_REQUESTS.jsonl',{**event,'event':'RESERVED'})
        start=time.perf_counter()
        response=self.service.session.post(self.service.base_url+'/api/embed',json={'model':TAG,'input':[text],
            'truncate':False,'keep_alive':'5m','options':{'num_ctx':4096,'num_gpu':999,'num_thread':4}},timeout=180)
        response.raise_for_status();raw=response.json();matrix=np.asarray(raw['embeddings'],dtype=np.float32)
        if matrix.shape!=(1,1024) or not np.isfinite(matrix).all() or np.linalg.norm(matrix[0])<=0:raise ValueError('Invalid online embedding')
        vector=matrix[0]/np.linalg.norm(matrix[0]);seconds=time.perf_counter()-start
        append_jsonl(self.run/'ONLINE_EMBEDDING_REQUESTS.jsonl',{**event,'event':'COMPLETED','wall_seconds':seconds,
            'vector_hash':signature(vector.tolist()),'backend_metadata':{k:v for k,v in raw.items() if k!='embeddings'}})
        return vector,seconds
