"""Pinned local GGUF embeddings on the root-owned v3 GPU backend."""
import json
import os
from pathlib import Path
import time
import numpy as np
from ragscope.experiments.ledger import append_jsonl,read_records
from ragscope.experiments.runtime import require_ram
from ragscope.indexing.array_index import ArrayCache
from ragscope.indexing.identity import artifact,atomic_json,file_hash,signature
from ragscope.indexing.local_embedding import HFJsonTokenizer,QUERY_INSTRUCTION

WEIGHT_SHA='06507c7b42688469c4e7298b0a1e16deff06caf291cf0a5b278c308249c3e439'
TOKENIZER_SHA='def76fb086971c7867b829c23a26261e38d9d74e02139253b38aeb9df8b4b50a'
TAG='ragscope-v3-embed:q8_0'


def register_existing_weight(run,service):
    runtime=Path(run).parents[1]
    weight=runtime/'models/qwen3-embedding-0.6b/Qwen3-Embedding-0.6B-Q8_0.gguf'
    tokenizer=runtime/'models/qwen3-embedding-0.6b/tokenizer.json'
    if file_hash(weight)!=WEIGHT_SHA or file_hash(tokenizer)!=TOKENIZER_SHA:raise ValueError('Pinned embedding source changed')
    blob=runtime/'models/ollama/blobs'/('sha256-'+WEIGHT_SHA)
    if blob.exists():
        if file_hash(blob)!=WEIGHT_SHA:raise ValueError('Existing blob does not match pinned weight')
    else:os.link(weight,blob)
    manifest=runtime/'models/ollama/manifests/registry.ollama.ai/library/ragscope-v3-embed/q8_0'
    service.verify()
    if not manifest.exists():
        payload={'model':TAG,'files':{'Qwen3-Embedding-0.6B-Q8_0.gguf':'sha256:'+WEIGHT_SHA},'stream':False}
        response=service.session.post(service.base_url+'/api/create',json=payload,timeout=120)
        response.raise_for_status()
        if response.json().get('status')!='success':raise ValueError('Private embedding registration unsuccessful')
    saved=json.loads(manifest.read_text('utf8'))
    model_layers=[x['digest'] for x in saved['layers'] if x['mediaType']=='application/vnd.ollama.image.model']
    if model_layers!=['sha256:'+WEIGHT_SHA]:raise ValueError('Tag maps to another weight; do not overwrite')
    response=service.session.post(service.base_url+'/api/show',json={'model':TAG},timeout=30);response.raise_for_status()
    show=response.json()
    receipt={'tag':TAG,'model':'Qwen/Qwen3-Embedding-0.6B','gguf_revision':'370f27d7550e0def9b39c1f16d3fbaa13aa67728',
        'weight':artifact(weight),'tokenizer':artifact(tokenizer),'tag_manifest':artifact(manifest),
        'model_info':show.get('model_info'),'capabilities':show.get('capabilities'),
        'source_weight_hardlink_reused':os.path.samefile(weight,blob),'new_download_bytes':0,
        'backend':service.record['backend_version'],'service_receipt':str(service.path)}
    destination=Path(run)/'V3_EMBEDDING_TAG.json'
    if destination.exists():
        old=json.loads(destination.read_text('utf8'))
        if old['weight']!=receipt['weight'] or old['tag_manifest']!=receipt['tag_manifest']:raise ValueError('Frozen registration changed')
    else:atomic_json(destination,receipt)
    return receipt


class PrivateEmbeddingClient:
    def __init__(self,run,service,max_tokens=4096):
        self.run,self.service=Path(run),service
        receipt=register_existing_weight(run,service)
        self.tokenizer=HFJsonTokenizer(receipt['tokenizer']['path'],'Qwen/Qwen3-Embedding-0.6B','97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3')
        self.max_tokens=max_tokens
        self.identity={'model':receipt['model'],'gguf_revision':receipt['gguf_revision'],'weight_sha256':WEIGHT_SHA,
            'tokenizer_revision':self.tokenizer.revision,'tokenizer_sha256':TOKENIZER_SHA,'backend':'ollama',
            'backend_version':service.record['backend_version'],'tag_manifest_sha256':receipt['tag_manifest']['sha256'],
            'device':'cuda:0','num_gpu':999,'num_ctx':max_tokens,'quantization':'Q8_0','dimensions':1024,
            'pooling':'GGUF_BACKEND_MODEL_DEFINED','model_info':receipt['model_info'],
            'normalization':'unit_l2_float32','query_prefix':QUERY_INSTRUCTION,'truncate':False}
        self.cache=ArrayCache(self.run/'embedding_cache',self.identity)
        self.ledger=self.run/'EMBEDDING_REQUESTS.jsonl'
        self.used={x['request_id'] for x in read_records(self.ledger)}
        self.last_metrics=[]

    def embed(self,texts):
        if not texts:return np.empty((0,1024),dtype=np.float32)
        if any(len(self.tokenizer.encode(t))+1>self.max_tokens for t in texts):raise ValueError('Embedding input over full context bound; no truncation')
        vectors=[self.cache.get(t) for t in texts];missing=list(dict.fromkeys(t for t,v in zip(texts,vectors) if v is None))
        self.last_metrics=[]
        for offset in range(0,len(missing),8):
            batch=missing[offset:offset+8];rid=signature([self.identity,batch]);require_ram(2*1024**3);self.service.verify()
            if rid in self.used:raise RuntimeError('Embedding request already issued without complete cache; reconcile failure, do not silently repeat')
            row={'request_id':rid,'model_identity':self.identity,'input_hashes':[signature(t) for t in batch],
                'inputs':batch,'body_tokens':[len(self.tokenizer.encode(t)) for t in batch],
                'status':'RESERVED','new_generation_requests':0,'reserved_unix':time.time()}
            row['record_hash']=signature(row);append_jsonl(self.ledger,row);self.used.add(rid)
            started=time.perf_counter()
            response=self.service.session.post(self.service.base_url+'/api/embed',json={'model':TAG,'input':batch,
                'truncate':False,'keep_alive':'5m','options':{'num_ctx':self.max_tokens,'num_gpu':999,'num_thread':4}},timeout=180)
            response.raise_for_status();raw=response.json()
            matrix=np.asarray(raw['embeddings'],dtype=np.float32)
            if matrix.shape!=(len(batch),1024) or not np.isfinite(matrix).all():raise ValueError('Invalid embedding dimensions or values')
            norms=np.linalg.norm(matrix,axis=1,keepdims=True)
            if np.any(norms<=0):raise ValueError('Zero embedding')
            normalized=matrix/norms
            for text,vector in zip(batch,normalized):self.cache.put(text,vector)
            meta={k:v for k,v in raw.items() if k!='embeddings'}
            metric={'request_id':rid,'raw_backend_metadata':meta,'wall_seconds':time.perf_counter()-started,
                'vector_artifacts':[artifact(self.cache.path/(self.cache.key(t)+'.npy')) for t in batch],
                'raw_embedding_array_sha256':signature(raw['embeddings']),'normalization_l2':norms[:,0].tolist()}
            atomic_json(self.run/'embedding_responses'/(rid+'.json'),metric);self.last_metrics.append(metric)
        vectors=[v if v is not None else self.cache.get(t) for t,v in zip(texts,vectors)]
        return np.stack(vectors).astype(np.float32)

    def embed_query(self,question):return self.embed([QUERY_INSTRUCTION+question])[0]
