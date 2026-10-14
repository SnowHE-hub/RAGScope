"""Build one selected real dense variant with cache and batch-level resume."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
PROJECT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(PROJECT))
from filelock import FileLock
from ragscope.experiments.ledger import RecordStore
from ragscope.experiments.runtime import configure,require_ram,require_disk
from ragscope.indexing.identity import artifact,atomic_json,canonical_json,signature
from ragscope.v3.local_runtime import ManagedOllama
from ragscope.v3.private_embeddings import PrivateEmbeddingClient
from ragscope.v3.embedding_service_guard import EmbeddingServiceGuard


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign',required=True)
    p.add_argument('--variant',required=True,choices=['R512_body','R512_metadata','F512_O0_metadata','P256_1024_metadata'])
    p.add_argument('--resume',action='store_true');p.add_argument('--dry-run',action='store_true');a=p.parse_args()
    runtime=Path(configure(cpu_only=True)['home']).resolve();run=Path(a.campaign).resolve()
    if run.parent!=runtime/'runs':raise ValueError('Wrong campaign')
    source=json.loads((run/'S3_CHUNK_SPARSE_INDICES.json').read_text('utf8'))
    pointer=source['variants'][a.variant]
    if artifact(pointer['path'])!=pointer:raise ValueError('Sparse manifest changed')
    sparse_manifest=json.loads(Path(pointer['path']).read_text('utf8'));directory=Path(pointer['path']).parent
    for x in sparse_manifest['artifacts']:
        if artifact(x['path'])!=x:raise ValueError('Sparse index artifacts changed')
    chunks=[json.loads(l) for l in (directory/'chunks.jsonl').read_text('utf8').splitlines()]
    field=sparse_manifest['identity']['retrieval_text_field']
    if a.dry_run:
        print(canonical_json({'variant':a.variant,'chunks':len(chunks),'field':field,'vector_bytes':len(chunks)*1024*4,
            'maximum_body_tokens':max(x['n_tokens'] for x in chunks),'generation_requests':0}));return 0
    require_ram(2*1024**3);require_disk(len(chunks)*1024*4)
    with FileLock(str(run/'chunk_dense_build.lock'),timeout=0):
        completed_path=directory/'DENSE_MANIFEST.json'
        if completed_path.exists():
            completed=json.loads(completed_path.read_text('utf8'))
            if not a.resume or any(artifact(x['path'])!=x for x in completed['artifacts']):raise ValueError('Completed dense index changed')
            print(canonical_json({'status':'REUSED_COMPLETE_DENSE_INDEX','variant':a.variant}));return 0
        with ManagedOllama(run,runtime/'tools/ollama-0.34.4/ollama.exe',num_ctx=4096) as service:
            client=PrivateEmbeddingClient(run,EmbeddingServiceGuard(service))
            identity={'sparse_manifest':pointer,'embedding':client.identity,'variant':a.variant,'canonical_body_preserved':True}
            config={'identity':identity,'runner':artifact(__file__),'embedding_client':artifact(PROJECT/'ragscope/v3/private_embeddings.py'),
                'batch_rows':64,'new_generation_requests':0,'gpu_consumer_check_seconds':5,
                'ownership_and_lease_check_every_request':True,
                'gpu_guard':artifact(PROJECT/'ragscope/v3/embedding_service_guard.py')}
            journal=directory/'DENSE_BATCHES.jsonl'
            if journal.exists() and not a.resume:raise ValueError('Partial dense index requires --resume')
            store=RecordStore(journal,config,max_records=(len(chunks)+63)//64)
            matrix_path=directory/'vectors.npy'
            matrix=np.lib.format.open_memmap(matrix_path,mode='r+' if matrix_path.exists() else 'w+',dtype=np.float32,shape=(len(chunks),1024))
            for offset in range(0,len(chunks),64):
                rid=a.variant+':dense:'+str(offset)
                if rid in store.ids:continue
                require_ram(2*1024**3);batch=chunks[offset:offset+64];texts=[x[field] for x in batch]
                vectors=client.embed(texts);matrix[offset:offset+len(batch)]=vectors;matrix.flush()
                store.add(rid,{'offset':offset,'chunk_ids':[x['chunk_id'] for x in batch],
                    'vector_artifacts':[artifact(client.cache.path/(client.cache.key(t)+'.npy')) for t in texts],
                    'actual_backend_responses':client.last_metrics,'new_generation_requests':0})
                if len(store.ids)%10==0:print(canonical_json({'event':'CHUNK_DENSE_PROGRESS','variant':a.variant,
                    'batches':len(store.ids),'chunks_completed':min(offset+64,len(chunks)),'expected_chunks':len(chunks)}),flush=True)
            for offset in range(0,len(chunks),64):
                for i,x in enumerate(chunks[offset:offset+64],offset):
                    vector=client.cache.get(x[field])
                    if vector is None:raise ValueError('Journal row lacks a valid real cached vector')
                    matrix[i]=vector
            matrix.flush()
            for offset in range(0,len(chunks),512):
                block=matrix[offset:offset+512]
                if not np.isfinite(block).all() or not np.allclose(np.linalg.norm(block,axis=1),1,atol=1e-5):raise ValueError('Invalid matrix block')
            del matrix
            outputs=sparse_manifest['artifacts']+[artifact(matrix_path),artifact(journal),artifact(journal.with_suffix('.config.json'))]
            result={'status':'complete','identity':identity,'signature':signature(identity),'count':len(chunks),
                'artifacts':outputs,'variant':a.variant,'new_generation_requests':0}
            atomic_json(completed_path,result)
            atomic_json(run/('S3_CHUNK_DENSE_'+a.variant+'.json'),{'status':'PASS_REAL_CHUNK_DENSE_INDEX',
                'index_manifest':artifact(completed_path),'chunk_n':len(chunks),'variant':a.variant,'new_generation_requests':0})
            print(canonical_json({'status':'PASS_REAL_CHUNK_DENSE_INDEX','variant':a.variant,'chunk_n':len(chunks)}))
    return 0


if __name__=='__main__':raise SystemExit(main())
