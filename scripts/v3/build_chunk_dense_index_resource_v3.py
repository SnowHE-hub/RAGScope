"""One bounded compatibility attempt: fresh process/backend per 1024 vectors."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
PROJECT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(PROJECT))
from ragscope.experiments.runtime import configure,require_ram
from ragscope.indexing.identity import artifact,atomic_json,canonical_json,signature


def worker(run,runtime,variant,segment):
    import numpy as np
    import psutil
    from filelock import FileLock
    from ragscope.experiments.ledger import RecordStore
    from ragscope.indexing.array_index import ArrayCache
    from ragscope.v3.local_runtime import ManagedOllama
    from ragscope.v3.private_embeddings import PrivateEmbeddingClient
    from ragscope.v3.embedding_service_guard import EmbeddingServiceGuard
    source=json.loads((run/'S3_CHUNK_SPARSE_INDICES.json').read_text('utf8'))['variants'][variant]
    if artifact(source['path'])!=source:raise ValueError('Sparse manifest changed')
    sparse=json.loads(Path(source['path']).read_text('utf8'));directory=Path(source['path']).parent
    chunks=[json.loads(l) for l in (directory/'chunks.jsonl').read_text('utf8').splitlines()]
    identity=json.loads((run/'S3_PRIVATE_EMBEDDING_SMOKE.json').read_text('utf8'))['model_identity']
    field=sparse['identity']['retrieval_text_field'];begin=segment*1024;end=min(len(chunks),begin+1024)
    if begin>=len(chunks):raise ValueError('Segment outside declared corpus')
    config={'sparse_manifest':source,'embedding':identity,'runner':artifact(__file__),
        'embedding_client':artifact(PROJECT/'ragscope/v3/private_embeddings.py'),
        'guard':artifact(PROJECT/'ragscope/v3/embedding_service_guard.py'),
        'resource_fix':artifact(run/'S3_CHUNK_DENSE_RESOURCE_COMPATIBILITY.json'),
        'segment_max_vectors':1024,'batch_rows':64,'original_source_journals_preserved':True}
    with FileLock(str(run/'chunk_dense_resource_worker.lock'),timeout=0):
        store=RecordStore(directory/'DENSE_BATCHES_RESOURCE_V3.jsonl',config,max_records=(len(chunks)+63)//64)
        pending=[offset for offset in range(begin,end,64) if variant+':resource_v3:'+str(offset) not in store.ids]
        if not pending:return 0
        cache=ArrayCache(run/'embedding_cache',identity)
        need_model=any(cache.get(x[field]) is None for offset in pending for x in chunks[offset:offset+64])
        matrix=np.lib.format.open_memmap(directory/'vectors.npy',mode='r+' if (directory/'vectors.npy').exists() else 'w+',dtype=np.float32,shape=(len(chunks),1024))
        print(canonical_json({'event':'RESOURCE_SEGMENT_START','variant':variant,'segment':segment,
            'available_host_bytes':psutil.virtual_memory().available,'own_rss_bytes':psutil.Process().memory_info().rss,'needs_model':need_model}),flush=True)
        def consume(client=None):
            for offset in pending:
                require_ram(2*1024**3);batch=chunks[offset:offset+64];texts=[x[field] for x in batch]
                cached=sum(cache.get(t) is not None for t in texts)
                vectors=client.embed(texts) if client else np.stack([cache.get(t) for t in texts])
                if vectors.shape!=(len(batch),1024) or not np.isfinite(vectors).all():raise ValueError('Invalid real segment vectors')
                matrix[offset:offset+len(batch)]=vectors;matrix.flush()
                store.add(variant+':resource_v3:'+str(offset),{'offset':offset,'chunk_ids':[x['chunk_id'] for x in batch],
                    'cache_hits_verified_before':cached,'vector_artifacts':[artifact(cache.path/(cache.key(t)+'.npy')) for t in texts],
                    'actual_backend_responses':client.last_metrics if client else [],'new_generation_requests':0})
        if need_model:
            with ManagedOllama(run,runtime/'tools/ollama-0.34.4/ollama.exe',num_ctx=4096) as service:
                client=PrivateEmbeddingClient(run,EmbeddingServiceGuard(service))
                if client.identity!=identity:raise ValueError('Embedding space changed during resource compatibility')
                consume(client)
        else:consume()
        del matrix
        print(canonical_json({'event':'RESOURCE_SEGMENT_TERMINAL','variant':variant,'segment':segment,
            'available_host_bytes':psutil.virtual_memory().available,'own_rss_bytes':psutil.Process().memory_info().rss}),flush=True)
    return 0


def finalize(run,variant):
    import numpy as np
    from ragscope.experiments.ledger import read_records
    source=json.loads((run/'S3_CHUNK_SPARSE_INDICES.json').read_text('utf8'))['variants'][variant]
    sparse=json.loads(Path(source['path']).read_text('utf8'));directory=Path(source['path']).parent
    journal=directory/'DENSE_BATCHES_RESOURCE_V3.jsonl';rows=read_records(journal)
    if len(rows)!=(sparse['count']+63)//64:raise ValueError('Segment journal not complete')
    matrix=np.load(directory/'vectors.npy',mmap_mode='r',allow_pickle=False)
    for offset in range(0,len(matrix),512):
        require_ram(2*1024**3)
        block=matrix[offset:offset+512]
        if not np.isfinite(block).all() or not np.allclose(np.linalg.norm(block,axis=1),1,atol=1e-5):raise ValueError('Matrix contains missing or invalid real vectors')
    identity={'sparse_manifest':source,'embedding':json.loads((run/'S3_PRIVATE_EMBEDDING_SMOKE.json').read_text('utf8'))['model_identity'],
        'variant':variant,'canonical_body_preserved':True,'resource_policy':'fresh process/backend per 1024 vectors'}
    outputs=sparse['artifacts']+[artifact(directory/'vectors.npy'),artifact(journal),artifact(journal.with_suffix('.config.json'))]
    result={'status':'complete','identity':identity,'signature':signature(identity),'count':sparse['count'],'artifacts':outputs,
        'variant':variant,'new_generation_requests':0,'resource_compatibility':True}
    destination=directory/'DENSE_MANIFEST.json'
    if destination.exists():raise ValueError('Do not overwrite a complete dense manifest')
    atomic_json(destination,result)
    atomic_json(run/('S3_CHUNK_DENSE_'+variant+'.json'),{'status':'PASS_REAL_CHUNK_DENSE_INDEX',
        'index_manifest':artifact(destination),'chunk_n':sparse['count'],'variant':variant,'new_generation_requests':0,'resource_compatibility':True})


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign',required=True)
    p.add_argument('--worker',action='store_true');p.add_argument('--variant',choices=['R512_body','R512_metadata']);p.add_argument('--segment',type=int)
    a=p.parse_args();runtime=Path(configure(cpu_only=True)['home']).resolve();run=Path(a.campaign).resolve()
    if run.parent!=runtime/'runs':raise ValueError('Wrong campaign')
    require_ram(2*1024**3)
    if a.worker:return worker(run,runtime,a.variant,a.segment)
    source=json.loads((run/'S3_CHUNK_SPARSE_INDICES.json').read_text('utf8'))
    for variant in ('R512_body','R512_metadata'):
        if (run/('S3_CHUNK_DENSE_'+variant+'.json')).exists():continue
        pointer=source['variants'][variant];sparse=json.loads(Path(pointer['path']).read_text('utf8'))
        for segment in range((sparse['count']+1023)//1024):
            command=[sys.executable,str(Path(__file__).resolve()),'--campaign',str(run),'--worker','--variant',variant,'--segment',str(segment)]
            result=subprocess.run(command,cwd=PROJECT,creationflags=subprocess.CREATE_NO_WINDOW if sys.platform=='win32' else 0)
            if result.returncode!=0:
                atomic_json(run/'S3_CHUNK_DENSE_FINAL_RESOURCE_BLOCK.json',{'status':'BLOCKED_BOUNDED_ATTEMPTS_EXHAUSTED',
                    'variant':variant,'segment':segment,'actual_worker_exit':result.returncode,'retry_further':False,
                    'original_and_partial_results_preserved':True});return result.returncode
        finalize(run,variant)
        print(canonical_json({'event':'DENSE_VARIANT_COMPLETE','variant':variant,'chunk_n':sparse['count']}),flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
