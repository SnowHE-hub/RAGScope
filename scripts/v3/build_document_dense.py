"""Resumable real document embeddings, aligned to the immutable BM25 documents."""
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
from ragscope.indexing.sparse_index import BM25Index
from ragscope.v3.local_runtime import ManagedOllama
from ragscope.v3.private_embeddings import PrivateEmbeddingClient


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign',required=True);p.add_argument('--resume',action='store_true');a=p.parse_args()
    runtime=Path(configure(cpu_only=True)['home']).resolve();run=Path(a.campaign).resolve()
    if run.parent!=runtime/'runs':raise ValueError('Wrong campaign')
    require_ram(2*1024**3);require_disk(20*1024**2)
    smoke=json.loads((run/'S3_PRIVATE_EMBEDDING_SMOKE.json').read_text('utf8'))
    if smoke['status']!='PASS_CONSTRUCTED_EMBEDDING_SMOKE':raise ValueError('Embedding gate did not pass')
    source_path=run/'S3_DOCUMENT_BM25_INDEX.json';source=json.loads(source_path.read_text('utf8'))
    if any(artifact(x['path'])!=x for x in source['outputs']):raise ValueError('Document BM25 artifacts changed')
    docs=BM25Index.load(source['outputs'][0]['path']).chunks
    with FileLock(str(run/'document_dense_build.lock'),timeout=0):
        with ManagedOllama(run,runtime/'tools/ollama-0.34.4/ollama.exe',num_ctx=4096) as service:
            client=PrivateEmbeddingClient(run,service)
            if client.identity!=smoke['model_identity']:raise ValueError('Embedding backend differs from calibrated identity')
            config={'source':artifact(source_path),'embedding':client.identity,'runner':artifact(__file__),
                'client':artifact(PROJECT/'ragscope/v3/private_embeddings.py'),'batch_size':8,'paper_ids':[d['doc_id'] for d in docs],
                'body_token_max':max(len(client.tokenizer.encode(d['text'])) for d in docs),'question_or_answer_gold_read':False}
            if config['body_token_max']+1>4096:raise ValueError('Complete document retrieval input exceeds budget')
            directory=runtime/'indexes'/run.name/'document_dense'/signature(config);directory.mkdir(parents=True,exist_ok=True)
            journal=directory/'BATCHES.jsonl'
            if journal.exists() and not a.resume:raise ValueError('Existing vector batches need --resume')
            store=RecordStore(journal,config,max_records=(len(docs)+7)//8)
            matrix_path=directory/'vectors.npy'
            matrix=np.lib.format.open_memmap(matrix_path,mode='r+' if matrix_path.exists() else 'w+',dtype=np.float32,shape=(len(docs),1024))
            for offset in range(0,len(docs),8):
                rid='document_dense:'+str(offset)
                if rid in store.ids:continue
                require_ram(2*1024**3);batch=docs[offset:offset+8];texts=[d['text'] for d in batch]
                vectors=client.embed(texts);matrix[offset:offset+len(batch)]=vectors;matrix.flush()
                store.add(rid,{'offset':offset,'paper_ids':[d['doc_id'] for d in batch],
                    'vector_artifacts':[artifact(client.cache.path/(client.cache.key(t)+'.npy')) for t in texts],
                    'actual_backend_responses':client.last_metrics,'model_identity':client.identity,
                    'new_generation_requests':0})
                if len(store.ids)%10==0:print(canonical_json({'event':'DOCUMENT_DENSE_PROGRESS','batches':len(store.ids),'papers_completed':min(offset+8,len(docs))}),flush=True)
            # Reconstruct all rows from independently verified cache artifacts, including resumes.
            for i,d in enumerate(docs):
                vector=client.cache.get(d['text'])
                if vector is None:raise ValueError('Complete journal lacks a real vector')
                matrix[i]=vector
            matrix.flush()
            norms=np.linalg.norm(matrix,axis=1)
            if not np.isfinite(matrix).all() or not np.allclose(norms,1,atol=1e-5):raise ValueError('Invalid document vector normalization')
            del matrix
            outputs=[artifact(matrix_path),artifact(journal),artifact(journal.with_suffix('.config.json'))]
            result={'status':'PASS_REAL_DOCUMENT_DENSE_INDEX','paper_n':len(docs),'dimensions':1024,
                'directory':str(directory),'config':config,'outputs':outputs,'new_generation_requests':0,
                'quality_status':'NOT_YET_EVALUATED','old_backend_comparability':'new GPU Ollama identity, not CPU llama'}
            atomic_json(run/'S3_DOCUMENT_DENSE_INDEX.json',result)
            print(canonical_json({'status':result['status'],'paper_n':len(docs),'batches':len(store.ids)}))
    return 0


if __name__=='__main__':raise SystemExit(main())
