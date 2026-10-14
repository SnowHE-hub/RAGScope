"""Compact PAPER_GIVEN development vectors; never claims a full-corpus index."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
PROJECT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(PROJECT))
from ragscope.experiments.runtime import configure,require_ram
from ragscope.indexing.identity import artifact,atomic_json,canonical_json,signature
from ragscope.v3.splits import read_split


def worker(run,runtime,segment):
    import numpy as np
    from filelock import FileLock
    from ragscope.experiments.ledger import RecordStore
    from ragscope.indexing.array_index import ArrayCache
    from ragscope.v3.private_embeddings import PrivateEmbeddingClient
    from ragscope.v3.local_runtime import ManagedOllama
    from ragscope.v3.embedding_service_guard import EmbeddingServiceGuard
    recipe_path=run/'S3_SCOPED_VECTOR_INPUTS.json';recipe=json.loads(recipe_path.read_text('utf8'))
    selected=recipe['papers'][segment*20:(segment+1)*20]
    if not selected:raise ValueError('Segment outside compact scope recipe')
    identity=recipe['embedding'];cache=ArrayCache(run/'embedding_cache',identity)
    config={'recipe':artifact(recipe_path),'runner':artifact(__file__),'model_identity':identity,
        'client':artifact(PROJECT/'ragscope/v3/private_embeddings.py'),'scope':'PAPER_GIVEN_COMPACT_ALTERNATIVE_NOT_FULL_INDEX'}
    with FileLock(str(run/'scoped_vectors_writer.lock'),timeout=0):
        store=RecordStore(run/'S3_SCOPED_VECTOR_PAPERS.jsonl',config,max_records=len(recipe['papers']))
        pending=[p for p in selected if p['doc_id'] not in store.ids]
        docs=[]
        for p in pending:
            if artifact(p['source']['path'])!=p['source']:raise ValueError('Source paper chunks changed')
            d=json.loads(Path(p['source']['path']).read_text('utf8'));docs.append((p,d['chunks']))
        missing=any(cache.get(x['text']) is None for _,chunks in docs for x in chunks)
        def consume(client=None):
            for p,chunks in docs:
                require_ram(2*1024**3)
                texts=[x['text'] for x in chunks]
                vectors=client.embed(texts) if client else np.stack([cache.get(t) for t in texts])
                if vectors.shape!=(len(chunks),1024):raise ValueError('Compact paper vector alignment invalid')
                path=run/'scoped_vectors'/(signature(p['doc_id'])+'.npz');path.parent.mkdir(exist_ok=True)
                if path.exists():
                    old=np.load(path,allow_pickle=False)
                    if not np.array_equal(old['vectors'],vectors) or old['chunk_ids'].tolist()!=[x['chunk_id'] for x in chunks]:raise ValueError('Unreceipted compact artifact differs')
                else:np.savez(path,chunk_ids=np.asarray([x['chunk_id'] for x in chunks]),vectors=vectors)
                store.add(p['doc_id'],{'doc_id':p['doc_id'],'chunk_n':len(chunks),'source':p['source'],'vectors':artifact(path),
                    'actual_backend_responses':client.last_metrics if client else [],'scope':'PAPER_GIVEN','new_generation_requests':0})
                print(canonical_json({'event':'SCOPED_PAPER_VECTORS','papers':len(store.ids),'expected':len(recipe['papers'])}),flush=True)
        if missing:
            with ManagedOllama(run,runtime/'tools/ollama-0.34.4/ollama.exe',num_ctx=4096) as service:
                client=PrivateEmbeddingClient(run,EmbeddingServiceGuard(service))
                if client.identity!=identity:raise ValueError('Compact vector space changed')
                consume(client)
        else:consume()
    return 0


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign',required=True);p.add_argument('--worker',action='store_true');p.add_argument('--segment',type=int)
    a=p.parse_args();runtime=Path(configure(cpu_only=True)['home']).resolve();run=Path(a.campaign).resolve()
    if run.parent!=runtime/'runs':raise ValueError('Wrong campaign')
    require_ram(2*1024**3)
    if a.worker:return worker(run,runtime,a.segment)
    recipe_path=run/'S3_SCOPED_VECTOR_INPUTS.json'
    if not recipe_path.exists():
        public=read_split(run,'D_search',purpose='inference')+read_split(run,'D_confirm',purpose='inference')
        wanted=sorted({d for q in public for d in q['scope']['doc_ids']})
        from ragscope.experiments.ledger import read_records
        source=json.loads((run/'S3_RETRIEVAL_SOURCE_PREPARATION.json').read_text('utf8'))
        records=read_records(Path(source['journal']['path']))
        papers=[{'doc_id':x['doc_id'],'source':x['output']} for x in records if x['group']=='R512' and x['doc_id'] in wanted]
        if {p['doc_id'] for p in papers}!=set(wanted):raise ValueError('Missing visible development paper')
        recipe={'papers':papers,'embedding':json.loads((run/'S3_PRIVATE_EMBEDDING_SMOKE.json').read_text('utf8'))['model_identity'],
            'splits':artifact(run/'SPLITS_V3.json'),'source':artifact(run/'S3_RETRIEVAL_SOURCE_PREPARATION.json'),
            'runner':artifact(__file__),'H_QA_read':False,'question_gold_read':False,
            'scope':'PAPER_GIVEN_COMPACT_ALTERNATIVE; full corpus Dense remains BLOCKED'}
        atomic_json(recipe_path,recipe)
    recipe=json.loads(recipe_path.read_text('utf8'))
    if recipe['runner']!=artifact(__file__):raise ValueError('Compact recipe implementation changed')
    for segment in range((len(recipe['papers'])+19)//20):
        command=[sys.executable,str(Path(__file__).resolve()),'--campaign',str(run),'--worker','--segment',str(segment)]
        # Capture child output directly: the earlier Windows inherited-handle log was empty.
        log=run/('S3_SCOPED_VECTOR_SEGMENT_'+str(segment)+'.log')
        with log.open('ab') as handle:
            result=subprocess.run(command,cwd=PROJECT,stdout=handle,stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform=='win32' else 0)
        if result.returncode:
            atomic_json(run/'S3_SCOPED_VECTOR_BLOCK.json',{'status':'BLOCKED_COMPACT_ALTERNATIVE','segment':segment,
                'actual_worker_exit':result.returncode,'log':artifact(log),'no_automatic_retry':True});return result.returncode
    from ragscope.experiments.ledger import read_records
    rows=read_records(run/'S3_SCOPED_VECTOR_PAPERS.jsonl')
    if len(rows)!=len(recipe['papers']):raise ValueError('Compact paper vector coverage incomplete')
    atomic_json(run/'S3_SCOPED_VECTOR_COMPLETION.json',{'status':'PASS_COMPACT_PAPER_GIVEN_VECTORS','paper_n':len(rows),
        'chunk_n':sum(x['chunk_n'] for x in rows),'records':artifact(run/'S3_SCOPED_VECTOR_PAPERS.jsonl'),
        'recipe':artifact(recipe_path),'new_generation_requests':0,'full_corpus_index_status':'BLOCKED','H_QA_read':False})
    print(canonical_json({'status':'PASS_COMPACT_PAPER_GIVEN_VECTORS','papers':len(rows)}));return 0


if __name__=='__main__':raise SystemExit(main())
