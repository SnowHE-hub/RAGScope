"""Audit six old CPU vectors and freeze real dev-only GPU query embeddings."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
PROJECT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(PROJECT))
from ragscope.experiments.runtime import configure
from ragscope.indexing.identity import artifact,atomic_json,canonical_json,signature
from ragscope.v3.local_runtime import ManagedOllama
from ragscope.v3.private_embeddings import PrivateEmbeddingClient
from ragscope.v3.splits import read_split
from ragscope.indexing.local_embedding import QUERY_INSTRUCTION


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign',required=True);a=p.parse_args()
    runtime=Path(configure(cpu_only=True)['home']).resolve();run=Path(a.campaign).resolve()
    if run.parent!=runtime/'runs':raise ValueError('Wrong campaign')
    old_path=runtime/'indexes/20261014_v2/all_chunks/ba23a7f21f5bc3ed9fccb5778163c07a6b2fda530c769ff235e15d44dd5238f6/dense_397d90b3f4fbe03e/manifest.json'
    old=json.loads(old_path.read_text('utf8'))
    if any(artifact(x['path'])!=x for x in old['artifacts']):raise ValueError('Old real index changed')
    chunks=json.loads((old_path.parent/'chunks.json').read_text('utf8'))
    old_matrix=np.load(old_path.parent/'vectors.npy',mmap_mode='r',allow_pickle=False)
    selected=sorted(range(len(chunks)),key=lambda i:signature([42,'backend_bridge_predeclared',chunks[i]['chunk_id']]))[:6]
    texts=[chunks[i]['text'] for i in selected];old_vectors=np.stack([old_matrix[i] for i in selected]);ids=[chunks[i]['chunk_id'] for i in selected]
    del chunks,old_matrix
    questions=[]
    for split in ('D_search','D_confirm'):
        questions.extend([{**q,'dataset_role':split} for q in read_split(run,split,purpose='inference')])
    if len(questions)!=220 or len({q['query_id'] for q in questions})!=220:raise ValueError('Unexpected development query design')
    with ManagedOllama(run,runtime/'tools/ollama-0.34.4/ollama.exe',num_ctx=4096) as service:
        client=PrivateEmbeddingClient(run,service)
        new_vectors=client.embed(texts)
        similarities=np.sum(new_vectors*old_vectors,axis=1)
        differences=np.max(np.abs(new_vectors-old_vectors),axis=1)
        bridge={'status':'PASS_SIX_VECTOR_NUMERIC_PROXY' if float(similarities.min())>=.999 else 'FAIL_SIX_VECTOR_NUMERIC_PROXY',
            'cases':[{'chunk_id':cid,'cosine':float(cos),'max_coordinate_difference':float(diff)} for cid,cos,diff in zip(ids,similarities,differences)],
            'minimum_cosine_gate':.999,'old_index':artifact(old_path),'old_identity':old['identity'].get('embedding'),
            'new_identity':client.identity,'new_generation_requests':0,
            'limitations':['Only six predetermined canonical inputs; does not establish all-query ranking equivalence or CPU/GPU cost equivalence.',
                'New GPU identity always retained; the historical CPU journals remain unchanged.']}
        atomic_json(run/'S3_EMBEDDING_BACKEND_BRIDGE.json',bridge)
        output=run/'S3_DEV_QUERY_VECTORS.npz'
        if output.exists():raise ValueError('Frozen query matrix exists; do not overwrite')
        vectors=[]
        for offset in range(0,len(questions),8):
            batch=questions[offset:offset+8]
            vectors.extend(client.embed([QUERY_INSTRUCTION+q['question'] for q in batch]))
            if offset%40==0:print(canonical_json({'event':'DEV_QUERY_VECTORS','completed':min(offset+8,len(questions))}),flush=True)
        np.savez(output,query_ids=np.asarray([q['query_id'] for q in questions]),vectors=np.stack(vectors))
        result={'status':'PASS_REAL_DEV_QUERY_VECTORS','question_n':220,'D_search_n':120,'D_confirm_n':100,
            'model_identity':client.identity,'vectors':artifact(output),'splits':artifact(run/'SPLITS_V3.json'),
            'question_gold_read':False,'H_QA_questions_read':False,'new_generation_requests':0,
            'bridge':artifact(run/'S3_EMBEDDING_BACKEND_BRIDGE.json'),'runner':artifact(__file__)}
        atomic_json(run/'S3_DEV_QUERY_VECTORS.json',result)
        print(canonical_json({'bridge_status':bridge['status'],'minimum_cosine':float(similarities.min()),'query_n':220,'status':result['status']}))
    return 0


if __name__=='__main__':raise SystemExit(main())
