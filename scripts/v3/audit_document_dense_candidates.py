"""Evaluate real raw-question document candidates; ambiguous scope stays diagnostic."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
PROJECT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(PROJECT))
from ragscope.experiments.runtime import configure
from ragscope.indexing.identity import artifact,atomic_json,canonical_json
from ragscope.indexing.sparse_index import BM25Index
from ragscope.v3.splits import read_split


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign',required=True);a=p.parse_args()
    runtime=Path(configure(cpu_only=True)['home']).resolve();run=Path(a.campaign).resolve()
    if run.parent!=runtime/'runs':raise ValueError('Wrong campaign')
    dense_path=run/'S3_DOCUMENT_DENSE_INDEX.json';dense=json.loads(dense_path.read_text('utf8'))
    query_path=run/'S3_DEV_QUERY_VECTORS.json';query=json.loads(query_path.read_text('utf8'))
    bm_path=run/'S3_DOCUMENT_BM25_INDEX.json';bm=json.loads(bm_path.read_text('utf8'))
    if query['model_identity']!=dense['config']['embedding']:raise ValueError('Vector spaces differ')
    for x in dense['outputs']+[query['vectors']]+bm['outputs']:
        if artifact(x['path'])!=x:raise ValueError('Frozen artifact changed')
    docs=BM25Index.load(bm['outputs'][0]['path'])
    vectors=np.load(Path(dense['directory'])/'vectors.npy',mmap_mode='r',allow_pickle=False)
    qnp=np.load(query['vectors']['path'],allow_pickle=False)
    qvectors=dict(zip(qnp['query_ids'].tolist(),qnp['vectors']))
    ids=[x['doc_id'] for x in docs.chunks]
    if ids!=dense['config']['paper_ids']:raise ValueError('Dense/document BM25 row alignment changed')
    output=run/'S3_DOCUMENT_DENSE_CANDIDATES.jsonl'
    if output.exists():raise ValueError('Preserve existing document audit')
    summaries={}
    with output.open('x',encoding='utf8',newline='\n') as f:
        for split in ('D_search','D_confirm'):
            rows=read_split(run,split,purpose='inference');sums={m:{k:0 for k in (1,5,10)} for m in ('bm25','dense','rrf_half')}
            for q in rows:
                target=q['scope']['doc_ids'][0]
                sparse_candidates=docs.search(q['question'],k=10)
                sparse_ids=[x['doc_id'] for x in sparse_candidates]
                scores=vectors@qvectors[q['query_id']]
                ranking=sorted(range(len(ids)),key=lambda i:(-scores[i],ids[i]))[:10]
                dense_ids=[ids[i] for i in ranking]
                fusion={}
                for ranking_ids in (sparse_ids,dense_ids):
                    for rank,paper in enumerate(ranking_ids,1):fusion[paper]=fusion.get(paper,0)+.5/(60+rank)
                fused=sorted(fusion,key=lambda x:(-fusion[x],x))[:10]
                for method,candidates in [('bm25',sparse_ids),('dense',dense_ids),('rrf_half',fused)]:
                    rank=next((i+1 for i,x in enumerate(candidates) if x==target),None)
                    for k in sums[method]:sums[method][k]+=int(rank is not None and rank<=k)
                    f.write(canonical_json({'query_id':q['query_id'],'dataset_role':split,'condition':'RAW_GLOBAL_DIAGNOSTIC_UNRESOLVED',
                        'method':method,'question':q['question'],'candidate_document_ids':candidates,'visible_selected_paper_rank':rank,
                        'target_source':'original user-scope paper, used only by evaluator; not provided to candidate lookup',
                        'human_review_status':'NOT_REVIEWED'})+'\n')
            summaries[split]={'n':len(rows),'diagnostic_object_match':{m:{str(k):v/len(rows) for k,v in s.items()} for m,s in sums.items()}}
    result={'status':'PASS_REAL_DOCUMENT_DENSE_FUSION_AUDIT','groups':summaries,'candidate_depth':10,'fusion_alpha':.5,
        'source_dense':artifact(dense_path),'source_query_vectors':artifact(query_path),'records':artifact(output),
        'new_model_calls':0,'human_review_status':'NOT_REVIEWED',
        'selection_policy':'fixed alpha .5 for document stage; this unresolved-paper diagnostic is not a task-quality optimization target',
        'limitations':['No unique object is established in raw generic questions; matching dataset paper is only diagnostic.',
            'PAPER_GIVEN deployment bypasses document lookup. All full-corpus chunk bypasses remain required.']}
    atomic_json(run/'S3_DOCUMENT_DENSE_CANDIDATES.json',result);print(canonical_json(result));return 0


if __name__=='__main__':raise SystemExit(main())
