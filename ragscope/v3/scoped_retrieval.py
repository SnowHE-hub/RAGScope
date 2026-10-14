"""Exact scoped hybrid ranking from frozen global BM25 scores and real vectors."""
import numpy as np


def rank_scoped(chunks,global_rows,sparse_scores,query_vector,vectors,*,alpha=.5,k=50):
    if alpha not in {.25,.5,.75} or k<1:raise ValueError('Unapproved scoped retrieval bounds')
    if len(chunks)!=len(global_rows) or np.asarray(vectors).shape!=(len(chunks),1024):raise ValueError('Scoped vector/body alignment mismatch')
    if len({c['chunk_id'] for c in chunks})!=len(chunks):raise ValueError('Duplicate scoped chunk IDs')
    if len({c['doc_id'] for c in chunks})>1:raise ValueError('PAPER_GIVEN ranker requires exactly one user-selected paper')
    dense=np.asarray(vectors)@np.asarray(query_vector,dtype=np.float32)
    lexical=np.asarray(sparse_scores)[global_rows]
    sparse_order=sorted(range(len(chunks)),key=lambda i:(-lexical[i],chunks[i]['chunk_id']))[:50]
    dense_order=sorted(range(len(chunks)),key=lambda i:(-dense[i],chunks[i]['chunk_id']))[:50]
    scores={}
    for weight,order in ((1-alpha,sparse_order),(alpha,dense_order)):
        for rank,i in enumerate(order,1):scores[i]=scores.get(i,0)+weight/(60+rank)
    order=sorted(scores,key=lambda i:(-scores[i],chunks[i]['chunk_id']))[:k]
    return [{**chunks[i],'score':scores[i],'bm25_score':float(lexical[i]),'dense_score':float(dense[i])} for i in order]
