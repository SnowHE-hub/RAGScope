"""Scoped bypass or document fusion plus a real global sparse chunk escape path."""
from collections import defaultdict
import numpy as np
from ragscope.v3.scoped_retrieval import rank_scoped


def fuse_rankings(rankings,*,k,weights=None):
    weights=weights or [1/len(rankings)]*len(rankings)
    if len(weights)!=len(rankings) or any(w<0 for w in weights):raise ValueError('Invalid rank weights')
    scores=defaultdict(float)
    for weight,ranking in zip(weights,rankings):
        if len(set(ranking))!=len(ranking):raise ValueError('Rank list contains duplicates')
        for rank,key in enumerate(ranking,1):scores[key]+=weight/(60+rank)
    return sorted(scores,key=lambda key:(-scores[key],key))[:k],dict(scores)


class HierarchicalRetriever:
    def __init__(self,*,documents,document_sparse,document_vectors,chunks,chunk_sparse,scoped_vectors,embedding_identity):
        self.documents,self.document_sparse,self.document_vectors=documents,document_sparse,np.asarray(document_vectors)
        self.chunks,self.chunk_sparse,self.scoped_vectors=chunks,chunk_sparse,scoped_vectors
        self.embedding_identity=embedding_identity;self.by_doc=defaultdict(list)
        if self.document_vectors.shape!=(len(documents),1024):raise ValueError('Document vector alignment mismatch')
        if len({d['doc_id'] for d in documents})!=len(documents) or len({c['chunk_id'] for c in chunks})!=len(chunks):raise ValueError('Duplicate source IDs')
        for i,c in enumerate(chunks):self.by_doc[c['doc_id']].append(i)

    def _vector(self,query_vector,identity):
        if identity!=self.embedding_identity:raise ValueError('Query embedding space mismatch')
        vector=np.asarray(query_vector,dtype=np.float32)
        if vector.shape!=(1024,) or not np.isfinite(vector).all():raise ValueError('Invalid query vector')
        return vector

    def _paper_chunks(self,pid,sparse_scores,query_vector,k):
        ids=self.by_doc.get(pid,[]);chunks=[self.chunks[i] for i in ids]
        vectors=self.scoped_vectors.get(pid)
        if vectors is not None:return rank_scoped(chunks,ids,sparse_scores,query_vector,vectors,k=k)
        order=sorted(ids,key=lambda i:(-sparse_scores[i],self.chunks[i]['chunk_id']))[:k]
        return [{**self.chunks[i],'score':float(sparse_scores[i]),'dense_score':None,'retrieval_branch':'SCOPED_SPARSE_FULL_DENSE_BLOCKED'} for i in order]

    def search(self,question,*,scope,query_vector,embedding_identity,visible_object_identified=False,k=50):
        vector=self._vector(query_vector,embedding_identity)
        specified=scope.get('doc_ids',[]) if isinstance(scope,dict) else []
        lexical=self.chunk_sparse.scores(question)
        if specified:
            if any(pid not in self.by_doc for pid in specified):raise ValueError('Specified document is not indexed')
            groups=[self._paper_chunks(pid,lexical,vector,min(k,50)) for pid in specified]
            output=[c for round_index in range(max(map(len,groups),default=0)) for group in groups for c in group[round_index:round_index+1]][:k]
            return {'action':'RETRIEVE','candidates':output,'document_candidates':[],
                'document_lookup_performed':False,'scope_policy':'USER_SPECIFIED_DOCUMENTS','global_bypass_used':False,
                'full_chunk_dense_status':'BLOCKED','human_review_status':'NOT_REVIEWED'}
        sparse=self.document_sparse.scores(question);dense=self.document_vectors@vector
        sparse_ids=sorted(range(len(self.documents)),key=lambda i:(-sparse[i],self.documents[i]['doc_id']))[:10]
        dense_ids=sorted(range(len(self.documents)),key=lambda i:(-dense[i],self.documents[i]['doc_id']))[:10]
        paper_ids,scores=fuse_rankings([[self.documents[i]['doc_id'] for i in sparse_ids],
            [self.documents[i]['doc_id'] for i in dense_ids]],k=5)
        titles={d['doc_id']:d['title'] for d in self.documents}
        docs=[{'doc_id':pid,'title':titles[pid],'rank_fusion_score':scores[pid]} for pid in paper_ids]
        if not visible_object_identified:
            return {'action':'CLARIFY_DOCUMENT','message':'请指定要分析的论文或文档。','document_candidates':docs,
                'candidates':[],'document_lookup_performed':True,'scope_policy':'AMBIGUOUS_NO_GOLD_OBJECT_INJECTED',
                'global_bypass_used':False,'human_review_status':'NOT_REVIEWED'}
        groups=[self._paper_chunks(pid,lexical,vector,10) for pid in paper_ids]
        hierarchical=[c['chunk_id'] for rank in range(10) for group in groups for c in group[rank:rank+1]]
        global_rows=sorted(range(len(self.chunks)),key=lambda i:(-lexical[i],self.chunks[i]['chunk_id']))[:50]
        bypass=[self.chunks[i]['chunk_id'] for i in global_rows]
        merged,weights=fuse_rankings([hierarchical,bypass],k=k)
        chunks={c['chunk_id']:c for c in self.chunks}
        return {'action':'RETRIEVE','candidates':[{**chunks[cid],'score':weights[cid]} for cid in merged],
            'document_candidates':docs,'document_lookup_performed':True,'scope_policy':'VISIBLE_OBJECT_IDENTIFIED',
            'global_bypass_used':True,'global_bypass':'REAL_GLOBAL_CHUNK_BM25','full_chunk_dense_status':'BLOCKED',
            'human_review_status':'NOT_REVIEWED'}
