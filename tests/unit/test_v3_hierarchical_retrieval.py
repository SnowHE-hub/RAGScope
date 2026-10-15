import numpy as np
from ragscope.v3.hierarchical_retrieval import HierarchicalRetriever,fuse_rankings


class Sparse:
    def __init__(self,values):self.values=np.array(values)
    def scores(self,question):return self.values


def retriever():
    docs=[{'doc_id':'a','title':'A'},{'doc_id':'b','title':'B'}]
    chunks=[{'chunk_id':str(i),'doc_id':pid,'text':f'Fact {i}.'} for i,pid in enumerate(['a','a','b','b'])]
    return HierarchicalRetriever(documents=docs,document_sparse=Sparse([5,0]),document_vectors=np.zeros((2,1024)),
        chunks=chunks,chunk_sparse=Sparse([0,1,100,50]),scoped_vectors={},embedding_identity={'test':'fixture'})


def test_specified_scope_skips_document_lookup_and_never_leaks_global_chunks():
    result=retriever().search('query',scope={'doc_ids':['a']},query_vector=np.zeros(1024),embedding_identity={'test':'fixture'})
    assert not result['document_lookup_performed'] and not result['global_bypass_used']
    assert {c['doc_id'] for c in result['candidates']}=={'a'}


def test_ambiguity_clarifies_but_identifiable_query_keeps_global_sparse_escape():
    engine=retriever();args={'scope':{'kind':'global'},'query_vector':np.zeros(1024),'embedding_identity':{'test':'fixture'}}
    ambiguous=engine.search('they',**args)
    assert ambiguous['action']=='CLARIFY_DOCUMENT' and ambiguous['candidates']==[]
    result=engine.search('Paper title: A',visible_object_identified=True,**args)
    assert result['global_bypass_used'] and '2' in {c['chunk_id'] for c in result['candidates']}


def test_fusion_uses_ranks_and_deduplicates_branch_overlap():
    ranking,scores=fuse_rankings([['a','b'],['b','c']],k=3)
    assert ranking[0]=='b' and len(ranking)==len(set(ranking))==3
    assert scores['b']==.5/62+.5/61
