import numpy as np
import pytest
from ragscope.v3.scoped_retrieval import rank_scoped


def test_scoped_ranking_uses_global_row_mapping_and_keeps_source_body():
    chunks=[{'chunk_id':'a','doc_id':'p','text':'source A'},{'chunk_id':'b','doc_id':'p','text':'source B'}]
    vectors=np.zeros((2,1024),dtype=np.float32);vectors[0,0]=1;vectors[1,1]=1
    query=np.zeros(1024,dtype=np.float32);query[1]=1
    results=rank_scoped(chunks,[3,1],np.array([99.,3.,99.,1.]),query,vectors)
    assert results[0]['chunk_id']=='b' and results[0]['bm25_score']==3
    assert results[0]['text']=='source B'
    with pytest.raises(ValueError,match='exactly one'):
        rank_scoped([chunks[0],{**chunks[1],'doc_id':'other'}],[3,1],np.arange(4),query,vectors)
