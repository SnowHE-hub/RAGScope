import numpy as np
import pytest
from rank_bm25 import BM25Okapi
from ragscope.indexing.array_index import tokenize
from ragscope.v3.compressed_index import CompressedBM25,CanonicalIndex


def test_compressed_scores_match_original_bm25_with_repeated_terms_and_negative_idf():
    texts=['common alpha alpha 中文','common beta','common gamma alpha','']
    reference=BM25Okapi([tokenize(t) or ['__empty__'] for t in texts])
    index=CompressedBM25.build(texts)
    for query in ('common','alpha alpha 中文','absent','gamma beta'):
        np.testing.assert_allclose(index.scores(query),reference.get_scores(tokenize(query)),atol=1e-12)


def test_retrieval_keeps_canonical_body_scope_and_rejects_another_vector_space():
    chunks=[{'chunk_id':'a','doc_id':'d','text':'canonical body'}, {'chunk_id':'b','doc_id':'other','text':'other body'}]
    bm25=CompressedBM25.build(['Title: fake metadata\ncanonical body','Title: unrelated\nother body'])
    vectors=np.zeros((2,1024),dtype=np.float32);vectors[:,0]=1
    index=CanonicalIndex(chunks,bm25,vectors,{'embedding':{'id':'space'}})
    result=index.search('metadata',method='hybrid',query_vector=vectors[0],embedding_identity={'id':'space'},scope='d',k=1)
    assert result[0]['text']=='canonical body'
    assert result[0]['doc_id']=='d'
    with pytest.raises(ValueError,match='space mismatch'):
        index.search('body',query_vector=vectors[0],embedding_identity={'id':'wrong'})
