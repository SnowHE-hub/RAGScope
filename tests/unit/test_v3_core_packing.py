from dataclasses import asdict
from ragscope.generation.context import pack_context
from ragscope.v3.core_packing import pack_core_first


def test_neighbors_do_not_displace_later_ranked_core_or_cross_document():
    a={'chunk_id':'a','doc_id':'d','text':'A'*20,'next_chunk_id':'n'}
    b={'chunk_id':'b','doc_id':'d','text':'B'*20,'next_chunk_id':'outside'}
    lookup={'a':a,'b':b,'n':{'chunk_id':'n','doc_id':'d','text':'N'*40},
            'outside':{'chunk_id':'outside','doc_id':'other','text':'X'}}
    scope={'kind':'document','doc_ids':['d']}
    old=asdict(pack_context([a,b],max_tokens=80,scope=scope,expansion='neighbor',chunk_lookup=lookup,token_counter=len))
    new,audit=pack_core_first([a,b],max_tokens=80,scope=scope,chunk_lookup=lookup,token_counter=len)
    assert 'b' not in {p['chunk_id'] for p in old['pieces']}
    assert {'a','b'}<={p['chunk_id'] for p in new.pieces}
    assert all(p['doc_id']=='d' for p in new.pieces)
    assert audit['neighbor_incremental_reference_tokens']<=20
