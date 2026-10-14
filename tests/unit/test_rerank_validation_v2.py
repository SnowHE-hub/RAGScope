"""Arithmetic/integrity fixtures only; never real reranker experiment outputs."""
from copy import deepcopy
import pytest
from ragscope.experiments.rerank_validation import validate_windows


def fixture():
    chunks=[{'chunk_id':f'c{i:02}', 'doc_id':'d', 'text':'abcdefgh', 'canonical_start':i*10,
             'canonical_end':i*10+8, 'unit_spans':[]} for i in range(11)]
    raw=[{key:c[key] for key in ('chunk_id','doc_id','canonical_start','canonical_end','unit_spans')} |
         {'rank':i+1,'score':1/(i+1)} for i,c in enumerate(chunks)]
    source={'scope':{'kind':'paper','doc_ids':['d']},'raw_retrieval':raw,'record_hash':'offline_fixture_hash','record_id':'q'}
    windows=[{'chunk_id':c['chunk_id'],'start':0,'end':8,'pair_tokens':12,'score':float(i)} for i,c in enumerate(chunks)]
    selected=[{**raw[i], 'rank':rank, 'hybrid_rank':i+1, 'rerank_score':float(i), 'rerank_windows':[windows[i]]}
              for rank,i in enumerate(range(10,0,-1),1)]
    row={'raw_retrieval':deepcopy(raw),'source_record_hash':source['record_hash'],'source_record_id':'q',
         'candidate_count':11,'scoped_chunk_count':11,'rerank_windows':deepcopy(windows),
         'selected_retrieval':deepcopy(selected),'error':None}
    return row,source,chunks


def test_all_candidates_and_max_ranking_are_verified():
    row,source,chunks=fixture()
    assert validate_windows(row,source,chunks)=={'failed':False,'windows':11,'max_pair_tokens':12}


@pytest.mark.parametrize('change,match',[
    ('missing_unselected','population'),('short_tail','drops'),('bad_budget','budget'),
    ('wrong_order','order'),('wrong_selected_score','score'),('source_hash','identity')])
def test_corrupted_window_arithmetic_cannot_pass(change,match):
    row,source,chunks=fixture()
    if change=='missing_unselected':row['rerank_windows'].pop(0)
    elif change=='short_tail':row['rerank_windows'][0]['end']=7
    elif change=='bad_budget':row['rerank_windows'][0]['pair_tokens']=513
    elif change=='wrong_order':row['selected_retrieval'].reverse()
    elif change=='wrong_selected_score':row['selected_retrieval'][0]['rerank_score']=-100
    elif change=='source_hash':row['source_record_hash']='changed'
    with pytest.raises(ValueError,match=match):validate_windows(row,source,chunks)


def test_gapped_multiwindow_coverage_cannot_pass():
    row,source,chunks=fixture()
    first=row['rerank_windows'].pop(0)
    row['rerank_windows'].extend([{**first,'end':3},{**first,'start':4}])
    with pytest.raises(ValueError,match='gap'):validate_windows(row,source,chunks)


def test_failed_call_keeps_denominator_without_fabricated_scores():
    row,source,chunks=fixture()
    row.update(error={'type':'FixtureFailure'},selected_retrieval=[],rerank_windows=[])
    assert validate_windows(row,source,chunks)=={'failed':True,'windows':0,'max_pair_tokens':None}
