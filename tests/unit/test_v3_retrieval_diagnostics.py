from ragscope.v3.retrieval_diagnostics import coverage, top2_ranking


def test_overlapping_intervals_do_not_inflate_coverage_and_complete_reference_requires_all_units():
    units = {'a':{'canonical_start':0,'canonical_end':10},'b':{'canonical_start':20,'canonical_end':30}}
    pieces = [{'unit_spans':[{'unit_id':'a','start':-2,'end':7},{'unit_id':'a','start':4,'end':11}]}]
    assert coverage(pieces,[['a','b']],units)=={'continuous_coverage':.5,'full_unit_recall':.5,'full_reference_coverage':0.0}
    assert coverage(pieces,[['a','b'],['a']],units)['full_reference_coverage']==1.0


def test_top2_mean_reuses_identical_scores_and_can_change_window_max_preference():
    candidates = [{'chunk_id':'long'},{'chunk_id':'short'}]
    windows = [{'chunk_id':'long','score':.9},{'chunk_id':'long','score':.1},{'chunk_id':'short','score':.7}]
    assert top2_ranking(candidates,windows)[0]['chunk_id']=='short'
