from ragscope.experiments.error_cases import classify

UNIT = {'unit_id': 'u', 'doc_id': 'd', 'canonical_start': 0, 'canonical_end': 4}
QUERY = {'doc_id': 'd', 'answerability': 'answerable', 'evidence_reference_sets': [['u']]}


def test_missing_document_and_packing_loss_are_distinct():
    row = {'experiment_id': 'E2', 'scores': {'context_evidence': {'full_reference_coverage': 0}},
           'citations': ['c'], 'raw_retrieval': []}
    result = classify(row, QUERY, {'u': UNIT})
    assert [f['category'] for f in result['flags']] == ['document_not_retrieved']
    row['raw_retrieval'] = [{'doc_id': 'd', 'unit_spans': [dict(UNIT, start=0, end=4)]}]
    result = classify(row, QUERY, {'u': UNIT})
    assert [f['category'] for f in result['flags']] == ['packing_evidence_loss']


def test_semantic_error_and_noise_are_uncertain_not_fabricated_causes():
    row = {'experiment_id': 'E3', 'scores': {'token_f1': 0, 'context_evidence': {'full_reference_coverage': 1}},
           'citations': ['c'], 'context_bundle': {'pieces': [{'text': 'abcd'}]}}
    flags = classify(row, QUERY, {'u': UNIT})['flags']
    assert len(flags) == 2 and all(f['certainty'] == 'uncertain' for f in flags)
    assert not any(f['category'] == 'document_not_retrieved' for f in flags)


def test_parse_failure_is_not_a_missing_citation_quality_claim():
    row = {'experiment_id': 'E3', 'error': {'kind': 'parse_error'}, 'scores': {}, 'citations': []}
    assert [f['category'] for f in classify(row, QUERY, {'u': UNIT})['flags']] == ['json_error']


def test_raw50_to_top10_loss_is_not_mislabelled_token_budget_loss():
    relevant = {'doc_id': 'd', 'unit_spans': [dict(UNIT, start=0, end=4)]}
    row = {'experiment_id': 'E2', 'scores': {'context_evidence': {'full_reference_coverage': 0}},
           'citations': ['c'], 'raw_retrieval': [relevant]+[{'doc_id': 'd', 'unit_spans': []}]*49,
           'selected_retrieval': [{'doc_id': 'd', 'unit_spans': []}]*10}
    flags = classify(row, QUERY, {'u': UNIT})['flags']
    assert not any(f['category'] == 'packing_evidence_loss' for f in flags)
    assert flags[0]['certainty'] == 'uncertain'
