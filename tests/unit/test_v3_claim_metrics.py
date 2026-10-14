from ragscope.v3.claim_metrics import (bounds, proposition, reference_units, retained_failure_metrics,
                                    score_claims, union_relation)


def test_qa_completion_never_needs_reference_and_keeps_negation():
    assert proposition('Do they use Delta?', 'No')['text'] == 'They do not use Delta.'
    assert proposition('How many samples did the experiment use?', '128')['text'] == 'The experiment did use 128 samples.'
    assert not proposition('Did they not use Delta?', 'Yes')['eligible']
    assert not proposition('What corpus was used?', 'Delta')['eligible']


def test_unknown_is_a_bound_not_false_or_silently_dropped():
    value = bounds(['SUPPORTED_PROXY', 'UNKNOWN', 'CONTRADICTED_PROXY'])
    assert value['value'] is None and value['n'] == 3
    assert value['lower'] == 1/3 and value['upper'] == 2/3
    assert union_relation(['NOT_ENTAILED_PROXY', 'UNKNOWN']) == 'UNKNOWN'


def test_conditions_are_preserved_and_failure_is_in_quality_denominator():
    text = 'The model improves only on English; it does not improve on French.'
    units = reference_units(text)
    assert len(units) == 1 and units[0]['source_quote'] == text
    assert retained_failure_metrics('ABSTAINED')['claim_f1_proxy'] == 0


def test_reference_agreement_does_not_imply_context_support_or_good_citation():
    def judge(premise, hypothesis):
        if not premise['eligible'] or not hypothesis['eligible']:
            return {'verdict': 'UNKNOWN'}
        return {'verdict': 'SUPPORTED_PROXY' if premise['text'] == hypothesis['text'] else 'NOT_ENTAILED_PROXY'}
    answer = 'The model uses Delta.'
    result = score_claims('What does it use?', answer, reference_units(answer), [answer],
        [{'chunk_id': 'wrong', 'text': 'The model uses Sigma.'}], ['wrong'], judge)
    assert result['claim_f1_proxy'] == 1
    assert result['context_support_precision_proxy']['value'] == 0
    assert result['citation_completeness_proxy']['value'] == 0
