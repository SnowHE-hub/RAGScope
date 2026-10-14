import pytest
from ragscope.v3.judge_protocol import context_input, validate_sufficiency


def test_sufficiency_input_rejects_answers_references_and_answerability():
    safe = {'question': 'What dataset?', 'context_pieces': [{'text': 'They used Delta.'}]}
    assert context_input(safe)['passage'] == 'They used Delta.'
    for field in ('answer', 'references', 'document_answerable', 'model_abstained'):
        with pytest.raises(ValueError):
            context_input({**safe, field: 'hidden'})


def test_sufficiency_requires_visible_quotes_and_keeps_uncertainty():
    valid = {'sufficient': 'YES', 'confidence': .95, 'quotes': ['Delta'], 'reason': 'stated'}
    assert validate_sufficiency(valid, 'They used Delta.') == 'YES'
    assert validate_sufficiency({**valid, 'confidence': .6}, 'They used Delta.') == 'UNKNOWN'
    with pytest.raises(ValueError):
        validate_sufficiency({**valid, 'quotes': ['Sigma']}, 'They used Delta.')
    with pytest.raises(ValueError):
        validate_sufficiency({**valid, 'quotes': []}, 'They used Delta.')
