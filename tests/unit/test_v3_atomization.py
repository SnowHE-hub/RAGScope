import pytest
from ragscope.v3.atomization import validate_units


def test_negation_and_conditions_cannot_be_dropped_or_rewritten():
    text = 'Only on dataset Delta, Alpha did not beat Beta.'
    result = validate_units(text, [{'start': 0, 'end': len(text), 'kind': 'claim'}])
    assert result[0]['source_quote'] == text
    with pytest.raises(ValueError, match='drops original'):
        validate_units(text, [{'start': text.index('Alpha'), 'end': len(text), 'kind': 'claim'}])
    with pytest.raises(ValueError, match='rewritten'):
        validate_units(text, [{'start': 0, 'end': len(text), 'kind': 'claim', 'claim': 'Alpha beat Beta.'}])


def test_unicode_coordinates_overlap_and_boolean_offsets_are_checked():
    text = '样本数为128。'
    assert validate_units(text, [{'start': 0, 'end': len(text), 'kind': 'unclear'}])[0]['source_quote'] == text
    with pytest.raises(ValueError, match='Overlapping'):
        validate_units(text, [{'start': 0, 'end': len(text), 'kind': 'claim'}, {'start': 1, 'end': 2, 'kind': 'claim'}])
    with pytest.raises(ValueError, match='Invalid'):
        validate_units(text, [{'start': False, 'end': len(text), 'kind': 'claim'}])
