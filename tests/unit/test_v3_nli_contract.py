import pytest
from ragscope.v3.nli import aggregate_windows, label_map


def test_relation_indices_are_taken_from_model_metadata():
    assert label_map({'0': 'contradiction', '1': 'entailment', '2': 'neutral'})[1] == 'entailment'
    with pytest.raises(ValueError, match='explicit'):
        label_map({'0': 'LABEL_0', '1': 'LABEL_1', '2': 'LABEL_2'})


def test_conflicting_windows_and_jointly_uncertain_evidence_remain_unknown():
    ent = {'probabilities': {'entailment': 0.9, 'neutral': 0.05, 'contradiction': 0.05}}
    contra = {'probabilities': {'entailment': 0.05, 'neutral': 0.05, 'contradiction': 0.9}}
    neutral = {'probabilities': {'entailment': 0.05, 'neutral': 0.9, 'contradiction': 0.05}}
    assert aggregate_windows([ent, contra]) == 'UNKNOWN'
    assert aggregate_windows([neutral, neutral]) == 'UNKNOWN'
    assert aggregate_windows([neutral]) == 'NOT_ENTAILED_PROXY'
    assert aggregate_windows([ent]) == 'SUPPORTED_PROXY'
