import pytest
from ragscope.v3.task_references import validate_reference


def fixture():
    sentences = ['The experiment did not use Delta.', 'Method Alpha scored 14.',
                 'Method Beta scored 12.', 'Higher scores are better.']
    paper = {'source_id': 'p', 'title': 'Test paper', 'text': ' '.join(sentences)}
    task = {'reference_answer': 'A synthetic reference.', 'entity_a': 'Alpha', 'entity_b': 'Beta',
            'keypoints': [{'quote': s, 'facet': 'comparison'} for s in sentences]}
    return paper, task


def test_source_coordinates_and_synthetic_disclosure_are_preserved():
    paper, task = fixture()
    public, gold = validate_reference(paper, task, 'compare')
    assert 'Alpha' in public['question'] and 'Beta' in public['question']
    assert gold['human_review_status'] == 'NOT_REVIEWED'
    assert all(paper['text'][p['canonical_start']:p['canonical_end']] == p['text'] for p in gold['source_keypoints'])


def test_exact_substring_that_drops_negation_is_not_a_valid_keypoint():
    paper, task = fixture()
    task['keypoints'][0]['quote'] = 'use Delta.'
    with pytest.raises(ValueError, match='clipped clause'):
        validate_reference(paper, task, 'compare')
    paper, task = fixture()
    task['entity_b'] = 'Gamma'
    with pytest.raises(ValueError, match='paper entities'):
        validate_reference(paper, task, 'compare')
