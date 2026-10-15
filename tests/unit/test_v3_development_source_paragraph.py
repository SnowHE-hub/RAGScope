import pytest
from scripts.v3.repair_development_task_paragraphs import source_paragraph


def test_full_source_paragraph_preserves_negation_and_conditions():
    source='Intro.\n\nAccuracy improved only with labeled data; it did not improve without labels.\n\nEnd.'
    assert source_paragraph(source,'Accuracy improved only with labeled data')=='Accuracy improved only with labeled data; it did not improve without labels.'
    with pytest.raises(ValueError):source_paragraph(source,'Accuracy improved without labels.')
