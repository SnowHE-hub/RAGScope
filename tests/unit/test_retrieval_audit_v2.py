"""Hand-calculated cases for the independent artifact audit, no model calls."""
import pytest

from scripts.audit_retrieval_v2 import independently_score, merged_length, quantile


def test_independent_union_removes_overlap_and_clips_to_reference():
    units = {"a": {"doc_id": "d", "canonical_start": 0, "canonical_end": 10},
             "b": {"doc_id": "d", "canonical_start": 20, "canonical_end": 30}}
    query = {"answerability": "answerable", "evidence_reference_sets": [["a", "b"]]}
    spans = [{"doc_id": "d", "unit_id": "a", "start": 2, "end": 8},
             {"doc_id": "d", "unit_id": "a", "start": 5, "end": 12},
             {"doc_id": "d", "unit_id": "b", "start": 20, "end": 25}]
    assert merged_length([(2, 8), (5, 10)]) == 8
    result = independently_score(query, spans, units)
    assert result["continuous_coverage"] == pytest.approx(.65)
    assert result["full_unit_recall"] == result["full_reference_coverage"] == 0


def test_independent_best_reference_is_full_set_then_full_unit_then_continuous():
    units = {name: {"doc_id": "d", "canonical_start": i*20, "canonical_end": i*20+10}
             for i, name in enumerate("abc")}
    query = {"answerability": "answerable", "evidence_reference_sets": [["a", "b"], ["c"]]}
    spans = [{"doc_id": "d", "unit_id": "a", "start": 0, "end": 10},
             {"doc_id": "d", "unit_id": "c", "start": 40, "end": 48}]
    # First reference has one full unit; second only has a partial unit.
    assert independently_score(query, spans, units)["continuous_coverage"] == .5
    spans[-1]["end"] = 50
    assert independently_score(query, spans, units)["full_reference_coverage"] == 1


def test_independent_failure_and_unanswerable_keep_distinct_denominators():
    query = {"answerability": "answerable", "evidence_reference_sets": [["unknown"]]}
    assert set(independently_score(query, [], {}, failed=True).values()) == {0}
    query["answerability"] = "unanswerable"
    assert set(independently_score(query, [], {}, failed=True).values()) == {None}
    assert quantile([0, 10, 20, 30], .5) == 15
    assert quantile([], .95) is None
