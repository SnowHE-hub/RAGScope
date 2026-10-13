from copy import deepcopy
import pytest
from ragscope.ingestion.squad_v2 import convert_squad, select_squad_short


def payload():
    return {"data": [{"title": "Article", "paragraphs": [{"context": "Alpha beta gamma.", "qas": [
        {"id": "a", "question": "What follows Alpha?", "is_impossible": False, "answers": [{"text": "beta", "answer_start": 6}]},
        {"id": "b", "question": "Who?", "is_impossible": True, "answers": []}]}]}]}


def test_raw_offsets_and_source_article_clustering():
    documents, queries, quarantine = convert_squad(payload(), lambda t: len(t.split()))
    assert not quarantine
    assert queries[0]["answer_refs"] == ["beta"]
    assert queries[0]["metadata"]["article_cluster_id"] == documents[0]["metadata"]["article_cluster_id"]
    span = queries[0]["metadata"]["answer_annotations"][0]
    assert documents[0]["text"][span["canonical_start"]:span["canonical_end"]] == "beta"
    assert "doc_id" not in queries[0] and "doc_ids" not in queries[0]["scope"]
    assert select_squad_short(queries) == [queries[0]]
    assert queries[1]["metadata"]["unanswerable_scope"] == "original_paragraph_only"


def test_bad_offsets_excluded_before_freeze_without_deleting_source():
    data = payload()
    data["data"][0]["paragraphs"][0]["qas"][0]["answers"][0]["answer_start"] = 0
    documents, queries, quarantine = convert_squad(data, len)
    assert len(documents) == 1 and len(queries) == 2 and len(quarantine) == 1
    assert select_squad_short(queries) == []
    with pytest.raises(ValueError):
        select_squad_short(queries, 0)


def test_length_boundary_and_deterministic_subset():
    _, queries, _ = convert_squad(payload(), lambda t: 512)
    assert len(select_squad_short(queries)) == 1
    queries[0]["metadata"]["reference_n_tokens"] = 513
    assert select_squad_short(queries) == []
