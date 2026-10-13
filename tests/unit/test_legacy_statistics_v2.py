import pytest
from ragscope.experiments.legacy_statistics import article_mapping


def fixture():
    queries = [{"query_id": "q1", "gold_doc_id": "p1", "question": "Which?", "answer": "A"}]
    docs = [{"doc_id": "p1", "title": "Article", "text": "A is here."}]
    raw = {"data": [{"title": "Article", "paragraphs": [{"context": "A is here.", "qas": [
        {"id": "official1", "question": "Which?", "answers": [{"text": "A"}], "is_impossible": False}]}]}]}
    return queries, docs, raw


def test_original_article_mapping_preserves_source_identity():
    queries, docs, raw = fixture()
    second = {"query_id": "q2", "gold_doc_id": "p2", "question": "Where?", "answer": "B"}
    queries.append(second)
    docs.append({"doc_id": "p2", "title": "Article", "text": "B elsewhere."})
    raw["data"][0]["paragraphs"].append({"context": "B elsewhere.", "qas": [
        {"id": "official2", "question": "Where?", "answers": [{"text": "B"}]}]})
    rows = article_mapping(queries, docs, raw)
    assert rows[0]["cluster_id"] == rows[1]["cluster_id"]
    assert rows[0]["paragraph_doc_id"] != rows[1]["paragraph_doc_id"]
    assert rows[0]["original_qa_ids"] == ["official1"]


def test_article_mapping_reproduces_only_archived_boundary_strip():
    queries, docs, raw = fixture()
    raw["data"][0]["paragraphs"][0]["context"] = "\nA is here.  "
    assert len(article_mapping(queries, docs, raw)) == 1
    raw["data"][0]["paragraphs"][0]["context"] = "A  is here."
    with pytest.raises(ValueError):
        article_mapping(queries, docs, raw)


@pytest.mark.parametrize("change", ["missing_doc", "wrong_title", "wrong_text", "wrong_question", "wrong_answer", "ambiguous_article"])
def test_article_mapping_has_no_unverified_paragraph_fallback(change):
    queries, docs, raw = fixture()
    if change == "missing_doc": docs.clear()
    if change == "wrong_title": docs[0]["title"] = "Other"
    if change == "wrong_text": docs[0]["text"] += "changed"
    if change == "wrong_question": queries[0]["question"] = "Who?"
    if change == "wrong_answer": queries[0]["answer"] = "C"
    if change == "ambiguous_article": raw["data"].append(raw["data"][0])
    with pytest.raises(ValueError):
        article_mapping(queries, docs, raw)
