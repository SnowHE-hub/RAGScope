"""Offline safety/correctness tests for frozen real-data preparation."""

import json
from copy import deepcopy

import pytest

from ragscope.ingestion.qasper import (convert_question, digest, duplicate_groups,
    freeze_splits, make_document, normalize_text)
from ragscope.ingestion.technical import build_technical_document, CASE_SPECS


def paper(text="A model is evaluated on a held-out corpus."):
    return {"title": "Paper", "abstract": "An abstract.",
            "full_text": [{"section_name": "Methods", "paragraphs": [text]}]}


def annotation(text="A model is evaluated on a held-out corpus.", **updates):
    answer = {"unanswerable": False, "extractive_spans": [], "free_form_answer": "A held-out corpus",
              "yes_no": None, "evidence": [text]}
    answer.update(updates)
    return {"annotation_id": "a1", "answer": answer}


def query(answers=None):
    return {"question_id": "q1", "question": "What data is used?",
            "answers": [annotation()] if answers is None else answers}


def test_stable_source_ids_coordinates_and_unicode():
    doc = make_document("id", paper("Code: x < y\n    print('中文')"), "train")
    again = make_document("id", paper("Code: x < y\n    print('中文')"), "train")
    assert doc == again
    assert "\n    print" in doc["text"]
    for unit in doc["evidence_units"]:
        assert doc["text"][unit["canonical_start"]:unit["canonical_end"]] == unit["text"]
    assert "s0:p0" in doc["evidence_units"][-1]["unit_id"]
    assert normalize_text("e\u0301\r\ncode") == "é\ncode"


def test_whitespace_evidence_match_and_no_chunk_dependency():
    doc = make_document("id", paper(), "dev")
    converted, reasons = convert_question(query([annotation("A model is evaluated\non a held-out corpus.")]), doc)
    assert not reasons
    assert converted["evidence_reference_sets"] == [[doc["evidence_units"][-1]["unit_id"]]]
    assert converted["scope"] == {"kind": "document", "doc_ids": [doc["doc_id"]]}


@pytest.mark.parametrize("kind,updates,expected", [
    ("float", {"evidence": ["FLOAT SELECTED: Figure 1"]}, "float_evidence"),
    ("unmatched", {"evidence": ["missing paragraph"]}, "unmatched_evidence"),
    ("missing", {"evidence": []}, "missing_text_evidence"),
    ("invalid", {"unanswerable": "no"}, "malformed_answerability"),
    ("answer", {"free_form_answer": ""}, "missing_answer"),
])
def test_bad_annotations_are_explicitly_quarantined(kind, updates, expected):
    converted, reasons = convert_question(query([annotation(**updates)]), make_document("id", paper(), "test"))
    assert expected in reasons
    assert expected in converted["metadata"]["quarantine_reasons"]


def test_answerability_conflict_keeps_both_annotations():
    converted, reasons = convert_question(query([annotation(), annotation(unanswerable=True, evidence=[])]),
                                         make_document("id", paper(), "test"))
    assert converted["answerability"] == "conflict"
    assert "answerability_conflict" in reasons
    assert len(converted["metadata"]["annotations"]) == 2


def test_unanswerable_scope_and_missing_annotations():
    doc = make_document("id", paper(), "test")
    converted, reasons = convert_question(query([annotation(unanswerable=True, evidence=[])]), doc)
    assert not reasons and converted["answerability"] == "unanswerable"
    assert converted["answer_refs"] == [] and converted["evidence_reference_sets"] == []
    assert converted["metadata"]["unanswerable_scope"] == "specified_paper_only"
    assert "missing_annotations" in convert_question(query([]), doc)[1]


def test_multiple_references_types_and_disagreements():
    annotations = [annotation(), annotation(free_form_answer="", extractive_spans=["a", "b"]),
                   annotation(free_form_answer="", yes_no=False)]
    converted, reasons = convert_question(query(annotations), make_document("id", paper(), "test"))
    assert not reasons
    assert converted["answer_refs"] == ["A held-out corpus", "a, b", "No"]
    assert "multiple_answer_texts" in converted["annotation_disagreements"]


def test_ambiguous_duplicate_paragraph_does_not_invent_a_location():
    source = paper()
    source["full_text"][0]["paragraphs"] *= 2
    converted, reasons = convert_question(query(), make_document("id", source, "train"))
    assert "ambiguous_evidence_location" in reasons
    assert converted["evidence_reference_sets"] == []


def test_duplicate_cross_split_and_aliases_do_not_leak():
    docs = [make_document("a", paper(), "train"), make_document("b", paper(), "test"),
            make_document("c", {**paper("unique dev"), "title": "Unique"}, "dev")]
    groups = duplicate_groups(iter(docs))
    assert len(groups) == 1 and groups[0]["splits"] == ["test", "train"]
    queries = []
    for i, doc in enumerate(docs):
        q, _ = convert_question({**query([annotation(doc["evidence_units"][-1]["text"])]), "question_id": str(i)}, doc)
        queries.append(q)
    result = freeze_splits(docs, queries, groups, {"train": 1, "dev": 1, "test": 1})
    assert result["counts"] == {"dev": 1}
    assert result["cross_split_quarantined_doc_ids"] == ["qasper:a", "qasper:b"]
    docs[1]["split"] = "train"
    groups = duplicate_groups(docs)
    result = freeze_splits(docs, queries, groups, {"train": 1, "dev": 1, "test": 1})
    assert result["aliases"] == {"qasper:b": "qasper:a"}


def test_seeded_split_selection_is_input_order_independent_and_bounded():
    docs, queries = [], []
    for split in ("train", "dev", "test"):
        for i in range(3):
            doc = make_document(f"{split}{i}", paper(), split)
            q, _ = convert_question({**query(), "question_id": f"{split}{i}"}, doc)
            docs.append(doc)
            queries.append(q)
    targets = {"train": 2, "dev": 1, "test": 2}
    result = freeze_splits(docs, queries, [], targets)
    reverse = freeze_splits(docs[::-1], queries[::-1], [], targets)
    assert result == reverse
    assert result["counts"] == targets
    assert len(result["global_queries"]) == 2
    assert "doc_id" not in result["global_queries"][0]
    assert result["global_queries"][0]["scope"]["kind"] == "global"
    assert len(freeze_splits(docs, queries, [], targets, max_docs=2)["documents"]) == 2
    with pytest.raises(ValueError):
        freeze_splits(docs, queries, [], targets, max_docs=0)


def test_technical_cases_require_unique_source_evidence(monkeypatch):
    monkeypatch.setitem(CASE_SPECS, "sample", [("Question?", "Source evidence", "Answer")])
    doc, cases = build_technical_document("sample", "Title\n=====\n\nSource evidence here.\n\n    x = 1")
    assert cases[0]["metadata"]["authorship"] == "agent_authored_not_human_validated"
    assert not cases[0]["metadata"]["independent_quality_benchmark"]
    start, end = cases[0]["metadata"]["evidence_canonical_span"]
    assert "Source evidence" in doc["text"][start:end]
    with pytest.raises(ValueError):
        build_technical_document("sample", "No matching source")


def test_stream_parser_and_immutable_outputs(tmp_path):
    from scripts.prepare_v2_data import iter_json_object, immutable_json, immutable_jsonl, read_jsonl
    source = tmp_path / "raw.json"
    payload = {"中文": {"text": "x" * 100000}, "two": {"text": "word"}}
    source.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    assert dict(iter_json_object(source)) == payload
    frozen = tmp_path / "frozen.json"
    immutable_json(frozen, {"id": 1})
    immutable_json(frozen, {"id": 1})
    with pytest.raises(ValueError):
        immutable_json(frozen, {"id": 2})
    lines = tmp_path / "rows.jsonl"
    immutable_jsonl(lines, [{"id": 1}])
    immutable_jsonl(lines, [{"id": 1}])
    assert list(read_jsonl(lines)) == [{"id": 1}]
    with pytest.raises(ValueError):
        immutable_jsonl(lines, [{"id": 2}])
