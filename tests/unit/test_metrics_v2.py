import json
import math

import pytest

from ragscope.evaluation.evidence import evidence_metrics, context_statistics, merge_intervals
from ragscope.evaluation.generation_metrics import exact_match, token_f1, rouge_l, answer_reference_texts
from ragscope.evaluation.retrieval_metrics import recall_at_k, hit_rate_at_k, precision_at_k, ndcg_at_k, mrr
from ragscope.evaluation.v2 import score_generation, summarize_records, cluster_bootstrap, paired_cluster_bootstrap, holm_adjust


def span(unit="u", start=0, end=10, doc="d"):
    return {"unit_id": unit, "doc_id": doc, "start": start, "end": end}


def test_multireference_and_declared_normalization():
    assert exact_match("The Answer!", ["wrong", "answer"]) == 1
    assert token_f1("blue blue red", ["blue red red"]) == pytest.approx(2/3)
    assert rouge_l("one two three", ["one three two", "one two three"]) == 1
    assert rouge_l("one two three", "one three two") == pytest.approx(2/3)
    for score in [exact_match, token_f1, rouge_l]:
        assert score("", "") == 1 and score("a dog", "") == 0 and score("x", []) == 0
    assert answer_reference_texts([{"yes_no": True}, {"yes_no": False}, {"free_form_answer": "text"},
        {"extractive_spans": ["one", "two"]}, {"text": "alternative"}, {"answer": "last"},
        {"unanswerable": True}, "plain"]) == ["yes", "no", "text", "one, two", "alternative", "last", "plain"]


def test_continuous_full_unit_and_full_reference_are_distinct():
    references = [[span(), span("v", 10, 20)]]
    partial = evidence_metrics(references, [span(end=5), span("v", 10, 20)])
    assert partial["continuous_coverage"] == .75
    assert partial["full_unit_recall"] == .5
    assert partial["full_reference_coverage"] == 0
    complete = evidence_metrics(references, [span(end=8), span(start=6), span("v", 10, 20)])
    assert complete["full_reference_coverage"] == 1
    assert complete["continuous_coverage"] == 1  # overlap cannot count twice


def test_best_complete_legal_set_and_unit_lookup():
    units = {"u": span(), "v": span("v", 10, 20)}
    result = evidence_metrics([["u", "v"], ["u"]], [span()], evidence_units=units)
    assert result["best_reference_index"] == 1 and result["full_reference_coverage"] == 1
    result = evidence_metrics([["u"]], [span()], evidence_units=list(units.values()))
    assert result["continuous_coverage"] == 1
    with pytest.raises(KeyError):
        evidence_metrics([["missing"]], [], evidence_units=units)


def test_evidence_not_applicable_and_invalid_coordinates():
    assert evidence_metrics([[span()]], [], "unanswerable")["continuous_coverage"] is None
    assert evidence_metrics([], [])["full_unit_recall"] is None
    assert evidence_metrics([[]], [])["full_unit_recall"] is None
    with pytest.raises(ValueError):
        evidence_metrics([[span(end=0)]], [])
    with pytest.raises(ValueError):
        merge_intervals([(2, 1)])
    assert merge_intervals([(0, 0), (0, 5), (5, 10)]) == [(0, 10)]
    assert evidence_metrics([[span()]], [span(doc="other")])["continuous_coverage"] == 0


def test_context_statistics_counts_overlap():
    result = context_statistics({"pieces": [{}, {}], "unit_spans": [span(), span(start=5, end=15)],
                                 "budget": {"reference_tokens": 10, "max_reference_tokens": 20}})
    assert result == {"piece_count": 2, "duplicate_ratio": .25, "budget_use": .5, "reference_tokens": 10}


def result(answer="correct", error=None, abstain=False):
    return {"answer": answer, "citations": ["c"], "parsed_output": {"answer": answer, "citations": ["c"]},
            "error": error, "abstain": abstain, "context_bundle": {"pieces": [
            {"chunk_id": "c", "unit_spans": [span()]}], "unit_spans": [span()]}}


def test_generation_scoring_failed_answer_wrong_abstain_and_citations():
    query = {"answerability": "answerable", "answer_refs": ["wrong", "correct"], "evidence_reference_sets": [[span()]]}
    good = score_generation(query, result())
    assert good["token_f1"] == 1 and good["citation_visible_rate"] == 1
    assert good["citation_evidence"]["full_reference_coverage"] == 1
    failed = score_generation(query, result(error={"kind": "timeout"}))
    assert failed["token_f1"] == 0 and failed["json_valid"] == 0
    abstain = score_generation(query, result(abstain=True))
    assert abstain["wrong_abstention"] == 1 and abstain["token_f1"] == 0
    unanswerable = score_generation({**query, "answerability": "unanswerable"}, result(abstain=True))
    assert unanswerable["token_f1"] is None and unanswerable["context_evidence"]["continuous_coverage"] is None


def test_failure_denominator_and_abstention_summary():
    rows = [{"scores": {"token_f1": 1, "gold_unanswerable": False}, "latency_ms": 1},
            {"scores": {"token_f1": 0, "failed": True, "gold_unanswerable": False}, "latency_ms": 3},
            {"scores": {"token_f1": None, "gold_unanswerable": True, "predicted_abstain": True}},
            {"scores": {"token_f1": None, "gold_unanswerable": True, "failed": True}}]
    summary = summarize_records(rows)
    assert summary["n_total"] == 4 and summary["n_failed"] == 2
    assert summary["token_f1"]["overall"] == .5 and summary["token_f1"]["success_conditional"] == 1
    assert summary["token_f1"]["n_applicable"] == 2
    assert summary["abstention"]["recall"] == .5 and summary["latency_ms"]["p50"] == 2
    assert summarize_records([])["latency_ms"]["p50"] is None


def test_document_metrics_do_not_reward_duplicate_chunks():
    docs = ["d", "d", "d"]
    assert recall_at_k(docs, {"d"}, 3) == hit_rate_at_k(docs, {"d"}, 3) == 1
    assert precision_at_k(docs, {"d"}, 3) == pytest.approx(1/3)
    assert ndcg_at_k(docs, {"d"}, 3) == 1
    assert mrr(["other", "d"], {"d"}) == .5
    assert recall_at_k(["d"], {"d", "e"}, 1) == .5
    assert hit_rate_at_k(["d"], {"d", "e"}, 1) == 1
    for score in (recall_at_k, precision_at_k, hit_rate_at_k, ndcg_at_k):
        assert score([], [], 0) == 0 and score(["d"], {"d"}, -1) == 0


def test_paper_cluster_bootstrap_and_paired_alignment():
    rows = [{"query_id": "a", "doc_id": "paper1", "f1": 0.0},
            {"query_id": "b", "doc_id": "paper1", "f1": 0.0},
            {"query_id": "c", "doc_id": "paper2", "f1": 1.0}]
    bootstrap = cluster_bootstrap(rows, "f1")
    assert bootstrap["n_resamples"] == 2000 and bootstrap["n_clusters"] == 2
    assert bootstrap["mean"] == pytest.approx(1/3) and bootstrap["low"] == 0 and bootstrap["high"] == 1
    assert bootstrap == cluster_bootstrap(rows, "f1")
    paired = paired_cluster_bootstrap(rows, rows, "f1")
    assert paired["mean"] == paired["low"] == paired["high"] == 0
    with pytest.raises(ValueError):
        paired_cluster_bootstrap(rows, rows[:1], "f1")
    with pytest.raises(ValueError):
        paired_cluster_bootstrap(rows+rows[:1], rows, "f1")
    assert cluster_bootstrap([], "f1")["n_clusters"] == 0
    with pytest.raises(ValueError):
        cluster_bootstrap(rows, "f1", n_resamples=0)
    assert holm_adjust([.01, .03, .04]) == [.03, .06, .06]


def test_benchmark_retains_all_failures_and_reranks_50_to_10(tmp_path):
    from ragscope.evaluation.benchmark import run_retrieval_benchmark
    class Retriever:
        depths = []
        def retrieve(self, question, k):
            self.depths.append(k)
            if question == "bad":
                raise RuntimeError("intentional offline failure")
            return [{"doc_id": "d"}]*k
    class Reranker:
        def rerank(self, question, candidates, top_k):
            assert len(candidates) == 50 and top_k == 10
            return candidates[:top_k]
    retriever = Retriever()
    path = tmp_path / "records.jsonl"
    frame = run_retrieval_benchmark([{"name": "test", "retriever": retriever, "use_reranker": True, "reranker": Reranker()}],
        [{"question": "good", "gold_doc_id": "d"}, {"question": "bad", "gold_doc_id": "d"}], output_path=path)
    assert frame.iloc[0]["n_total"] == 2 and frame.iloc[0]["n_failed"] == 1
    assert frame.iloc[0]["Recall@10"] == .5 and frame.iloc[0]["Recall@10_success_conditional"] == 1
    assert len(frame.attrs["records"]) == len(path.read_text().splitlines()) == 2
    assert retriever.depths == [50, 50]
    all_failed = run_retrieval_benchmark([{"name": "bad", "retriever": retriever}], [{"question": "bad", "gold_doc_id": "d"}])
    assert len(all_failed) == 1 and all_failed.iloc[0]["n_failed"] == 1 and all_failed.iloc[0]["Recall@10"] == 0
