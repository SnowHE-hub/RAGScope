"""Offline harness contract tests; fake clients here are never experiment evidence."""
import copy
import json
from dataclasses import asdict

import pytest

from ragscope.experiments.generation_campaign import (BudgetExhausted, GenerationBudget,
    CountedClient, freeze_campaign, load_campaign, oracle_context, position_contexts,
    run_model, validate_core_matrix)
from ragscope.experiments.ledger import read_records
from ragscope.indexing.identity import artifact
from ragscope.generation.context import pack_context, validate_bundle
from ragscope.generation.llm_client import LocalLLMClient, MockLLMClient

IDENTITY = {"model": "offline-harness-unit-fixture", "revision": "fixture-revision",
            "weight_sha256": "fixture-no-real-weights", "backend_version": "fixture"}
UNIT = {"unit_id": "u", "doc_id": "d", "canonical_start": 0,
        "canonical_end": 6, "text": "source"}


def piece(cid="c", text="source", start=0):
    return {"chunk_id": cid, "doc_id": "d", "text": text,
            "canonical_start": start, "canonical_end": start+len(text),
            "unit_spans": [{"unit_id": "u", "doc_id": "d", "start": start, "end": start+len(text)}]}


def sample(qid="q"):
    return {"query": {"query_id": qid, "question": "Question?", "doc_id": "d", "scope": "d",
        "split": "test", "answerability": "answerable", "answer_refs": ["source"],
        "evidence_reference_sets": [["u"]]},
        "context_bundle": asdict(pack_context([piece()], token_counter=len))}


class OfflineContractClient:
    """Identity satisfies formal guard solely to exercise it in tmp_path tests."""
    real_model = True
    identity = IDENTITY
    last_metadata = {}
    prompt_token_counter = staticmethod(lambda messages: 1)

    def __init__(self, outputs=None):
        self.calls = 0
        self.outputs = list(outputs or [])

    def chat(self, messages, **kwargs):
        self.calls += 1
        output = self.outputs.pop(0) if self.outputs else '{"answer":"source","citations":["c"],"abstain":false}'
        if isinstance(output, Exception):
            raise output
        return output


def campaign(tmp_path, n=2):
    inputs = {f"input{i}": sample(f"q{i}") for i in range(n)}
    return freeze_campaign(tmp_path, {"fixture": IDENTITY}, [{"experiment_id": "E2", "method": "R0",
        "model_ids": ["fixture"], "input_ids": list(inputs)}], inputs)


def test_budget_protects_core_and_is_durable(tmp_path):
    budget = GenerationBudget(tmp_path / "budget.json", limit=6, core_reserved=4)
    for i in range(2):
        assert budget.reserve(purpose="pilot", query_id=str(i), model_identity=IDENTITY)["call_number"] == i+1
    with pytest.raises(BudgetExhausted):
        budget.reserve(purpose="extension", query_id="x", model_identity=IDENTITY)
    for i in range(4):
        budget.reserve(purpose="E2", query_id=str(i), model_identity=IDENTITY)
    with pytest.raises(BudgetExhausted):
        budget.reserve(purpose="E3", query_id="x", model_identity=IDENTITY)
    assert GenerationBudget(budget.path, limit=6, core_reserved=4).state()["reserved_calls"] == 6
    assert len(read_records(budget.path.with_suffix(".calls.jsonl"))) == 6
    with pytest.raises(ValueError):
        GenerationBudget(budget.path, limit=7, core_reserved=4)
    with pytest.raises(ValueError):
        GenerationBudget(tmp_path / "bad", limit=10001)


def test_external_shard_ownership_skips_missing_tasks_and_keeps_original_rows(tmp_path):
    from ragscope.experiments.generation_campaign import iter_record_sources
    inputs = {f'i{i}': sample(f'q{i}') for i in range(2)}
    e3 = {'experiment_id': 'E3', 'method': 'oracle', 'model_ids': ['fixture'], 'input_ids': list(inputs)}
    shard, main = tmp_path/'shard', tmp_path/'main'
    freeze_campaign(shard, {'fixture': IDENTITY}, [e3], inputs)
    budget = GenerationBudget(tmp_path/'shared_budget', limit=30, core_reserved=0)
    run_model(shard, 'fixture', OfflineContractClient(), budget, evidence_units={'u': UNIT}, max_tasks=1)
    before = (shard/'records/E3_fixture.jsonl').read_bytes()
    freeze_campaign(main, {'fixture': IDENTITY}, [e3, {**e3, 'experiment_id': 'E2', 'method': 'R0'}], inputs,
        provenance={'external_campaigns': [{'manifest': artifact(shard/'EXPERIMENT_MANIFEST.json')}]})
    manifest, frozen = load_campaign(main)
    sources = list(iter_record_sources(main, manifest, frozen))
    assert len(sources) == 2 and len(sources[0]['owned_task_ids']) == len(sources[1]['owned_task_ids']) == 2
    client = OfflineContractClient()
    result = run_model(main, 'fixture', client, budget, evidence_units={'u': UNIT})
    assert client.calls == 5 and result['E2']['n_pending'] == 0 and result['E3']['n_pending'] == 1
    assert (shard/'records/E3_fixture.jsonl').read_bytes() == before
    assert not (main/'records/E3_fixture.jsonl').exists()


def test_external_shard_rejects_semantic_drift_and_duplicate_ownership(tmp_path):
    from ragscope.experiments.generation_campaign import iter_record_sources
    inputs = {'i': sample()}
    group = {'experiment_id': 'E3', 'method': 'oracle', 'model_ids': ['fixture'], 'input_ids': ['i']}
    shard = tmp_path/'shard'
    freeze_campaign(shard, {'fixture': IDENTITY}, [group], inputs)
    reference = {'manifest': artifact(shard/'EXPERIMENT_MANIFEST.json')}
    for label, references, seed in [('duplicate', [reference, reference], 42), ('seed', [reference], 43)]:
        main = tmp_path/label
        with pytest.raises(ValueError, match='duplicate|seed'):
            freeze_campaign(main, {'fixture': IDENTITY}, [group], inputs, seed=seed,
                provenance={'external_campaigns': references})
        assert not (main/'EXPERIMENT_MANIFEST.json').exists()


def test_counted_failure_and_no_ticket_after_exhaustion(tmp_path):
    raw = OfflineContractClient([TimeoutError("offline fixture")])
    counted = CountedClient(raw, GenerationBudget(tmp_path / "b", limit=1, core_reserved=0), "E2", "q")
    with pytest.raises(TimeoutError):
        counted.chat([])
    assert counted.last_ticket["call_number"] == 1
    with pytest.raises(BudgetExhausted):
        counted.chat([])
    assert counted.last_ticket is None and raw.calls == 1


def test_frozen_inputs_and_manifest_tamper_rejected(tmp_path):
    manifest = campaign(tmp_path)
    assert campaign(tmp_path) == manifest
    loaded, inputs = load_campaign(tmp_path)
    assert loaded == manifest and len(inputs) == 2
    path = tmp_path / "FROZEN_INPUTS.json"
    data = json.loads(path.read_text())
    data["input0"]["query"]["question"] = "tampered"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="content changed"):
        load_campaign(tmp_path)
    with pytest.raises(ValueError, match="differ"):
        campaign(tmp_path)
    path = tmp_path / "EXPERIMENT_MANIFEST.json"
    manifest["seed"] = 1
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="manifest hash"):
        load_campaign(tmp_path)


def test_freeze_rejects_unsafe_identity_duplicate_query_and_changed_config(tmp_path):
    group = {"experiment_id": "E2", "method": "R0", "model_ids": ["fixture"], "input_ids": ["i"]}
    with pytest.raises(ValueError, match="Unsafe"):
        freeze_campaign(tmp_path, {"../x": IDENTITY}, [], {})
    with pytest.raises(ValueError, match="pinned"):
        freeze_campaign(tmp_path, {"fixture": {}}, [], {})
    with pytest.raises(ValueError, match="Duplicate"):
        freeze_campaign(tmp_path, {"fixture": IDENTITY}, [{**group, "input_ids": ["i", "j"]}], {"i": sample(), "j": sample()})
    campaign(tmp_path)
    with pytest.raises(ValueError, match="Immutable"):
        freeze_campaign(tmp_path, {"fixture": IDENTITY}, [group], {"i": sample()})


def test_blocked_core_conditions_are_explicit_and_keep_full_denominator(tmp_path):
    groups = [{"experiment_id": exp, "method": method, "model_ids": ["unavailable"],
               "status": "BLOCKED", "reason": "unit fixture unavailable", "expected_n": 2}
              for exp, methods in {"E2": ["R0", "R1", "R2", "R3"], "E3": ["oracle", "no_context"]}.items()
              for method in methods]
    manifest = freeze_campaign(tmp_path, {}, groups, {})
    assert validate_core_matrix(manifest, ["q1", "q2"], ["unavailable"])
    bad = copy.deepcopy(manifest)
    bad["groups"][0]["expected_n"] = 1
    with pytest.raises(ValueError, match="denominator"):
        validate_core_matrix(bad, ["q1", "q2"], ["unavailable"])
    with pytest.raises(ValueError, match="Missing"):
        validate_core_matrix({**manifest, "groups": groups[:-1]}, ["q1", "q2"], ["unavailable"])


def test_oracle_contains_only_sources_and_budget_loss_is_visible():
    query = sample()["query"]
    query["answer_refs"] = ["SECRET_ANSWER_NOT_IN_EVIDENCE"]
    bundle = oracle_context(query, {"u": UNIT}, token_counter=len)
    assert bundle.pieces[0]["text"] == "source" and "SECRET" not in json.dumps(asdict(bundle))
    assert not oracle_context({**query, "answerability": "unanswerable"}, {}, token_counter=len).pieces
    truncated = oracle_context(query, {"u": UNIT}, max_tokens=1, token_counter=len)
    assert not truncated.pieces and truncated.truncations
    with pytest.raises(ValueError):
        oracle_context({**query, "evidence_reference_sets": []}, {}, token_counter=len)


def test_position_moves_only_identical_text_and_scope_enforced():
    evidence, noise = [piece()], [piece("n1", "first noise", 20), piece("n2", "last noise", 50)]
    variants = position_contexts(evidence, noise, token_counter=len)
    assert [p["chunk_id"] for p in variants["middle"].pieces] == ["n1", "c", "n2"]
    assert len({b.budget["reference_tokens"] for b in variants.values()}) == 1
    with pytest.raises(ValueError, match="identical"):
        position_contexts(evidence, noise, max_tokens=1, token_counter=len)
    with pytest.raises(ValueError, match="distinct"):
        position_contexts(evidence, evidence, token_counter=len)
    with pytest.raises(ValueError, match="outside"):
        validate_bundle(variants["start"], scope="other")


def test_campaign_resumes_without_duplicates_and_includes_failed_rows(tmp_path):
    campaign(tmp_path)
    budget = GenerationBudget(tmp_path / "budget", limit=20, core_reserved=2)
    # Three warmups, then malformed model response retained as zero-scoring row.
    client = OfflineContractClient(["{}", "{}", "{}", "bad json"])
    results = run_model(tmp_path, "fixture", client, budget, evidence_units={"u": UNIT}, max_tasks=1, token_counter=len)
    assert results["E2"]["n_total"] == 1 and results["E2"]["n_failed"] == 1
    assert results["E2"]["n_pending"] == 1
    results = run_model(tmp_path, "fixture", client, budget, evidence_units={"u": UNIT}, token_counter=len)
    assert results["E2"]["n_total"] == 2 and results["E2"]["n_failed"] == 1
    assert results["E2"]["methods"]["R0"]["token_f1"]["overall"] == .5
    calls = client.calls
    run_model(tmp_path, "fixture", client, budget, evidence_units={"u": UNIT}, token_counter=len)
    assert client.calls == calls == budget.state()["reserved_calls"] == 8
    rows = read_records(tmp_path / "records" / "E2_fixture.jsonl")
    assert len({r["record_id"] for r in rows}) == 2
    assert rows[0]["error"]["kind"] == "parse_error" and rows[0]["scores"]["token_f1"] == 0


def test_budget_stop_does_not_fabricate_formal_results(tmp_path):
    campaign(tmp_path)
    budget = GenerationBudget(tmp_path / "budget", limit=3, core_reserved=0)
    client = OfflineContractClient()
    results = run_model(tmp_path, "fixture", client, budget, evidence_units={"u": UNIT}, token_counter=len)
    assert results["E2"]["n_total"] == 0 and results["E2"]["n_pending"] == 2
    assert results["E2"]["stop_reason"]["kind"] == "generation_budget_exhausted"
    assert client.calls == 3


def test_campaign_rejects_mock_wrong_warmups_and_invalid_evidence_before_calls(tmp_path):
    campaign(tmp_path)
    budget = GenerationBudget(tmp_path / "budget", limit=10, core_reserved=2)
    with pytest.raises(ValueError, match="Real loaded"):
        run_model(tmp_path, "fixture", MockLLMClient(), budget, evidence_units={})
    client = OfflineContractClient()
    with pytest.raises(ValueError, match="three"):
        run_model(tmp_path, "fixture", client, budget, evidence_units={}, warmups=0)
    with pytest.raises(KeyError):
        run_model(tmp_path, "fixture", client, budget, evidence_units={})
    assert budget.state()["reserved_calls"] == client.calls == 0


def test_local_client_reports_actual_backend_token_difference(monkeypatch):
    client = LocalLLMClient("fixture", prompt_token_counter=lambda messages: 10)
    monkeypatch.setattr(client, "_request", lambda path, payload: {"message": {"content": "{}"}, "prompt_eval_count": 12})
    assert client.chat([]) == "{}"
    assert client.last_metadata["prompt_token_delta"] == 2
    assert client.last_metadata["token_count_match"] is False


def test_backend_circuit_break_preserves_unattempted_formal_population(tmp_path):
    campaign(tmp_path)
    budget=GenerationBudget(tmp_path/'budget',limit=10,core_reserved=2)
    client=OfflineContractClient([TimeoutError('offline unavailable')]*3)
    result=run_model(tmp_path,'fixture',client,budget,evidence_units={'u':UNIT},token_counter=len)
    assert result['E2']['n_total']==0 and result['E2']['n_pending']==2
    assert result['E2']['stop_reason']['kind']=='backend_circuit_break'
    assert budget.state()['reserved_calls']==3
    marker=json.loads((tmp_path/'summaries'/'MODEL_fixture.json').read_text())
    assert marker['status']=='INCOMPLETE' and marker['n_pending']==2
