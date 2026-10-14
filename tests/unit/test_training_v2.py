import json
from pathlib import Path
from types import SimpleNamespace
from dataclasses import asdict

import pytest

from ragscope.training.checkpoint import CheckpointStore
from ragscope.training.data import encode_assistant_only, prepare_examples
from ragscope.training.evaluate import validate_comparison_inputs
from ragscope.generation.context import pack_context
from ragscope.training.run import validate_config, verify_lease, discover_attention_targets, resource_guard


class Tokenizer:
    """Explicit offline character-token test double, never used in experiments."""
    def encode(self, text, **kwargs):
        return [ord(char) for char in text]
    def apply_chat_template(self, messages, add_generation_prompt, **kwargs):
        text = "".join(f"<{row['role']}>"+row["content"]+"</end>" for row in messages)
        if add_generation_prompt:
            text += "<assistant>"
        return self.encode(text)


def config():
    return json.loads((Path(__file__).parents[2]/"configs/v2/lora.json").read_text(encoding="utf-8-sig"))


def test_labels_supervise_only_assistant_and_do_not_truncate():
    messages = [{"role": "user", "content": "evidence"}, {"role": "assistant", "content": "answer"}]
    row = encode_assistant_only(Tokenizer(), messages)
    n = row["prompt_tokens"]
    assert row["labels"][:n] == [-100]*n
    assert row["labels"][n:] == row["input_ids"][n:]
    assert "answer" in "".join(chr(token) for token in row["labels"][n:])
    with pytest.raises(ValueError, match="no truncation"):
        encode_assistant_only(Tokenizer(), messages, 5)


def test_transformers5_default_dict_is_explicitly_disabled():
    class Transformers5Contract(Tokenizer):
        def apply_chat_template(self,messages,add_generation_prompt,return_dict=True,**kwargs):
            result=super().apply_chat_template(messages,add_generation_prompt,**kwargs)
            return {'input_ids':result} if return_dict else result
    row=encode_assistant_only(Transformers5Contract(),[{'role':'user','content':'source'},{'role':'assistant','content':'answer'}])
    assert row['supervised_tokens']>0


def test_template_boundary_mismatch_and_missing_supervision_rejected():
    class BadTokenizer(Tokenizer):
        def apply_chat_template(self, messages, add_generation_prompt, **kwargs):
            return [1, 2] if add_generation_prompt else [2, 3]
    with pytest.raises(ValueError, match="prefix"):
        encode_assistant_only(BadTokenizer(), [{"role": "assistant", "content": "x"}])
    with pytest.raises(ValueError):
        encode_assistant_only(Tokenizer(), [{"role": "user", "content": "x"}])
    with pytest.raises(ValueError):
        encode_assistant_only(Tokenizer(), [{"role": "assistant", "content": "x"}, {"role": "assistant", "content": "y"}])


def document(split="train"):
    return {"doc_id": "d", "split": split, "title": "", "text": "A fact.", "evidence_units": [
        {"unit_id": "u", "doc_id": "d", "canonical_start": 0, "canonical_end": 7, "text": "A fact."}]}


def query(split="train", qid="q"):
    return {"query_id": qid, "doc_id": "d", "split": split, "question": "What?", "answerability": "answerable",
            "answer_refs": ["A fact."], "evidence_reference_sets": [["u"]]}


def test_prepare_preserves_evidence_and_isolates_test_and_conflict():
    records, rejects = prepare_examples([query(), query("test", "test"), {**query(qid="conflict"), "answerability": "conflict"}],
        [document()], Tokenizer(), {"d"})
    assert len(records) == 1 and len(rejects) == 2
    assert "A fact." in records[0]["messages"][1]["content"]
    assert records[0]["split"] == "train" and records[0]["context_hash"]
    assert json.loads(records[0]["messages"][-1]["content"])["citations"] == ["train:u"]
    negative, rejected = prepare_examples([{**query(), "answerability": "unanswerable", "answer_refs": []}],
        [document()], Tokenizer(), {"d"})
    assert not rejected and negative[0]["evidence_policy"] == "first_real_paragraph"
    assert json.loads(negative[0]["messages"][-1]["content"])["abstain"] is True


def test_preparation_budget_and_split_checks():
    long_document = document()
    long_document["text"] = "x"*2000
    long_document["evidence_units"][0].update(text="x"*2000, canonical_end=2000)
    rows, rejected = prepare_examples([query()], [long_document], Tokenizer(), {"d"})
    assert not rows and "over_budget" in rejected[0]["reason"]
    dev, rejected = prepare_examples([query("dev")], [document("dev")], Tokenizer(), {"d"}, split="dev", max_examples=60)
    assert dev[0]["split"] == "dev" and not rejected
    for options in ({"max_seq_len": 2048}, {"max_examples": 1001}, {"split": "test"}):
        with pytest.raises(ValueError):
            prepare_examples([], [], Tokenizer(), set(), **options)


def state(step):
    return {"step": step, "epoch": 0, "data_position": step*8, "data_order": [0, 1],
            "rng": {"python": [1, 2, 3]}, "optimizer": {"momentum": step}, "scheduler": {"step": step}}


def writer(directory, payload):
    (directory/"adapter.json").write_text(json.dumps({"weights": payload["step"]}))
    (directory/"progress.json").write_text(json.dumps(payload))


def test_checkpoint_roundtrip_resume_and_three_owned_retention(tmp_path):
    store = CheckpointStore(tmp_path/"run", {"data": "fixed"})
    assert store.latest() is None
    first = store.save(1, state(1), writer, best=True)
    for step in range(2, 6):
        store.save(step, state(step), writer)
    assert first.exists() and len(list(store.root.glob("step-*"))) == 3
    restarted = CheckpointStore(store.root, {"data": "fixed"})
    checkpoint = restarted.latest()
    restored = json.loads((checkpoint/"progress.json").read_text())
    assert restored == state(5)
    assert restored["data_position"] == 40 and restored["optimizer"]["momentum"] == 5
    assert restarted.save(5, state(5), writer) == checkpoint


def test_partial_corrupt_and_unowned_checkpoints_never_resume(tmp_path):
    store = CheckpointStore(tmp_path/"owned", {"a": 1})
    partial = store.root/"step-000001.partial"
    partial.mkdir()
    (partial/"adapter.json").write_text("partial")
    assert store.latest() is None
    path = store.save(2, state(2), writer)
    (path/"adapter.json").write_text("damaged")
    with pytest.raises(ValueError, match="corrupt"):
        store.latest()
    with pytest.raises(ValueError, match="different inputs"):
        CheckpointStore(store.root, {"a": 2})
    unknown = tmp_path/"unknown"
    unknown.mkdir()
    (unknown/"user-data").write_text("preserve")
    with pytest.raises(ValueError, match="unowned"):
        CheckpointStore(unknown, {})
    with pytest.raises(ValueError, match="Incomplete"):
        store.save(3, {"step": 3}, writer)
    assert (unknown/"user-data").read_text() == "preserve"


def test_attention_targets_discovered_from_actual_ancestry_only():
    Linear = type("Linear", (), {})
    Attention = type("Qwen3_5Attention", (), {})
    Gated = type("Qwen3_5GatedDeltaNet", (), {})
    MLP = type("Qwen3_5MLP", (), {})
    def linear():
        result = Linear()
        result.weight = SimpleNamespace(ndim=2)
        return result
    model = SimpleNamespace(named_modules=lambda: [
        ("model.language.layers.0.block", Attention()), ("model.language.layers.0.block.unknown_proj", linear()),
        ("model.language.layers.1.delta", Gated()), ("model.language.layers.1.delta.fused_projection", linear()),
        ("model.vision.block", Attention()), ("model.vision.block.proj", linear()),
        ("model.mlp", MLP()), ("model.mlp.up", linear())])
    assert discover_attention_targets(model) == ["model.language.layers.0.block.unknown_proj", "model.language.layers.1.delta.fused_projection"]
    with pytest.raises(ValueError, match="do not guess"):
        discover_attention_targets(SimpleNamespace(named_modules=lambda: []))


def test_lease_rejects_cpu_stage_and_unverified_state(tmp_path, monkeypatch):
    import psutil
    import sys
    import time
    path = tmp_path/"lease.json"
    path.write_text(json.dumps({"status": "PENDING"}))
    with pytest.raises(ValueError):
        verify_lease(path)
    path.write_text(json.dumps({"status": "ACQUIRED", "owner": "ragscope", "safety_verified": True, "gpu_device": 0,
                               "host_platform": sys.platform, "host_boot_time_unix": psutil.boot_time(), "validated_at": time.time()}))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    with pytest.raises(ValueError, match="CPU-only"):
        verify_lease(path)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    assert verify_lease(path)["gpu_device"] == 0


def test_resource_guard_failure_is_not_claimed_oom(tmp_path, monkeypatch):
    import ragscope.training.run as run
    monkeypatch.setattr(run, "require_disk", lambda amount: {"free": 999})
    def deny(amount):
        raise RuntimeError("RAM budget: available=100, required=200")
    monkeypatch.setattr(run, "require_ram", deny)
    value = resource_guard(config(), config()["paths"][0], tmp_path/"guard.json")
    assert value["status"] == "BLOCKED_RESOURCE" and value["failure_type"] == "preload_resource_guard"
    assert value["gpu_allocated"] is value["torch_imported"] is value["weights_loaded"] is False


def test_frozen_training_bounds_and_paired_test_scope():
    assert validate_config(config())["max_steps"] == 200
    for changed in ({"r": 16}, {"max_steps": 201}, {"max_checkpoints": 4}, {"paths": [{}, {}, {}]}):
        with pytest.raises(ValueError):
            validate_config({**config(), **changed})
    rows = [{"query": {"query_id": "q", "doc_id": "d", "split": "test"},
             "context_bundle": asdict(pack_context([], token_counter=len))}]
    assert validate_comparison_inputs(rows, 1) == ["q"]
    with pytest.raises(ValueError):
        validate_comparison_inputs(rows, 100)
    with pytest.raises(ValueError):
        validate_comparison_inputs(rows, 1, {"d"})


def test_training_snapshot_rejects_partial_receipt_and_unlisted_weight_shards(tmp_path):
    from ragscope.training.run import verify_training_snapshot
    from ragscope.indexing.identity import artifact, atomic_json, file_hash
    model_config={'architectures':['Qwen3_5ForConditionalGeneration']}
    payload={'config.json':model_config,'tokenizer.json':{},'tokenizer_config.json':{},
             'model.safetensors.index.json':{'weight_map':{'parameter':'model-00001-of-00001.safetensors'}}}
    for name,value in payload.items():
        atomic_json(tmp_path/name,value)
    path_config={'model':'fixture','revision':'fixture','config_sha256':file_hash(tmp_path/'config.json')}
    receipt={'model':'fixture','revision':'fixture','status':'DOWNLOADING',
             'artifacts':[artifact(tmp_path/name) for name in payload]}
    atomic_json(tmp_path/'model_manifest.json',receipt)
    with pytest.raises(ValueError,match='VERIFIED'):
        verify_training_snapshot(tmp_path,path_config)
    receipt['status']='VERIFIED'; atomic_json(tmp_path/'model_manifest.json',receipt)
    with pytest.raises(ValueError,match='shards'):
        verify_training_snapshot(tmp_path,path_config)
    shard=tmp_path/'model-00001-of-00001.safetensors'; shard.write_bytes(b'offline fixture bytes')
    receipt['artifacts'].append(artifact(shard)); atomic_json(tmp_path/'model_manifest.json',receipt)
    assert verify_training_snapshot(tmp_path,path_config)=='Qwen3_5ForConditionalGeneration'


def test_training_pairs_answer_with_matching_annotation_and_negative_paragraph():
    q=query()
    q['answer_refs']=['wrong annotation answer','paired answer']
    q['metadata']={'annotations':[{'annotation_id':'first','answer':'wrong annotation answer','unit_ids':['other']},
                                  {'annotation_id':'matched','answer':'paired answer','unit_ids':['u']}]}
    accepted,rejected=prepare_examples([q],[document()],Tokenizer(),{'d'})
    assert not rejected and accepted[0]['annotation_id']=='matched'
    assert json.loads(accepted[0]['messages'][-1]['content'])['answer']=='paired answer'
    q['metadata']['annotations'][1]['unmatched_evidence']=['missing']
    accepted,rejected=prepare_examples([q],[document()],Tokenizer(),{'d'})
    assert not accepted and 'annotation_pair' in rejected[0]['reason']
    doc=document(); doc['evidence_units'][0]['kind']='title'
    accepted,rejected=prepare_examples([{**query(),'answerability':'unanswerable'}],[doc],Tokenizer(),{'d'})
    assert not accepted and rejected[0]['reason']=='no_real_context_units'


def test_bf16_comparison_counts_separate_warmups_and_resumes_without_repeating(tmp_path, monkeypatch):
    import ragscope.training.evaluate as evaluation
    from ragscope.experiments.generation_campaign import GenerationBudget
    from ragscope.experiments.ledger import read_records
    identity = {'model': 'offline-fixture', 'revision': 'offline', 'weight_sha256': 'no-real-weights', 'backend_version': 'fixture'}
    class Client:
        real_model = True
        last_metadata = {}
        prompt_token_counter = staticmethod(lambda messages: 1)
        def __init__(self, model, tokenizer, identity, adapter):
            self.identity = identity
        def chat(self, messages, **kwargs):
            return '{"answer":"","citations":[],"abstain":true}'
    monkeypatch.setattr(evaluation, 'BF16ModelClient', Client)
    rows = [{'query': {'query_id': 'q', 'doc_id': 'd', 'question': 'unknown?', 'split': 'test',
                      'answerability': 'unanswerable', 'answer_refs': [], 'evidence_reference_sets': []},
             'context_bundle': asdict(pack_context([]))}]
    budget = GenerationBudget(tmp_path/'budget.json', limit=20, core_reserved=0)
    def run():
        return evaluation.evaluate_paired(None, None, identity, rows, tmp_path/'paired',
            evidence_units={}, budget=budget, expected_n=1, adapter_identity={'sha256': 'offline-adapter'})
    result = run()
    assert len(result['baseline']) == len(result['adapted']) == 1
    assert budget.state()['reserved_calls'] == 8
    assert len(read_records(tmp_path/'paired/warmup_baseline.jsonl')) == 3
    assert len(read_records(tmp_path/'paired/warmup_adapted.jsonl')) == 3
    run()
    assert budget.state()['reserved_calls'] == 8
