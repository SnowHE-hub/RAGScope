import json
from dataclasses import asdict

import pytest

from ragscope.generation.cache import RAGCache, default_cache_dir
from ragscope.generation.context import pack_context, render_context, validate_bundle, _subtract
from ragscope.generation.llm_client import (GenerationError, LLMClient, LocalLLMClient,
    LlamaCppLLMClient, MockLLMClient, OllamaLLMClient, _local_url)
from ragscope.generation.pipeline import RAGPipeline


def chunk(cid="c1", text="abcdef", start=0, doc="d"):
    return {"chunk_id": cid, "doc_id": doc, "text": text, "canonical_start": start,
            "canonical_end": start+len(text), "unit_spans": [
                {"unit_id": "u", "doc_id": doc, "start": start, "end": start+len(text),
                 "chunk_start": 0, "chunk_end": len(text)}]}


class Retriever:
    def __init__(self, chunks=None):
        self.chunks = chunks if chunks is not None else [chunk()]
        self.calls = 0

    def retrieve(self, question, k=5):
        self.calls += 1
        return self.chunks[:k]


class Client(MockLLMClient):
    def __init__(self, output=None, error=None):
        self.output, self.error, self.calls = output, error, 0

    def chat(self, messages, **kwargs):
        self.calls += 1
        if self.error:
            raise self.error
        return self.output if self.output is not None else super().chat(messages, **kwargs)


def pipeline(tmp_path, client=None, chunks=None, **kwargs):
    return RAGPipeline(Retriever(chunks), client or Client(), cache_dir=str(tmp_path),
                       token_counter=len, **kwargs)


def test_budget_counts_labels_title_and_separator():
    chunks = [{**chunk(), "title": "Title"}, chunk("c2", "xyz", 6)]
    bundle = pack_context(chunks, max_tokens=20, token_counter=len)
    assert len(render_context(bundle)) == bundle.budget["reference_tokens"] <= 20
    assert len(bundle.pieces) == 1
    assert bundle.truncations[0]["reason"] == "reference_budget"
    assert "Title" in render_context(bundle)


def test_overlap_dedup_preserves_exact_source_coordinates():
    bundle = pack_context([chunk(), chunk("c2", "efgh", 4)], token_counter=len)
    assert [p["text"] for p in bundle.pieces] == ["abcdef", "gh"]
    assert bundle.pieces[1]["chunk_id"] == "c2:6-8"
    assert bundle.unit_spans[1]["start"] == 6
    assert bundle.unit_spans[1]["chunk_start"] == 0
    assert bundle.deduplication[0]["reason"] == "canonical_overlap"
    assert _subtract(0, 10, [(2, 4), (6, 8)]) == [(0, 2), (4, 6), (8, 10)]


def test_scope_and_parent_dedup_and_cross_document_guard():
    children = [{**chunk("a"), "parent_id": "p"}, {**chunk("b"), "parent_id": "p"}, chunk(doc="other")]
    bundle = pack_context(children, token_counter=len, scope={"doc_id": "d"}, expansion="parent", parent_lookup={"p": chunk("p")})
    assert [p["chunk_id"] for p in bundle.pieces] == ["p"]
    assert bundle.deduplication and bundle.truncations
    corrupt = pack_context(children[:1], token_counter=len, expansion="parent", parent_lookup={"p": chunk("p", doc="other")})
    assert not corrupt.pieces and corrupt.truncations[0]["reason"] == "cross_document_link"


def test_neighbor_and_missing_parent():
    c = {**chunk(), "next_chunk_id": "next"}
    bundle = pack_context([c], token_counter=len, expansion="neighbor", chunk_lookup={"next": chunk("next", "gh", 6)})
    assert len(bundle.pieces) == 2
    missing = pack_context([{**c, "parent_id": "absent"}], token_counter=len, expansion="parent")
    assert missing.truncations[0]["reason"] == "parent_missing_used_child"


@pytest.mark.parametrize("scope", ["d", ["d"], {"doc_ids": ["d"]}, "all"])
def test_scope_forms(scope):
    assert pack_context([chunk()], token_counter=len, scope=scope).pieces


@pytest.mark.parametrize("kwargs", [{"scope": {}}, {"max_tokens": -1}, {"expansion": "bad"}])
def test_invalid_context_configuration(kwargs):
    with pytest.raises(ValueError):
        pack_context([chunk()], token_counter=len, **kwargs)


def test_shared_model_budget_and_mutation_detection():
    bundle = pack_context([chunk()], token_counter=len, model_budget_checks=[lambda text: False])
    assert not bundle.pieces and bundle.truncations[0]["reason"] == "model_budget"
    bundle = pack_context([chunk()], token_counter=len)
    validate_bundle(bundle)
    bundle.pieces[0]["text"] = "changed"
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_bundle(bundle)


def test_legacy_chunks_empty_duplicate_and_invalid_ids():
    data = [{"chunk_id": "a", "doc_id": "d", "text": "text"}]
    assert len(pack_context(data+data, token_counter=len).pieces) == 1
    assert not pack_context([chunk(text="")], token_counter=len).pieces
    with pytest.raises(ValueError):
        pack_context([chunk("[unsafe]")], token_counter=len)


def test_cache_separates_full_model_index_context_prompt_and_config(tmp_path):
    p = pipeline(tmp_path)
    one, cached = p.answer("q"), p.answer("q")
    assert one.real_model is False and not one.cache_hit and cached.cache_hit
    assert p.llm_client.calls == 1
    p.index_identity = {"revision": "new"}
    assert not p.answer("q").cache_hit
    p.llm_client.identity = {"model": "other"}
    assert not p.answer("q").cache_hit
    p.system_prompt += " Changed."
    assert not p.answer("q").cache_hit
    p.config["temperature"] = .2
    assert not p.answer("q").cache_hit
    p.retriever.chunks = [chunk(text="different")]
    assert not p.answer("q").cache_hit
    p.cache.close()


@pytest.mark.parametrize("error,kind", [(TimeoutError("late"), "timeout"),
    (GenerationError("oom", "OOM"), "oom"), (RuntimeError("bad"), "transport_error")])
def test_generation_failures_are_not_answers_or_cached(tmp_path, error, kind):
    client = Client(error=error)
    p = pipeline(tmp_path, client)
    result = p.answer("q")
    assert result.error["kind"] == kind and result.answer == "" and result.citations == []
    assert getattr(result, kind)
    assert p.cache.size() == 0
    p.answer("q")
    assert client.calls == 2
    p.cache.close()


@pytest.mark.parametrize("raw", ["", "not json", "[]", '{"answer":3}',
    '{"answer":"ok","citations":"c1","abstain":false}',
    '{"answer":"ok","citations":[],"abstain":"false"}',
    '{"answer":"ok","citations":[],"abstain":false,"confidence":2}'])
def test_json_failures_never_fabricate_answers_or_citations(tmp_path, raw):
    p = pipeline(tmp_path, Client(raw))
    r = p.answer("q")
    assert r.parse_error and r.answer == "" and r.citations == [] and r.raw_output == raw
    assert p.cache.size() == 0
    assert p._parse_output(raw, [chunk()]) == ("", [], 0.0)
    p.cache.close()


def test_only_actually_sent_citations_are_accepted(tmp_path):
    raw = json.dumps({"answer": "ok", "citations": ["c1", "c2"], "abstain": False})
    p = pipeline(tmp_path, Client(raw), [chunk(), chunk("c2", "z"*100, 6)], context_config={"max_tokens": 15})
    r = p.answer("q")
    assert r.citations == ["c1"] and r.invalid_citations == ["c2"]
    assert len(r.retrieved_chunks) == 2 and len(r.context_bundle["pieces"]) == 1
    assert p.cache.size() == 0
    p.cache.close()


def test_frozen_context_bypasses_retrieval_and_formal_cache(tmp_path):
    client = Client()
    client.real_model = True  # test double only; this test creates no experiment output
    client.identity = {"model": "test", "revision": "test", "weight_sha256": "test", "backend_version": "test"}
    client.prompt_token_counter = lambda messages: 0
    p = pipeline(tmp_path, client)
    bundle = pack_context([chunk()], token_counter=len)
    r = p.answer("q", context_bundle=bundle, formal_evaluation=True)
    p.answer("q", context_bundle=bundle, formal_evaluation=True)
    assert p.retriever.calls == 0 and client.calls == 2
    assert r.prompt_hash and r.timings["first_token_ms"] is None
    assert p.cache.size() == 0
    p.cache.close()
    with pytest.raises(ValueError, match="Mock"):
        pipeline(tmp_path, use_cache=False).answer("q", formal_evaluation=True)


def test_cache_runtime_and_stats(tmp_path, monkeypatch):
    monkeypatch.setenv("RAGSCOPE_HOME", str(tmp_path))
    assert str(tmp_path) in default_cache_dir()
    cache = RAGCache()
    cache.set("x", "v")
    assert cache.get("x") == "v" and cache.get("missing") is None
    assert cache.stats()["hit_rate"] == .5
    cache.clear()
    assert cache.size() == 0
    cache.close()


@pytest.mark.parametrize("url", ["https://127.0.0.1", "http://example.com", "http://0.0.0.0", "http://user:pw@localhost", "http://localhost?x=1"])
def test_local_only_transports(url):
    with pytest.raises(ValueError):
        _local_url(url)


def test_client_rejects_cloud_keys_and_bad_configuration():
    with pytest.raises(ValueError):
        LLMClient(provider="openai")
    with pytest.raises(ValueError):
        LLMClient(api_key="not-a-secret")
    with pytest.raises(ValueError):
        LocalLLMClient("m", timeout=0)
    assert _local_url("http://[::1]:8080/") == "http://[::1]:8080"


def test_cpu_policy_blocks_before_network(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    client = OllamaLLMClient(num_gpu=1)
    with pytest.raises(GenerationError, match="CPU-only"):
        client.chat([])


def test_local_payloads_record_actual_timings_and_schema(monkeypatch):
    calls = []
    def request(self, path, payload):
        calls.append((path, payload))
        return {"message": {"content": "{}"}, "prompt_eval_count": 4,
                "choices": [{"message": {"content": "{}"}}], "usage": {"prompt_tokens": 4}}
    monkeypatch.setattr(LocalLLMClient, "_request", request)
    ollama = OllamaLLMClient(prompt_token_counter=lambda messages: 12)
    assert ollama.chat([]) == "{}"
    assert calls[-1][1]["options"]["num_gpu"] == 0 and calls[-1][1]["think"] is False
    assert calls[-1][1]["format"]["required"] == ["answer", "citations", "abstain"]
    assert ollama.last_metadata["actual_input_tokens"] == 12
    ollama.unload()
    assert calls[-1][1]["keep_alive"] == 0
    llama = LlamaCppLLMClient("m")
    assert llama.chat([]) == "{}" and calls[-1][1]["response_format"]["schema"]
    assert llama.estimate_cost(1, 1) == 0
    with pytest.raises(ValueError):
        llama.unload()
    budget = OllamaLLMClient(num_ctx=100, prompt_token_counter=lambda messages: 80)
    with pytest.raises(GenerationError, match="reserve"):
        budget.chat([], max_tokens=30)


def test_missing_backend_content_is_transport_error(monkeypatch):
    monkeypatch.setattr(LocalLLMClient, "_request", lambda *args: {})
    for client in (OllamaLLMClient(), LlamaCppLLMClient("m")):
        with pytest.raises(GenerationError):
            client.chat([])


def test_transport_success_error_timeout_and_redirect(monkeypatch):
    import io
    import urllib.error
    import urllib.request
    from ragscope.generation.llm_client import _NoRedirect
    class Response(io.BytesIO):
        pass
    class Opener:
        payload = b'{"message":{"content":"ok"}}'
        error = None
        def open(self, request, timeout):
            if self.error:
                raise self.error
            return Response(self.payload)
    opener = Opener()
    monkeypatch.setattr(urllib.request, "build_opener", lambda *args: opener)
    client = OllamaLLMClient()
    assert client.chat([]) == "ok"
    for failure, expected in [(TimeoutError("late"), "timeout"),
        (urllib.error.URLError("offline"), "transport_error"),
        (urllib.error.URLError(TimeoutError("late")), "timeout"),
        (urllib.error.HTTPError("local", 500, "OOM", {}, Response(b"out of memory")), "oom")]:
        opener.error = failure
        with pytest.raises(GenerationError) as caught:
            client.chat([])
        assert caught.value.kind == expected
    opener.error = None
    for payload, expected in [(b'{"error":"out of memory"}', "oom"), (b'not json', "transport_error")]:
        opener.payload = payload
        with pytest.raises(GenerationError) as caught:
            client.chat([])
        assert caught.value.kind == expected
    with pytest.raises(GenerationError, match="redirect"):
        _NoRedirect().redirect_request(None, None, 302, "", {}, "https://elsewhere")


def test_formal_identity_and_tokenizer_required(tmp_path):
    client = Client()
    client.real_model = True
    p = pipeline(tmp_path, client, use_cache=False)
    with pytest.raises(ValueError, match="identity"):
        p.answer("q", formal_evaluation=True)
    client.identity = {"model": "m", "revision": "r", "weight_sha256": "h", "backend_version": "v"}
    with pytest.raises(ValueError, match="token budget"):
        p.answer("q", formal_evaluation=True)


def test_llamacpp_cpu_service_must_be_verified(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    with pytest.raises(GenerationError, match="CPU-only launch"):
        LlamaCppLLMClient("m").chat([])


def test_cache_rejects_error_records(tmp_path):
    cache = RAGCache(str(tmp_path))
    cache.set("error", {"error": {"kind": "timeout"}})
    assert cache.size() == 0
    cache.close()
