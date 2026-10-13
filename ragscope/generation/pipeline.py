"""RAG pipeline with frozen visible context and explicit non-answer failures."""
from __future__ import annotations

import copy
import json
import math
import time
from dataclasses import dataclass, field

from ragscope.generation.cache import RAGCache
from ragscope.generation.context import (as_mapping, fingerprint, pack_context,
                                         reference_token_count, render_context, validate_bundle)
from ragscope.generation.prompts import (PROMPT_VERSION, RAG_SYSTEM_PROMPT, RAG_USER_TEMPLATE,
    OUTPUT_CONSTRAINT_VERSION, visible_output_schema)


@dataclass
class RAGResult:
    question: str
    answer: str
    citations: list[str]
    confidence: float
    retrieved_chunks: list[dict]
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    cache_hit: bool
    abstain: bool = False
    raw_output: str = ""
    parsed_output: dict = field(default_factory=dict)
    error: dict | None = None
    transport_error: str | None = None
    timeout: str | None = None
    oom: str | None = None
    parse_error: str | None = None
    context_bundle: dict = field(default_factory=dict)
    prompt: list[dict] = field(default_factory=list)
    prompt_hash: str = ""
    model_identity: dict = field(default_factory=dict)
    real_model: bool = False
    invalid_citations: list[str] = field(default_factory=list)
    timings: dict = field(default_factory=dict)
    backend_metadata: dict = field(default_factory=dict)
    config_hash: str = ""
    prompt_version: str = PROMPT_VERSION
    output_schema: dict = field(default_factory=dict)
    output_schema_hash: str = ""
    citation_constraint: dict = field(default_factory=dict)


class RAGPipeline:
    def __init__(self, retriever, llm_client, use_cache=True, cache_dir=None, *,
                 context_config=None, index_identity=None, config=None,
                 token_counter=None, system_prompt=RAG_SYSTEM_PROMPT,
                 user_template=RAG_USER_TEMPLATE):
        self.retriever, self.llm_client = retriever, llm_client
        self.use_cache = use_cache
        self.cache = RAGCache(cache_dir=cache_dir) if use_cache else None
        self.context_config = dict(context_config or {})
        self.index_identity = index_identity or getattr(retriever, "identity", {})
        self.config = {"max_tokens": 512, "temperature": 0.0, **(config or {}),
                       'output_constraint_version':OUTPUT_CONSTRAINT_VERSION}
        self.token_counter = token_counter or reference_token_count
        self.system_prompt, self.user_template = system_prompt, user_template

    def answer(self, question, k=5, *, context_bundle=None, scope=None, formal_evaluation=False):
        start = time.perf_counter()
        identity = getattr(self.llm_client, "identity", {"model": getattr(self.llm_client, "model", "unknown"),
                                                        "provider": getattr(self.llm_client, "provider", "unknown")})
        real = getattr(self.llm_client, "real_model", False)
        if formal_evaluation and not real:
            raise ValueError("Mock/unidentified client cannot produce formal generation results")
        if formal_evaluation:
            if not all(identity.get(key) for key in ("model", "revision", "weight_sha256", "backend_version")):
                raise ValueError("Formal evaluation requires pinned weight and backend identity")
            if not callable(getattr(self.llm_client, "prompt_token_counter", None)):
                raise ValueError("Formal evaluation requires real chat-template token budget validation")
        chunks, timings, error, raw, parsed, invalid = [], {}, None, "", {}, []
        bundle, messages = None, []
        key, prompt_hash = "", ""
        output_schema = {}
        constrained = bool(getattr(self.llm_client,'supports_output_schema',False) and getattr(self.llm_client,'json_constraint',False))
        phase = "retrieval_error"
        try:
            if context_bundle is None:
                retrieval_start = time.perf_counter()
                chunks = [as_mapping(c) for c in self.retriever.retrieve(question, k=k)]
                timings["retrieval_ms"] = (time.perf_counter() - retrieval_start) * 1000
                packing_start = time.perf_counter()
                phase = "context_error"
                bundle = pack_context(chunks, scope=scope, token_counter=self.token_counter, **self.context_config)
                timings["context_ms"] = (time.perf_counter() - packing_start) * 1000
            else:
                phase = "context_error"
                bundle = validate_bundle(context_bundle, scope=scope)
                chunks = copy.deepcopy(as_mapping(bundle)["pieces"])
                timings.update(retrieval_ms=0.0, context_ms=0.0, frozen_context=True)
            messages = [{"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": self.user_template.format(question=question, context=render_context(bundle))}]
            prompt_hash = fingerprint(messages)
            output_schema=visible_output_schema(as_mapping(bundle)['pieces']) if constrained else {}
            key = self._make_key(question, k, context_hash=as_mapping(bundle)["context_hash"],
                                 prompt_hash=prompt_hash, scope=scope)
            if self.cache is not None and not formal_evaluation:
                cached = self.cache.get(key)
                if cached is not None and not cached.error:
                    cached = copy.deepcopy(cached)
                    cached.cache_hit = True
                    cached.latency_ms = (time.perf_counter() - start) * 1000
                    return cached
            generation_start = time.perf_counter()
            phase = "transport_error"
            try:
                schema_options={'output_schema':output_schema} if constrained else {}
                raw = self.llm_client.chat(messages, max_tokens=self.config["max_tokens"],
                                            temperature=self.config["temperature"],**schema_options)
            finally:
                timings["generation_ms"] = (time.perf_counter() - generation_start) * 1000
            phase = "parse_error"
            parsed = self.parse_output(raw)
            visible = {p["chunk_id"] for p in as_mapping(bundle)["pieces"]}
            invalid = [cid for cid in parsed["citations"] if cid not in visible]
        except Exception as exc:
            kind = getattr(exc, "kind", None)
            if kind is None:
                kind = "timeout" if isinstance(exc, TimeoutError) else phase
            error = {"kind": kind, "message": str(exc), "exception": type(exc).__name__}
        timings["total_ms"] = (time.perf_counter() - start) * 1000
        timings["first_token_ms"] = None  # Never estimate first-token timing from a nonstreaming total.
        prompt_tokens = sum(self.token_counter(m["content"]) for m in messages)
        completion_tokens = self.token_counter(raw)
        flags = {kind: error["message"] if error and error["kind"] == kind else None
                 for kind in ("transport_error", "timeout", "oom", "parse_error")}
        result = RAGResult(question=question, answer=parsed.get("answer", "") if not error else "",
                           citations=[c for c in parsed.get("citations", []) if c not in invalid] if not error else [],
                           confidence=parsed.get("confidence", 0.0), retrieved_chunks=chunks,
                           latency_ms=timings["total_ms"], prompt_tokens=prompt_tokens,
                           completion_tokens=completion_tokens, cost_usd=0.0, cache_hit=False,
                           abstain=parsed.get("abstain", False) if not error else False,
                           raw_output=raw, parsed_output=parsed, error=error, **flags,
                           context_bundle=as_mapping(bundle) if bundle is not None else {},
                           prompt=messages, prompt_hash=prompt_hash, model_identity=identity, real_model=real,
                           invalid_citations=invalid, timings=timings,
                           output_schema=output_schema,output_schema_hash=fingerprint(output_schema),
                           citation_constraint={'applied':constrained,'version':OUTPUT_CONSTRAINT_VERSION,
                               'interpretation':'Visible citation IDs constrained by construction; not evidence-support or faithfulness quality'},
                           backend_metadata=copy.deepcopy(getattr(self.llm_client, "last_metadata", {})),
                           config_hash=fingerprint({"generation": self.config, "index": self.index_identity,
                                                    "model": identity, "prompt_version": PROMPT_VERSION,
                                                    "output_schema_hash":fingerprint(output_schema)}))
        if self.cache is not None and not formal_evaluation and not error and not invalid:
            self.cache.set(key, result)
        return result

    def _make_key(self, question, k, *, context_hash="", prompt_hash="", scope=None):
        # Resolved context and prompt hashes guard even legacy retrievers with no manifest.
        return fingerprint({"version": "rag-cache-v2", "question": question, "k": k, "scope": scope,
                            "retriever": type(self.retriever).__name__, "index": self.index_identity,
                            "context_hash": context_hash, "prompt_hash": prompt_hash,
                            "model": getattr(self.llm_client, "identity", {"model": getattr(self.llm_client, "model", "unknown")}),
                            "config": self.config, "prompt_version": PROMPT_VERSION})

    @staticmethod
    def parse_output(raw):
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, TypeError) as exc:
            raise ValueError("Model output is not valid JSON") from exc
        if not isinstance(data, dict) or not isinstance(data.get("answer"), str):
            raise ValueError("JSON output must contain a string answer")
        if not isinstance(data.get("citations"), list) or not all(isinstance(c, str) for c in data["citations"]):
            raise ValueError("JSON citations must be a list of strings")
        if not isinstance(data.get("abstain"), bool):
            raise ValueError("JSON output must declare boolean abstain")
        confidence = data.get("confidence", 0.0)
        if not isinstance(confidence, (float, int)) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError("Invalid confidence")
        return {**data, "citations": list(dict.fromkeys(data["citations"])), "confidence": float(confidence)}

    @staticmethod
    def _parse_output(raw, chunks):
        """Legacy tuple interface; malformed output yields no answer or citations."""
        try:
            data = RAGPipeline.parse_output(raw)
            visible = {c["chunk_id"] for c in chunks}
            return data["answer"], [c for c in data["citations"] if c in visible], data["confidence"]
        except ValueError:
            return "", [], 0.0
