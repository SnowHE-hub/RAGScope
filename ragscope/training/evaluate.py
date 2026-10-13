"""Paired BF16 baseline/adapter evaluation on identical frozen contexts."""
from __future__ import annotations
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
import os
import time

from ragscope.evaluation.v2 import score_generation
from ragscope.experiments.ledger import RecordStore
from ragscope.generation.pipeline import RAGPipeline
from ragscope.generation.context import validate_bundle
from ragscope.generation.llm_client import GenerationError
from ragscope.experiments.generation_campaign import CountedClient
from ragscope.indexing.identity import signature


def validate_comparison_inputs(frozen_rows, expected_n=100, train_doc_ids=()):
    if len(frozen_rows) != expected_n or not 1 <= expected_n <= 100:
        raise ValueError("LoRA comparison must use the frozen test count, at most 100")
    query_ids = [row["query"]["query_id"] for row in frozen_rows]
    if len(set(query_ids)) != len(query_ids):
        raise ValueError("Duplicate test queries")
    for row in frozen_rows:
        query = row["query"]
        if query.get("split") != "test" or query.get("doc_id") in set(train_doc_ids):
            raise ValueError("Invalid test split or train/test document overlap")
        if not row.get("context_bundle", {}).get("context_hash"):
            raise ValueError("Paired comparison requires frozen contexts")
        validate_bundle(row["context_bundle"], scope=query.get("scope"))
    return query_ids


class BF16ModelClient:
    real_model = True
    provider = "transformers-local-bf16"

    def __init__(self, model, tokenizer, identity, *, adapter=True, max_context=4096):
        self.model_object, self.tokenizer, self.identity = model, tokenizer, identity
        self.model = identity["model"]
        self.adapter, self.max_context, self.last_metadata = adapter, max_context, {}
        self.base_dtype_map = {name: str(parameter.dtype) for name, parameter in model.named_parameters()
                              if parameter.is_floating_point() and "lora_" not in name}

    def prompt_token_counter(self, messages):
        return len(self.tokenizer.apply_chat_template(messages, tokenize=True,
            add_generation_prompt=True, enable_thinking=False, return_dict=False))

    def chat(self, messages, max_tokens=512, temperature=0.0):
        import torch
        import psutil
        self.last_metadata={}
        if temperature != 0:
            raise ValueError("Paired comparison freezes greedy decoding")
        ids = self.tokenizer.apply_chat_template(messages, tokenize=True,
            add_generation_prompt=True, enable_thinking=False, return_dict=False)
        if len(ids)+max_tokens > self.max_context:
            raise ValueError("Frozen BF16 comparison prompt exceeds total budget")
        observed_dtypes = {name: str(p.dtype) for name, p in self.model_object.named_parameters() if p.is_floating_point() and "lora_" not in name}
        base_dtypes = set(observed_dtypes.values())
        if observed_dtypes != self.base_dtype_map or "torch.bfloat16" not in base_dtypes or not base_dtypes <= {"torch.bfloat16", "torch.float32"}:
            raise ValueError("Baseline and adapted model must retain BF16 base weights")
        self.model_object.eval()
        torch.cuda.reset_peak_memory_stats()
        started=time.perf_counter()
        inputs = torch.tensor([ids], device=next(self.model_object.parameters()).device)
        context = nullcontext() if self.adapter else self.model_object.disable_adapter()
        try:
            with context, torch.no_grad():
                output = self.model_object.generate(input_ids=inputs, attention_mask=torch.ones_like(inputs),
                    do_sample=False, max_new_tokens=max_tokens, use_cache=True)
        except torch.cuda.OutOfMemoryError as exc:
            self.last_metadata={'resources':{'peak_gpu_allocated_bytes':torch.cuda.max_memory_allocated(),
                'peak_gpu_reserved_bytes':torch.cuda.max_memory_reserved()},'error':'actual_cuda_oom'}
            raise GenerationError('oom',str(exc)) from exc
        new_ids = output[0, len(ids):]
        self.last_metadata = {"actual_input_tokens": len(ids), "actual_output_tokens": len(new_ids),
                              "dtype": "bfloat16", "adapter": self.adapter, "json_constraint": "prompt_only",
                              "base_dtype_signature": signature(observed_dtypes),
                              "first_token_ms": None,'generation_seconds':time.perf_counter()-started,
                              'resources':{'peak_gpu_allocated_bytes':torch.cuda.max_memory_allocated(),
                                'peak_gpu_reserved_bytes':torch.cuda.max_memory_reserved(),
                                'process_rss_after_request_bytes':psutil.Process(os.getpid()).memory_info().rss,
                                'host_available_after_request_bytes':psutil.virtual_memory().available,
                                'host_peak_not_measured':True,'gpu_measurement':'PyTorch allocator maximum including resident base weights'}}
        return self.tokenizer.decode(new_ids, skip_special_tokens=True)


def evaluate_paired(model, tokenizer, identity, frozen_rows, run_dir, *, evidence_units,
                    budget, expected_n=100, train_doc_ids=(), adapter_identity=None):
    query_ids = validate_comparison_inputs(frozen_rows, expected_n, train_doc_ids)
    if not adapter_identity:
        raise ValueError("Pinned adapter identity is required")
    common = {"query_ids": query_ids, "frozen_inputs_hash": signature(frozen_rows),
              "dtype": "bfloat16", "backend": "transformers", "max_tokens": 512, "temperature": 0.0}
    stores = {}
    # Same loaded BF16 model, same tokenizer/backend/prompt, adapters toggled.
    for adapter in (False, True):
        tag = "adapted" if adapter else "baseline"
        active_identity = {**identity, "adapter": adapter_identity if adapter else None}
        store = RecordStore(Path(run_dir) / f"{tag}.jsonl", {**common, "identity": active_identity},
                            real_only=True, max_records=expected_n)
        client = BF16ModelClient(model, tokenizer, active_identity, adapter=adapter)
        consecutive_backend_failures=0
        for row in frozen_rows:
            query = row["query"]
            if query["query_id"] in store.ids:
                continue
            counted = CountedClient(client, budget, "lora_"+tag, query["query_id"])
            pipeline = RAGPipeline(None, counted, use_cache=False)
            result = pipeline.answer(query["question"], context_bundle=row["context_bundle"],
                                     scope=query.get("scope"), formal_evaluation=True)
            if counted.last_ticket is None:
                raise RuntimeError(f"Comparison request did not start; preserved completed rows: {result.error}")
            record = {**asdict(result), "query_id": query["query_id"], "doc_id": query["doc_id"],
                      'cluster_id':query['doc_id'],'answerability':query['answerability'],
                      'document_length':row.get('document_length',{}),
                      'resources':result.backend_metadata.get('resources',{}),
                      "split": "test", "call_ticket": counted.last_ticket,
                      "scores": score_generation(query, result, evidence_units=evidence_units)}
            store.add(query["query_id"], record)
            kind=(result.error or {}).get('kind')
            consecutive_backend_failures=consecutive_backend_failures+1 if kind in {'transport_error','timeout','oom'} else 0
            if kind=='oom' or consecutive_backend_failures>=3:
                raise RuntimeError('BF16 comparison circuit-break; failed row retained and remaining frozen queries stay pending')
        stores[tag] = store.records
    return stores
