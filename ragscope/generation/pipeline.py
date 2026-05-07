"""
RAGPipeline — retrieval-augmented generation end-to-end:
    question → retrieve → format context → LLM → parse → cache → RAGResult
"""

from __future__ import annotations

# Allow `python ragscope/generation/pipeline.py` as well as `python -m`
import sys as _sys
from pathlib import Path as _Path
if str((_Path(__file__).resolve().parent.parent.parent)) not in _sys.path:
    _sys.path.insert(0, str(_Path(__file__).resolve().parent.parent.parent))

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import tiktoken
from loguru import logger

from ragscope.generation.cache import RAGCache
from ragscope.generation.prompts import RAG_SYSTEM_PROMPT, RAG_USER_TEMPLATE, format_context
from ragscope.retrieval.base import BaseRetriever

_ENC = tiktoken.get_encoding("cl100k_base")


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class RAGResult:
    question:          str
    answer:            str
    citations:         List[str]
    confidence:        float
    retrieved_chunks:  List[Dict]
    latency_ms:        float
    prompt_tokens:     int
    completion_tokens: int
    cost_usd:          float
    cache_hit:         bool


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

_DEFAULT_RAG_CACHE = os.path.join(os.path.expanduser("~"), ".cache", "ragscope", "rag")


class RAGPipeline:
    def __init__(
        self,
        retriever:  BaseRetriever,
        llm_client,
        use_cache:  bool = True,
        cache_dir:  str  = _DEFAULT_RAG_CACHE,
    ) -> None:
        self.retriever  = retriever
        self.llm_client = llm_client
        self.use_cache  = use_cache
        self.cache: Optional[RAGCache] = RAGCache(cache_dir=cache_dir) if use_cache else None

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def answer(self, question: str, k: int = 5) -> RAGResult:
        t0 = time.perf_counter()
        cache_key = self._make_key(question, k)

        # ── 1. Cache lookup ───────────────────────────────────────────
        if self.cache is not None:
            cached: Optional[RAGResult] = self.cache.get(cache_key)
            if cached is not None:
                cached.cache_hit  = True
                cached.latency_ms = (time.perf_counter() - t0) * 1000
                logger.debug(f"Cache hit — {question[:60]!r}")
                return cached

        # ── 2. Retrieve ───────────────────────────────────────────────
        chunks = self.retriever.retrieve(question, k=k)

        # ── 3. Format context ─────────────────────────────────────────
        context = format_context(chunks)

        # ── 4. Build messages ─────────────────────────────────────────
        user_content = RAG_USER_TEMPLATE.format(
            question=question,
            context=context,
        )
        messages = [
            {"role": "system", "content": RAG_SYSTEM_PROMPT},
            {"role": "user",   "content": user_content},
        ]
        prompt_tokens = sum(len(_ENC.encode(m["content"])) for m in messages)

        # ── 5. LLM call ───────────────────────────────────────────────
        try:
            raw_output = self.llm_client.chat(messages)
        except Exception as exc:
            logger.error(f"LLM call failed: {exc}")
            raw_output = json.dumps(
                {"answer": f"LLM error: {exc}", "citations": [], "confidence": 0.0}
            )

        # ── 6. Parse JSON ─────────────────────────────────────────────
        answer, citations, confidence = self._parse_output(raw_output, chunks)

        # ── 7. Validate citations ─────────────────────────────────────
        valid_ids = {c["chunk_id"] for c in chunks}
        citations = [cid for cid in citations if cid in valid_ids]

        completion_tokens = len(_ENC.encode(raw_output))
        cost_usd          = self.llm_client.estimate_cost(prompt_tokens, completion_tokens)

        result = RAGResult(
            question          = question,
            answer            = answer,
            citations         = citations,
            confidence        = confidence,
            retrieved_chunks  = chunks,
            latency_ms        = (time.perf_counter() - t0) * 1000,
            prompt_tokens     = prompt_tokens,
            completion_tokens = completion_tokens,
            cost_usd          = cost_usd,
            cache_hit         = False,
        )

        # ── 8. Cache ──────────────────────────────────────────────────
        if self.cache is not None:
            self.cache.set(cache_key, result)

        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_key(self, question: str, k: int) -> str:
        retriever_tag = type(self.retriever).__name__
        raw = f"{question}|{retriever_tag}|{k}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    @staticmethod
    def _parse_output(
        raw: str,
        chunks: List[Dict],
    ) -> Tuple[str, List[str], float]:
        """Parse JSON from LLM output.  Falls back gracefully on bad output."""
        try:
            data       = json.loads(raw)
            answer     = str(data.get("answer", raw))
            citations  = data.get("citations", [])
            confidence = float(data.get("confidence", 0.5))

            if not isinstance(citations, list):
                citations = [citations] if citations else []
            confidence = max(0.0, min(1.0, confidence))
            return answer, citations, confidence

        except (json.JSONDecodeError, ValueError, TypeError):
            logger.warning("LLM output was not valid JSON — returning raw text as answer")
            first_id = chunks[0]["chunk_id"] if chunks else ""
            return raw, ([first_id] if first_id else []), 0.0


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import shutil
    from ragscope.generation.llm_client import MockLLMClient

    class _MockRetriever:
        """Minimal fake retriever for local testing."""
        def retrieve(self, query: str, k: int = 5) -> List[Dict]:
            return [
                {
                    "chunk_id": f"squad_42_chunk_{i}",
                    "doc_id":   "squad_42",
                    "text": (
                        "Machine learning is a field of artificial intelligence that "
                        "uses statistical techniques to give computer systems the ability "
                        f"to learn from data. (passage {i})"
                    ),
                    "score":    round(1.0 - i * 0.15, 3),
                    "metadata": {"source": "squad"},
                }
                for i in range(min(k, 3))
            ]

    _TMP_CACHE = ".cache/rag_smoke_test"
    retriever  = _MockRetriever()
    llm        = MockLLMClient()
    pipeline   = RAGPipeline(retriever, llm, use_cache=True, cache_dir=_TMP_CACHE)

    QUESTION = "What is machine learning?"

    # ── First call (cache miss) ────────────────────────────────────────────
    r1 = pipeline.answer(QUESTION, k=3)
    print("\n=== RAGPipeline smoke test ===")
    print(f"Question         : {r1.question}")
    print(f"Answer           : {r1.answer}")
    print(f"Citations        : {r1.citations}")
    print(f"Confidence       : {r1.confidence}")
    print(f"Cache hit        : {r1.cache_hit}")
    print(f"Latency          : {r1.latency_ms:.1f} ms")
    print(f"Prompt tokens    : {r1.prompt_tokens}")
    print(f"Completion tokens: {r1.completion_tokens}")
    print(f"Cost             : ${r1.cost_usd:.8f}")
    print(f"Retrieved chunks : {len(r1.retrieved_chunks)}")

    assert not r1.cache_hit
    assert r1.answer
    assert r1.citations[0].startswith("squad_42_chunk_")
    assert 0.0 <= r1.confidence <= 1.0
    assert r1.prompt_tokens > 0

    # ── Second call (cache hit) ────────────────────────────────────────────
    r2 = pipeline.answer(QUESTION, k=3)
    assert r2.cache_hit, "Second call should be a cache hit"
    print(f"\nSecond call cache_hit: {r2.cache_hit}")
    print(f"Cache stats      : {pipeline.cache.stats()}")

    # ── Cleanup ────────────────────────────────────────────────────────────
    shutil.rmtree(_TMP_CACHE, ignore_errors=True)
    print("\nAll pipeline checks: PASS")
