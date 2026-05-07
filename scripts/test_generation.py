"""
Generation module smoke test — runs without an OpenAI key via MockLLMClient.

Tests:
  1. MockLLMClient output format
  2. format_context token cap
  3. RAGCache get/set/stats
  4. Full RAGPipeline (cache miss → cache hit)
  5. JSON parse failure fallback
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import tiktoken
from loguru import logger

from ragscope.generation.cache import RAGCache
from ragscope.generation.llm_client import MockLLMClient
from ragscope.generation.pipeline import RAGPipeline, RAGResult
from ragscope.generation.prompts import format_context

_ENC = tiktoken.get_encoding("cl100k_base")
# Use ASCII home path so diskcache works on Windows with non-ASCII working dirs
_TMP = os.path.join(os.path.expanduser("~"), ".cache", "ragscope", "_test_generation")

QUERY = "What is machine learning?"

# ---------------------------------------------------------------------------
# Tiny fake retriever
# ---------------------------------------------------------------------------

class _FakeRetriever:
    def retrieve(self, query: str, k: int = 5):
        return [
            {
                "chunk_id": f"squad_7_chunk_{i}",
                "doc_id":   "squad_7",
                "text": (
                    "Machine learning is a subset of artificial intelligence "
                    "that enables computers to learn from data without being "
                    f"explicitly programmed. (passage {i})"
                ),
                "score":    round(1.0 - i * 0.15, 3),
                "metadata": {"source": "squad"},
            }
            for i in range(min(k, 4))
        ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_mock_llm() -> None:
    llm = MockLLMClient()
    messages = [
        {"role": "system", "content": "Be precise."},
        {"role": "user",   "content": "Question: What is X?\n\nContext:\n[my_chunk_01] Some text.\n\nAnswer in JSON."},
    ]
    raw = llm.chat(messages)
    data = json.loads(raw)
    assert data["answer"] == "Mock answer based on context."
    assert data["citations"] == ["my_chunk_01"]
    assert data["confidence"] == 0.5
    assert llm.estimate_cost(1000, 500) == 0.0
    logger.info("test_mock_llm: PASS")


def test_format_context() -> None:
    chunks = [
        {"chunk_id": "c0", "text": "Alpha " * 100},  # ~100 tokens
        {"chunk_id": "c1", "text": "Beta "  * 100},
        {"chunk_id": "c2", "text": "Gamma " * 100},
        {"chunk_id": "c3", "text": "Delta " * 100},
    ]
    ctx = format_context(chunks, max_tokens=250)
    n_tok = len(_ENC.encode(ctx))
    assert n_tok <= 250, f"Context exceeded cap: {n_tok} tokens"
    assert "[c0]" in ctx
    assert "[c3]" not in ctx, "Should have been truncated"
    logger.info(f"test_format_context: PASS  ({n_tok} tokens, cap=250)")


def test_cache() -> None:
    cache = RAGCache(cache_dir=f"{_TMP}/cache_unit")
    cache.set("k1", "hello", ttl_hours=1)
    assert cache.get("k1") == "hello"
    assert cache.get("missing_key") is None
    s = cache.stats()
    assert s["hit_count"] == 1 and s["miss_count"] == 1
    assert s["hit_rate"] == 0.5
    cache.clear()
    assert cache.get("k1") is None
    logger.info("test_cache: PASS")


def test_pipeline_full() -> None:
    retriever = _FakeRetriever()
    llm       = MockLLMClient()
    pipeline  = RAGPipeline(retriever, llm, use_cache=True, cache_dir=f"{_TMP}/pipeline")

    # ── First call: cache miss ────────────────────────────────────────────
    r1 = pipeline.answer(QUERY, k=3)
    logger.info(
        f"r1 | answer={r1.answer!r} citations={r1.citations} "
        f"conf={r1.confidence} tok={r1.prompt_tokens}+{r1.completion_tokens} "
        f"cost=${r1.cost_usd:.8f} latency={r1.latency_ms:.1f}ms"
    )
    assert not r1.cache_hit
    assert r1.answer == "Mock answer based on context."
    assert r1.citations == ["squad_7_chunk_0"]
    assert r1.confidence == 0.5
    assert r1.prompt_tokens > 50
    assert r1.completion_tokens > 0
    assert r1.cost_usd == 0.0          # MockLLMClient always returns 0
    assert len(r1.retrieved_chunks) == 3

    # ── Second call: cache hit ────────────────────────────────────────────
    r2 = pipeline.answer(QUERY, k=3)
    assert r2.cache_hit, "Second call must be a cache hit"
    assert r2.answer == r1.answer
    logger.info(f"r2 | cache_hit={r2.cache_hit}  stats={pipeline.cache.stats()}")

    stats = pipeline.cache.stats()
    assert stats["hit_count"] == 1
    assert stats["miss_count"] == 1
    assert stats["hit_rate"] == 0.5

    logger.info("test_pipeline_full: PASS")


def test_pipeline_bad_json_fallback() -> None:
    """LLM returning non-JSON should fall back to raw text."""

    class _BadLLM:
        provider = model = "bad"
        def chat(self, messages, **kw): return "Not JSON at all!"
        def estimate_cost(self, *a): return 0.0

    pipeline = RAGPipeline(
        _FakeRetriever(), _BadLLM(),
        use_cache=False, cache_dir=f"{_TMP}/bad_json",
    )
    r = pipeline.answer("anything", k=2)
    assert r.answer == "Not JSON at all!"
    assert r.citations == ["squad_7_chunk_0"]  # falls back to first chunk id
    assert r.confidence == 0.0
    logger.info("test_pipeline_bad_json_fallback: PASS")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def main() -> None:
    print("\n" + "=" * 55)
    print("  Generation module smoke tests")
    print("=" * 55)
    try:
        test_mock_llm()
        test_format_context()
        test_cache()
        test_pipeline_full()
        test_pipeline_bad_json_fallback()
        print("\nAll generation checks: PASS")
    finally:
        shutil.rmtree(_TMP, ignore_errors=True)


if __name__ == "__main__":
    main()
