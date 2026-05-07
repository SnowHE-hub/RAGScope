"""
LLMClient  — thin OpenAI wrapper with cost estimation.
MockLLMClient — drop-in replacement for offline / retrieval-only evaluation.
"""

from __future__ import annotations

import json
import os
import re
from typing import Dict, List, Optional

from dotenv import load_dotenv
from loguru import logger

load_dotenv()

# gpt-4o-mini pricing (USD per 1 M tokens, as of 2024-07)
_COST_INPUT_PER_M  = 0.150   # $0.15 / 1M prompt tokens
_COST_OUTPUT_PER_M = 0.600   # $0.60 / 1M completion tokens


class LLMClient:
    """Synchronous OpenAI chat client."""

    def __init__(
        self,
        provider: str = "openai",
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
    ) -> None:
        self.provider = provider
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "No API key found.  Set OPENAI_API_KEY in .env or pass api_key=."
            )
        import openai
        self._client = openai.OpenAI(api_key=self.api_key)
        logger.info(f"LLMClient ready — model={model}")

    def chat(
        self,
        messages: List[Dict],
        max_tokens: int = 512,
        temperature: float = 0.1,
    ) -> str:
        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content or ""

    def estimate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        return (
            prompt_tokens   * _COST_INPUT_PER_M  / 1_000_000
            + completion_tokens * _COST_OUTPUT_PER_M / 1_000_000
        )


class MockLLMClient:
    """
    No-op LLM for CI / retrieval-quality evaluation.

    Returns well-formed JSON using the first retrieved chunk_id as citation,
    so the full pipeline (retrieve → prompt → parse → cache) can be exercised
    without an OpenAI key.
    """

    provider = "mock"
    model = "mock"

    def chat(
        self,
        messages: List[Dict],
        max_tokens: int = 512,
        temperature: float = 0.1,
    ) -> str:
        # Extract the first [chunk_id] reference from the user message context
        user_content = next(
            (m["content"] for m in messages if m.get("role") == "user"), ""
        )
        match = re.search(r"\[([^\]]+)\]", user_content)
        first_chunk_id = match.group(1) if match else "unknown"

        return json.dumps(
            {
                "answer": "Mock answer based on context.",
                "citations": [first_chunk_id],
                "confidence": 0.5,
            }
        )

    def estimate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        return 0.0


class OllamaLLMClient:
    """
    Local LLM client via Ollama's REST API.

    Requires `ollama serve` to be running (e.g. inside WSL) and the chosen
    model to be pulled: `ollama pull qwen2.5:7b`.

    Windows host accesses it at http://localhost:11434 (WSL port-forwarding
    is automatic in WSL2).
    """

    provider = "ollama"

    def __init__(
        self,
        model: str = "qwen2.5:7b",
        base_url: str = "http://localhost:11434",
    ) -> None:
        self.model    = model
        self.base_url = base_url.rstrip("/")
        logger.info(f"OllamaLLMClient ready — model={model} base_url={self.base_url}")

    def chat(
        self,
        messages: List[Dict],
        max_tokens: int = 512,
        temperature: float = 0.1,
    ) -> str:
        import requests

        payload = {
            "model":   self.model,
            "messages": messages,
            "stream":  False,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
            },
        }
        try:
            resp = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()
            return data["message"]["content"]
        except requests.exceptions.ConnectionError:
            raise RuntimeError(
                f"Cannot connect to Ollama at {self.base_url}. "
                "Make sure `ollama serve` is running in WSL."
            )
        except requests.exceptions.Timeout:
            raise RuntimeError("Ollama request timed out (120 s).")
        except (KeyError, TypeError) as exc:
            raise RuntimeError(f"Unexpected Ollama response format: {exc}")

    def estimate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        return 0.0
