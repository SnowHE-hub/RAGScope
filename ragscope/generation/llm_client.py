"""Local-only generation transports; construction never loads models or reads keys."""
from __future__ import annotations

import ipaddress
import json
import os
import re
import socket
import urllib.error
import urllib.request
from urllib.parse import urlsplit

from ragscope.generation.prompts import OUTPUT_SCHEMA


class GenerationError(RuntimeError):
    def __init__(self, kind, message):
        super().__init__(message)
        self.kind = kind


def _local_url(url):
    parsed = urlsplit(url)
    if parsed.scheme != "http" or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("Local backend requires an unauthenticated loopback HTTP URL")
    host = parsed.hostname
    if host != "localhost":
        try:
            if not ipaddress.ip_address(host).is_loopback:
                raise ValueError("Backend must bind to loopback")
        except (ValueError, TypeError) as exc:
            raise ValueError("Backend must bind to a literal loopback address or localhost") from exc
    return url.rstrip("/")


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise GenerationError("transport_error", "Local backend redirect rejected")


class LocalLLMClient:
    """Ollama or llama.cpp, one synchronous request, no hidden retries.

    Callers start and verify a project-owned service separately. Device choice is
    explicit; CPU-only environment rejects any GPU request before HTTP access.
    Identity fields describe pinned weights/backend, not just a mutable tag.
    """
    real_model = True

    def __init__(self, model, *, provider="ollama", base_url="http://127.0.0.1:11435",
                 revision="", weight_sha256="", quantization="", backend_version="",
                 chat_template_hash="", timeout=180, seed=42, num_ctx=4096,
                 num_gpu=0, think=False, json_constraint=True, keep_alive="5m",
                 prompt_token_counter=None, service_device_verified=None):
        if provider not in {"ollama", "llama.cpp"}:
            raise ValueError("Only local ollama and llama.cpp backends are enabled")
        if timeout <= 0 or num_ctx <= 0 or num_gpu < 0:
            raise ValueError("Invalid timeout/context/device configuration")
        self.provider, self.model = provider, model
        self.base_url = _local_url(base_url)
        self.timeout, self.seed, self.num_ctx, self.num_gpu = timeout, seed, num_ctx, num_gpu
        self.think, self.json_constraint, self.keep_alive = think, json_constraint, keep_alive
        self.prompt_token_counter = prompt_token_counter
        self.service_device_verified = service_device_verified
        self.identity = dict(provider=provider, model=model, revision=revision,
                             weight_sha256=weight_sha256, quantization=quantization,
                             backend_version=backend_version, chat_template_hash=chat_template_hash,
                             base_url=self.base_url, timeout=timeout, seed=seed, num_ctx=num_ctx,
                             num_gpu=num_gpu, think=think, json_constraint=json_constraint,
                             service_device_verified=service_device_verified)
        self.last_metadata = {}

    def _request(self, path, payload):
        if os.environ.get("CUDA_VISIBLE_DEVICES") == "" and self.num_gpu != 0:
            raise GenerationError("device_policy_error", "GPU request prohibited in CPU-only process")
        if (os.environ.get("CUDA_VISIBLE_DEVICES") == "" and self.provider == "llama.cpp"
                and self.service_device_verified != "cpu"):
            raise GenerationError("device_policy_error", "llama.cpp server CPU-only launch must be verified")
        request = urllib.request.Request(self.base_url + path,
                    data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"})
        try:
            # Explicitly ignore user/system proxy settings for local-only calls.
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())
            with opener.open(request, timeout=self.timeout) as response:
                data = json.load(response)
            if data.get("error"):
                message = str(data["error"])
                raise GenerationError("oom" if "out of memory" in message.lower() else "transport_error", message)
            return data
        except (TimeoutError, socket.timeout) as exc:
            raise GenerationError("timeout", str(exc)) from exc
        except urllib.error.HTTPError as exc:
            detail = exc.read(4096).decode("utf-8", errors="replace")
            kind = "oom" if "out of memory" in detail.lower() else "transport_error"
            raise GenerationError(kind, f"HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            kind = "timeout" if isinstance(exc.reason, (TimeoutError, socket.timeout)) else "transport_error"
            raise GenerationError(kind, str(exc.reason)) from exc
        except (ValueError, TypeError) as exc:
            raise GenerationError("transport_error", "Invalid local server response") from exc

    def chat(self, messages, max_tokens=512, temperature=0.0):
        self.last_metadata = {}
        actual_input = self.prompt_token_counter(messages) if self.prompt_token_counter else None
        if actual_input is not None and actual_input + max_tokens > self.num_ctx:
            raise GenerationError("input_budget_error", "Full prompt plus output reserve exceeds model context")
        if self.provider == "ollama":
            payload = dict(model=self.model, messages=messages, stream=False, think=self.think,
                           keep_alive=self.keep_alive,
                           options=dict(num_predict=max_tokens, temperature=temperature,
                                        seed=self.seed, num_ctx=self.num_ctx, num_gpu=self.num_gpu))
            if self.json_constraint:
                payload["format"] = OUTPUT_SCHEMA
            data = self._request("/api/chat", payload)
            self.last_metadata = {key: data.get(key) for key in (
                "model", "done_reason", "prompt_eval_count", "eval_count", "total_duration",
                "load_duration", "prompt_eval_duration", "eval_duration")}
            content = data.get("message", {}).get("content")
        else:
            payload = dict(model=self.model, messages=messages, stream=False, max_tokens=max_tokens,
                           temperature=temperature, seed=self.seed,
                           chat_template_kwargs={"enable_thinking": self.think})
            if self.json_constraint:
                payload["response_format"] = {"type": "json_object", "schema": OUTPUT_SCHEMA}
            data = self._request("/v1/chat/completions", payload)
            self.last_metadata = {"usage": data.get("usage"), "timings": data.get("timings"),
                                  "model": data.get("model"), "system_fingerprint": data.get("system_fingerprint")}
            try:
                content = data["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError) as exc:
                raise GenerationError("transport_error", "Missing llama.cpp answer") from exc
        self.last_metadata.update(actual_input_tokens=actual_input, first_token_ms=None,
                                  timing_mode="nonstreaming", thinking_requested=self.think)
        observed_input = (data.get("prompt_eval_count") if self.provider == "ollama"
                          else (data.get("usage") or {}).get("prompt_tokens"))
        self.last_metadata.update(prompt_token_delta=(observed_input-actual_input
                                  if isinstance(observed_input, int) and actual_input is not None else None),
                                  token_count_match=(observed_input == actual_input
                                  if isinstance(observed_input, int) and actual_input is not None else None))
        if not isinstance(content, str):
            raise GenerationError("transport_error", "Missing textual model response")
        return content

    def estimate_cost(self, prompt_tokens, completion_tokens):
        return 0.0

    def unload(self):
        if self.provider == "ollama":
            return self._request("/api/generate", {"model": self.model, "keep_alive": 0})
        raise ValueError("Stop the project-owned llama.cpp process using its recorded lifecycle")


class OllamaLLMClient(LocalLLMClient):
    def __init__(self, model="qwen2.5:7b", base_url="http://127.0.0.1:11435", **kwargs):
        super().__init__(model, provider="ollama", base_url=base_url, **kwargs)


class LlamaCppLLMClient(LocalLLMClient):
    def __init__(self, model, base_url="http://127.0.0.1:8081", **kwargs):
        super().__init__(model, provider="llama.cpp", base_url=base_url, **kwargs)


class LLMClient(LocalLLMClient):
    """Compatibility name, now explicitly local-only; cloud access is disabled."""
    def __init__(self, provider="ollama", model="qwen2.5:7b", api_key=None, **kwargs):
        if api_key is not None:
            raise ValueError("API keys are not used by the local-only v2 client")
        super().__init__(model, provider=provider, **kwargs)


class MockLLMClient:
    """Offline function-test double. Never eligible for formal model evaluation."""
    provider = "mock"
    model = "mock"
    real_model = False
    identity = {"provider": "mock", "model": "mock", "revision": "v2"}
    last_metadata = {}

    def chat(self, messages, max_tokens=512, temperature=0.0):
        user = next((m["content"] for m in messages if m.get("role") == "user"), "")
        context = user.split("Context:\n", 1)[-1]
        match = re.search(r"^\[([^\]]+)\]", context, re.MULTILINE)
        return json.dumps({"answer": "Mock answer based on context.",
                           "citations": [match.group(1)] if match else [],
                           "confidence": 0.5, "abstain": not bool(match)})

    def estimate_cost(self, prompt_tokens, completion_tokens):
        return 0.0
