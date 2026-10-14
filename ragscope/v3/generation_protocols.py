"""Controlled protocol differences over the existing pinned local backend."""
from __future__ import annotations
import copy
import json
from pathlib import Path
import re
import time

from ragscope.generation.prompts import RAG_SYSTEM_PROMPT, RAG_USER_TEMPLATE, visible_output_schema
from ragscope.generation.context import render_context
from ragscope.generation.resources import RequestResourceSampler
from ragscope.indexing.identity import artifact, signature
from ragscope.v3.local_runtime import CapturedClient

MINIMUM_ANSWER = (' In answer give only the minimum sufficient answer: the requested facts and essential '
    'conditions. Do not repeat the question or context, add a preface, or append unrelated explanation.')


def protocol_messages(question, bundle, protocol):
    if protocol not in {'G0', 'G1', 'G2', 'G3'}:
        raise ValueError('Unknown preregistered generation protocol')
    system = RAG_SYSTEM_PROMPT + (MINIMUM_ANSWER if protocol != 'G0' else '')
    return [{'role': 'system', 'content': system}, {'role': 'user',
        'content': RAG_USER_TEMPLATE.format(question=question, context=render_context(bundle))}]


def card_sampling(path, model):
    lines = Path(path).read_text('utf-8').splitlines()
    matches = []
    for index, line in enumerate(lines):
        lower = line.casefold()
        target = 'non-thinking mode for text tasks' in lower if model.endswith('2B') else 'non-thinking' in lower and 'general tasks' in lower
        if not target:
            continue
        pattern = r'(temperature|top_p|top_k|min_p|presence_penalty|repetition_penalty)\s*=\s*([0-9.]+)'
        pairs = dict(re.findall(pattern, line))
        if len(pairs) != 6 and index+1 < len(lines):
            pairs.update(re.findall(pattern, lines[index+1]))
        if len(pairs) != 6:
            raise ValueError('Official sampling line is incomplete')
        values = {k: int(v) if k == 'top_k' else float(v) for k, v in pairs.items()}
        matches.append((index+1, values))
    if not matches or len({signature(value) for _, value in matches}) != 1:
        raise ValueError('No unambiguous pinned non-thinking recommendation')
    values = dict(matches[0][1]); temperature = values.pop('temperature')
    values['repeat_penalty'] = values.pop('repetition_penalty')
    return {'temperature': temperature, 'options': values, 'model_card': artifact(path),
            'source_lines': [number for number, _ in matches],
            'parameter_name_mapping': {'repetition_penalty': 'ollama.repeat_penalty'},
            'workload': 'text general tasks; not VL or reasoning-specific alternatives'}


class ProtocolClient(CapturedClient):
    def __init__(self, original, service, *, protocol, seed=42, sampling=None):
        identity = original.identity
        options = {k: identity[k] for k in ('revision', 'weight_sha256', 'quantization', 'chat_template_hash',
            'timeout', 'think', 'backend_runtime_config') if k in identity}
        super().__init__(identity['model'], provider='ollama', base_url=service.base_url,
            backend_version=service.record['backend_version'], num_gpu=999, num_ctx=4096,
            service_device_verified='gpu', prompt_token_counter=original.prompt_token_counter,
            seed=seed, json_constraint=protocol != 'G2', **options)
        self.sampling = copy.deepcopy(sampling['options']) if sampling else {}
        self.service_receipt = service.path
        self.sent_payload = None
        if sampling:
            self.identity['sampling_options'] = self.sampling
            self.identity['sampling_source'] = sampling['model_card']

    def _request(self, path, payload):
        payload = copy.deepcopy(payload)
        if path == '/api/chat':
            payload['options'].update(self.sampling)
            self.sent_payload = copy.deepcopy(payload)
        return super()._request(path, payload)

    def chat(self, *args, **kwargs):
        sampler = RequestResourceSampler(self.service_receipt, self.base_url, self.model).start()
        started = time.perf_counter()
        try:
            return super().chat(*args, **kwargs)
        finally:
            response_finished = time.perf_counter()
            resources = sampler.finish()
            self.last_metadata.update(resources=resources,
                client_request_wall_ms=(response_finished-started)*1000,
                telemetry_finalize_ms=(time.perf_counter()-response_finished)*1000,
                timing_note='nonstreaming; HTTP time separate from telemetry finalization; no measured TTFT')


def protocol_schema(bundle, protocol):
    return None if protocol == 'G2' else visible_output_schema(bundle['pieces'])
