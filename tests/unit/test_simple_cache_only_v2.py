import json
import socket
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.materialize_simple_cache_only_v2 import CacheOnlyEmbedding, CacheOnlyPending, cached_item, no_network
from ragscope.indexing.identity import artifact


def service(tmp_path):
    tokenizer = SimpleNamespace(encode=lambda text: list(text))
    return CacheOnlyEmbedding('http://127.0.0.1:11436', {'dimensions': 2, 'fixture': True}, tokenizer, tmp_path)


def test_verified_cache_returns_original_vector_without_http(tmp_path, monkeypatch):
    backend = service(tmp_path)
    backend.cache.put('a', np.array([1, 0], dtype=np.float32))
    (backend.cache.path / (backend.cache.key('a')+'.usage.json')).write_text(json.dumps({
        'key': backend.cache.key('a'), 'body_tokens': 1, 'usage': {'prompt_tokens': 2}}))
    monkeypatch.setattr('psutil.virtual_memory', lambda: SimpleNamespace(available=2*1024**3))
    with no_network() as attempts:
        vector = backend.embed(['a'])
    assert vector.tolist() == [[1, 0]] and attempts == []
    assert backend.last_metrics[0]['cache_hit'] is True
    assert cached_item(backend, 'a')['backend_usage_observed']


def test_missing_cache_stops_pending_before_network_or_failure_row(tmp_path, monkeypatch):
    monkeypatch.setattr('psutil.virtual_memory', lambda: SimpleNamespace(available=2*1024**3))
    with no_network() as attempts, pytest.raises(CacheOnlyPending, match='no verified vector'):
        service(tmp_path).embed(['uncached'])
    assert attempts == []
    assert not issubclass(CacheOnlyPending, Exception)


def test_transport_guard_rejects_accidental_socket_call():
    with no_network() as attempts, pytest.raises(CacheOnlyPending, match='network access'):
        with socket.socket() as conn:
            conn.connect(('127.0.0.1', 11436))
    assert len(attempts) == 1


def test_only_explicitly_bound_historical_na_is_reused_without_fabrication(tmp_path, monkeypatch):
    backend = service(tmp_path)
    backend.cache.put('a', np.array([1, 0], dtype=np.float32))
    key = backend.cache.key('a')
    allow = {key: {'body_tokens': 1, 'vector': artifact(backend.cache.path/(key+'.npy')),
                   'vector_manifest': artifact(backend.cache.path/(key+'.json'))}}
    assert cached_item(backend, 'a', allow)['backend_usage_observed'] is False
    monkeypatch.setattr(CacheOnlyEmbedding, 'historical_usage', allow)
    monkeypatch.setattr('psutil.virtual_memory', lambda: SimpleNamespace(available=2*1024**3))
    with no_network() as attempts:
        assert backend.embed(['a']).tolist() == [[1, 0]]
    assert attempts == [] and backend.last_metrics[0]['backend_usage'] is None
    assert not (backend.cache.path/(key+'.usage.json')).exists()
    allow[key]['body_tokens'] = 2
    with pytest.raises(CacheOnlyPending, match='provenance mismatch'):
        cached_item(backend, 'a', allow)
