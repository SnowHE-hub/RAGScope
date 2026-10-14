from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import numpy as np
import pytest
import requests

from scripts.materialize_squad_v2 import PlannedStop, QueryOnlyEmbedding
from ragscope.indexing.identity import signature
from ragscope.indexing.local_embedding import LocalEmbeddingService


@pytest.fixture
def backend(tmp_path, monkeypatch):
    monkeypatch.setattr(QueryOnlyEmbedding, 'allowed_text_hashes', {signature('query')})
    monkeypatch.setattr(QueryOnlyEmbedding, 'backend_failed', False)
    monkeypatch.setattr(QueryOnlyEmbedding, 'query_observations', [])
    monkeypatch.setattr(QueryOnlyEmbedding, 'deadline', datetime.now(timezone.utc)+timedelta(hours=1))
    monkeypatch.setattr('psutil.virtual_memory', lambda: SimpleNamespace(available=2*1024**3))
    return QueryOnlyEmbedding('http://127.0.0.1:11436', {'dimensions': 2},
                              SimpleNamespace(encode=lambda text: list(text)), tmp_path)


def test_frozen_query_uses_original_cached_vector_without_http(backend, monkeypatch):
    backend.cache.put('query', np.array([1, 0], dtype=np.float32))
    monkeypatch.setattr(requests, 'post', lambda *args, **kwargs: pytest.fail('Unexpected HTTP'))
    assert backend.embed(['query']).tolist() == [[1, 0]]
    assert backend.query_observations[0]['cache_hit'] is True
    with pytest.raises(PlannedStop, match='only the frozen'):
        backend.embed(['corpus body'])


def test_first_actual_network_error_is_retained_then_next_query_is_pending(backend, monkeypatch):
    calls = []
    def fail(*args, **kwargs):
        calls.append(1)
        raise requests.ConnectionError('offline fixture')
    monkeypatch.setattr(requests, 'post', fail)
    with pytest.raises(requests.ConnectionError):
        backend.embed(['query'])
    with pytest.raises(PlannedStop, match='Prior actual backend failure'):
        backend.embed(['query'])
    assert calls == [1]
    assert not issubclass(PlannedStop, Exception)


def test_low_ram_and_deadline_leave_unattempted_queries_pending(backend, monkeypatch):
    monkeypatch.setattr(LocalEmbeddingService, 'embed', lambda *args: pytest.fail('Guard did not stop'))
    monkeypatch.setattr('psutil.virtual_memory', lambda: SimpleNamespace(available=1024**3-1))
    with pytest.raises(PlannedStop, match='Available RAM'):
        backend.embed(['query'])
    monkeypatch.setattr(QueryOnlyEmbedding, 'deadline', datetime.now(timezone.utc)-timedelta(seconds=1))
    with pytest.raises(PlannedStop, match='deadline'):
        backend.embed(['query'])
