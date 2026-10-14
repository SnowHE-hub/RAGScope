"""Offline behavior contracts; these fixtures are not real model results."""
from copy import deepcopy
import io
import json

import pytest

from scripts.configure_ui_v2 import assemble_candidate, build_candidate, metadata_get, validate_selection
from ragscope.indexing.identity import artifact, signature


def selected_fixture():
    identity = {'group': {'id': 'R512'}, 'selected_doc_ids': [str(i) for i in range(509)],
                'scope': 'global_or_explicit_paper', 'data_manifest_sha256': 'data'}
    index = {'identity': identity, 'signature': signature(identity), 'count': 6551}
    selection = {'split': 'dev', 'selected': {'structured': 'R512'}, 'data_manifest_sha256': 'data'}
    context = {'split': 'dev', 'test_used_for_selection': False,
               'candidate_selection_sha256': 'chosen', 'selected': {'R512': 'neighbor'}}
    return index, selection, context


def test_selection_rejects_test_or_wrong_corpus():
    index, choice, context = selected_fixture()
    assert validate_selection(index, choice, context, 'chosen') == 'neighbor'
    invalid = deepcopy(context)
    invalid['test_used_for_selection'] = True
    with pytest.raises(ValueError):
        validate_selection(index, choice, invalid, 'chosen')
    invalid = deepcopy(index)
    invalid['identity']['selected_doc_ids'].pop()
    invalid['signature'] = signature(invalid['identity'])
    with pytest.raises(ValueError):
        validate_selection(invalid, choice, context, 'chosen')


def test_offline_embedding_cannot_mark_neural_paths_ready():
    index, _, _ = selected_fixture()
    config = assemble_candidate(index, 'manifest', 'hash', 'chunks', 'neighbor',
        {'ready': False}, {'ready': True}, [{'id': 'qwen_2b', 'ready': False}], {}, [])
    assert [r['method'] for r in config['indexes'] if r['ready']] == ['bm25']
    assert all(r['expansion'] == 'neighbor' for r in config['indexes'])
    assert config['default_model'] == 'qwen_2b'
    assert config['routers'] == [] and config['default_router'] == 'fixed'


def test_publication_outside_runtime_candidate_is_rejected_before_reads(tmp_path):
    published = tmp_path / 'configs/ui.json'
    published.parent.mkdir()
    published.write_text('preserve')
    with pytest.raises(ValueError, match='no project publication'):
        build_candidate(tmp_path, 'campaign', published)
    assert published.read_text() == 'preserve'


def test_metadata_probe_is_get_only_and_does_not_follow_model_routes(monkeypatch):
    calls = []
    class Opener:
        def open(self, url, timeout):
            calls.append((url, timeout))
            return io.BytesIO(b'{"version":"0.34.4"}')
    monkeypatch.setattr('urllib.request.build_opener', lambda *args: Opener())
    assert metadata_get('http://127.0.0.1:11435', '/api/version')['available']
    with pytest.raises(ValueError):
        metadata_get('http://127.0.0.1:11435', '/api/generate')
    assert calls == [('http://127.0.0.1:11435/api/version', 2)]


def test_ui_query_tokenizer_uses_separate_frozen_tokenizer_revision(monkeypatch, tmp_path):
    from ui.chat_app import ScopedRetriever
    import ragscope.indexing.local_embedding as backend
    observed = []
    monkeypatch.setattr(backend, 'HFJsonTokenizer', lambda *args: observed.append(args) or object())
    class Service:
        def __init__(self, url, identity, tokenizer, cache, limit):
            self.identity = identity
        def embed_query(self, text):
            return [1.0]
    class Index:
        identity = {}
        def search(self, *args, **kwargs):
            return []
    monkeypatch.setattr(backend, 'LocalEmbeddingService', Service)
    entry = {'method': 'hybrid', 'embedding': {'ready': True, 'service_device_verified': 'cpu',
        'base_url': 'http://127.0.0.1:11436', 'identity': {'model_id': 'GGUF', 'revision': 'weights-rev'},
        'tokenizer_path': 'tokenizer.json', 'tokenizer_id': 'Qwen/tokenizer',
        'tokenizer_revision': 'tokenizer-rev', 'cache_dir': str(tmp_path)}}
    assert ScopedRetriever(Index(), entry, None).retrieve('test') == []
    assert observed == [('tokenizer.json', 'Qwen/tokenizer', 'tokenizer-rev')]


def test_ui_index_entry_rejects_changed_manifest_even_if_internally_consistent(tmp_path):
    from ui.chat_app import validate_index_entry
    chunks = tmp_path / 'chunks.json'
    chunks.write_text('[]')
    manifest = tmp_path / 'manifest.json'
    identity = {'fixture': 1}
    manifest.write_text(json.dumps({'identity': identity, 'signature': signature(identity),
                                    'artifacts': [artifact(chunks)]}))
    with pytest.raises(ValueError, match='固定版本'):
        validate_index_entry({'manifest_path': str(manifest), 'manifest_sha256': 'another-manifest'})
