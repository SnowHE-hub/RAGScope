import io
import json

import pytest

from ragscope.indexing.identity import artifact, atomic_json, file_hash
from ui.model_identity import tag_matches, verify_model_package


@pytest.fixture
def package(tmp_path, monkeypatch):
    weight = tmp_path/'tiny-offline-weight'
    weight.write_bytes(b'offline-weight-fixture-not-a-model')
    digest = file_hash(weight)
    target = tmp_path/'blobs'/('sha256-'+digest)
    target.parent.mkdir()
    weight.rename(target)
    manifest = tmp_path/'manifests/registry.ollama.ai/library/local-fixture/tag'
    atomic_json(manifest, {'layers': [{'mediaType': 'application/vnd.ollama.image.model',
        'digest': 'sha256:'+digest, 'size': target.stat().st_size}]})
    entry = {'provider': 'ollama', 'model': 'local-fixture:tag', 'base_url': 'http://127.0.0.1:11435',
             'identity': {'weight_sha256': digest}, 'ollama_manifest': artifact(manifest)}
    rows = [{'name': entry['model'], 'digest': entry['ollama_manifest']['sha256']}]
    class Opener:
        def open(self, url, timeout):
            assert url == entry['base_url']+'/api/tags'
            return io.BytesIO(json.dumps({'models': rows}).encode())
    monkeypatch.setattr('urllib.request.build_opener', lambda *args: Opener())
    service = tmp_path/'service.json'
    atomic_json(service, {'models_path': str(tmp_path), 'environment': {'OLLAMA_MODELS': str(tmp_path)}})
    return entry, rows, service, target


def test_verified_tag_and_actual_weight_bytes(package):
    entry, rows, service, _ = package
    assert tag_matches(entry, rows)
    verify_model_package(entry, service)


def test_same_name_different_digest_is_not_ready(package):
    entry, rows, service, _ = package
    rows[0]['digest'] = 'another-digest'
    assert not tag_matches(entry, rows)
    with pytest.raises(ValueError, match='digest'):
        verify_model_package(entry, service)


def test_same_tag_corrupt_weight_or_other_service_is_rejected(package, tmp_path):
    entry, rows, service, weight = package
    weight.write_bytes(b'corrupted')
    with pytest.raises(ValueError, match='实际字节'):
        verify_model_package(entry, service)
    atomic_json(service, {'models_path': str(tmp_path/'different'), 'environment': {'OLLAMA_MODELS': str(tmp_path/'different')}})
    with pytest.raises(ValueError, match='不属于'):
        verify_model_package(entry, service)
