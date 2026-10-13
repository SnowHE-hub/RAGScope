import importlib.util
import io
import json
from pathlib import Path

import pytest

_FILE = Path(__file__).resolve().parents[2] / "ui/chat_app.py"
spec = importlib.util.spec_from_file_location("ragscope_ui_v2", _FILE)
ui = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ui)


def test_missing_config_has_no_mock_or_ready_resources(tmp_path):
    config = ui.read_ui_config(tmp_path / "absent.json")
    assert config["models"] == config["indexes"] == []
    assert ui.ready_options([{"ready": False}, {"ready": True, "id": "one"}]) == [{"ready": True, "id": "one"}]
    invalid = tmp_path / "bad.json"
    invalid.write_text('{"models":{}}')
    with pytest.raises(ValueError):
        ui.read_ui_config(invalid)


def test_result_browser_rejects_corruption_and_unverified_rows(tmp_path):
    from ragscope.experiments.ledger import RecordStore
    store = RecordStore(tmp_path / "records.jsonl", {"version": "test"})
    store.add("q", {"answer": "offline test", "real_model": False})
    config, rows = ui.load_result_file(store.path)
    assert config == {"version": "test"} and len(rows) == 1
    raw = store.path.read_text()
    store.path.write_text(raw.replace("offline test", "corrupt data"))
    with pytest.raises(ValueError, match="Corrupt"):
        ui.load_result_file(store.path)
    store.path.write_text('{"record_id":"q"}\n')
    with pytest.raises(ValueError, match="校验和"):
        ui.load_result_file(store.path)


def test_index_manifest_validation(tmp_path):
    from ragscope.indexing.identity import signature, artifact
    chunks = tmp_path / "chunks.json"
    chunks.write_text("[]")
    identity = {"dataset": "offline-fixture"}
    manifest = tmp_path / "manifest.json"
    data = {"identity": identity, "signature": signature(identity), "count": 0, "artifacts": [artifact(chunks)]}
    manifest.write_text(json.dumps(data))
    assert ui.validate_index_entry({"manifest_path": str(manifest)})["count"] == 0
    chunks.write_text("changed")
    with pytest.raises(ValueError, match="校验和"):
        ui.validate_index_entry({"manifest_path": str(manifest)})


def test_probe_is_metadata_only_and_never_falls_back_to_mock(monkeypatch):
    import urllib.request
    assert ui.probe_model({"ready": False})[0] is False
    assert ui.probe_model({"ready": True})[0] is False
    paths = []
    class Opener:
        def open(self, path, timeout):
            paths.append(path)
            return io.BytesIO(b'{"data":[{"id":"test-model"}]}')
    monkeypatch.setattr(urllib.request, "build_opener", lambda *args: Opener())
    entry = {"ready": True, "model": "test-model", "provider": "llama.cpp", "base_url": "http://127.0.0.1:18081",
             "identity": {"revision": "r", "weight_sha256": "h", "backend_version": "v"}}
    assert ui.probe_model(entry)[0]
    assert paths == ["http://127.0.0.1:18081/v1/models"]
    entry["model"] = "absent"
    assert not ui.probe_model(entry)[0]


def test_scoped_retriever_uses_scope_and_explicit_depth():
    class Index:
        identity = {"fixture": "unit-test"}
        def search(self, question, **kwargs):
            assert kwargs["scope"] == "paper" and kwargs["candidate_depth"] == 50
            assert kwargs["method"] == "bm25" and kwargs["k"] == 10
            return [{"chunk_id": "offline"}]
    retriever = ui.ScopedRetriever(Index(), {"method": "bm25"}, "paper")
    assert retriever.retrieve("q")[0]["chunk_id"] == "offline"
    assert "ranking_ms" in retriever.last_timings


def test_streamlit_not_ready_app_renders_without_models(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest
    config = tmp_path / "ui.json"
    config.write_text(json.dumps({"models": [], "indexes": [], "result_files": []}))
    monkeypatch.setenv("RAGSCOPE_UI_CONFIG", str(config))
    app = AppTest.from_file(str(_FILE), default_timeout=15).run()
    assert not app.exception
    assert app.selectbox[0].value == "仅检索证据"
    assert app.button[0].disabled is True
    assert any("不会自动切换到 Mock" in item.value for item in app.info)


def test_streamlit_shows_actual_retrieval_and_validated_history(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest
    from ragscope.indexing.array_index import ArrayIndex
    from ragscope.experiments.ledger import RecordStore
    index_dir = tmp_path / "index"
    chunk = {"chunk_id": "test-chunk", "doc_id": "test-paper", "text": "Alpha is an offline engineering fixture.",
             "unit_spans": [], "metadata": {}, "canonical_start": 0, "canonical_end": 39}
    index = ArrayIndex([chunk], {"fixture": "offline-ui-test"})
    index.save(index_dir)
    store = RecordStore(tmp_path / "records.jsonl", {"fixture": "offline-ui-test"})
    store.add("q1", {"question": "unit question", "answer": "offline answer", "real_model": False})
    config = tmp_path / "ui.json"
    config.write_text(json.dumps({"models": [], "indexes": [{"id": "test-index", "ready": True, "method": "bm25",
                            "manifest_path": str(index_dir / "manifest.json")}],
                            "result_files": [str(store.path)], "default_scope": "test-paper"}))
    monkeypatch.setenv("RAGSCOPE_UI_CONFIG", str(config))
    app = AppTest.from_file(str(_FILE), default_timeout=15).run()
    assert not app.exception
    app.text_input[1].set_value("What is Alpha?")
    app.button[0].click().run()
    assert not app.exception
    assert any("test-chunk" in expander.label for expander in app.expander)
    next(button for button in app.button if button.label == "加载并校验记录").click().run()
    assert not app.exception
    assert any("已验证 1 条记录" in item.value for item in app.caption)
    assert any("Mock" in item.value for item in app.warning)
