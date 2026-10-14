from types import SimpleNamespace
import pytest
from ragscope.v3.gpu import gpu_observation


class FakeProcess:
    def __init__(self, pid):
        self.pid = pid

    def name(self):
        return 'ChatGPT.exe'

    def exe(self):
        return 'C:\\Program Files\\WindowsApps\\OpenAI.Codex_1.0_x64\\app\\ChatGPT.exe'

    def create_time(self):
        return 123.0


def fake_run(args, **kwargs):
    if '--query-compute-apps=pid,process_name' in args:
        return SimpleNamespace(stdout='77, C:\\app\\ChatGPT.exe\n')
    return SimpleNamespace(stdout='0, test GPU, 12, 8192, 0\n')


def test_packaged_codex_host_is_identified_without_termination(monkeypatch):
    monkeypatch.setattr('ragscope.v3.gpu.subprocess.run', fake_run)
    monkeypatch.setattr('ragscope.v3.gpu.psutil.Process', FakeProcess)
    result = gpu_observation()
    assert result['compute_apps'][0]['permitted']
    assert result['compute_apps'][0]['create_time'] == 123


def test_same_process_name_outside_verified_package_is_not_whitelisted(monkeypatch):
    monkeypatch.setattr('ragscope.v3.gpu.subprocess.run', fake_run)
    monkeypatch.setattr('ragscope.v3.gpu.psutil.Process', FakeProcess)
    monkeypatch.setattr(FakeProcess, 'exe', lambda self: 'C:\\unrelated\\ChatGPT.exe')
    with pytest.raises(RuntimeError, match='Unknown GPU consumer retained'):
        gpu_observation()
    assert gpu_observation([77])['compute_apps'][0]['permitted']
