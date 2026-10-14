import importlib.util
from pathlib import Path
from ragscope.experiments.ledger import RecordStore

path = Path(__file__).resolve().parents[2]/'scripts/v3/score_claim_metrics.py'
spec = importlib.util.spec_from_file_location('v3_claim_score_runner', path)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class FunctionalNLI:
    def __init__(self):
        self.calls = []

    def classify(self, premise, hypothesis):
        self.calls.append((premise, hypothesis))
        return {'verdict': 'SUPPORTED_PROXY', 'encoder_tokens': 3}


def test_uncalibrated_fragments_do_not_reach_nli(tmp_path):
    nli = FunctionalNLI()
    judge = runner.CachedJudge(nli, RecordStore(tmp_path/'cache.jsonl', {}), completion_gate=False)
    premise = {'text': 'The model used 128 samples.', 'eligible': True}
    assert judge(premise, {'text': '128', 'eligible': False})['verdict'] == 'UNKNOWN'
    assert judge(premise, {'text': 'The model used 128 samples.', 'eligible': True,
                          'reason': 'mechanical_quantity_qa_v1'})['verdict'] == 'UNKNOWN'
    assert not nli.calls


def test_cached_relationships_resume_without_repeating_model_execution(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, 'require_ram', lambda *args: None)
    path = tmp_path/'cache.jsonl'
    p = {'text': 'The model did not use Delta.', 'eligible': True}
    h = {'text': 'Delta was not used.', 'eligible': True}
    first = FunctionalNLI()
    judge = runner.CachedJudge(first, RecordStore(path, {}), completion_gate=True)
    judge(p, h); judge(p, h)
    assert len(first.calls) == 1
    second = FunctionalNLI()
    resumed = runner.CachedJudge(second, RecordStore(path, {}), completion_gate=True)
    resumed(p, h)
    assert not second.calls
    resumed(p, {'text': 'Delta was used.', 'eligible': True})
    assert len(second.calls) == 1
