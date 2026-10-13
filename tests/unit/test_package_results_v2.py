"""Packaging tests use explicit offline fixtures, never experimental evidence."""
import json
import zipfile

import pytest

from scripts import package_results_v2 as package
from test_analysis_report_v2 import execute


def acceptance(status='PASS'):
    return {'criteria': [{'criterion_id': str(i), 'status': status, 'evidence_paths': [],
                         'command': 'offline fixture', 'exit_code': 0,
                         'observed_value': 'fixture only', 'limitations': ['not real experiment']}
                        for i in range(1, 16)]}


def test_final_package_keeps_failures_and_excludes_unlisted_files(tmp_path, monkeypatch):
    monkeypatch.setattr(package, 'runtime_home', lambda: tmp_path)
    campaign, reports = tmp_path/'runs/campaign', tmp_path/'runs/final'
    good = '{"answer":"source","citations":["c"],"abstain":false}'
    execute(campaign, outputs=[good]*3+[TimeoutError('fixture'), good])
    original = (campaign/'records/E2_fixture.jsonl').read_bytes()
    reports.mkdir()
    (reports/'ACCEPTANCE.json').write_text(json.dumps(acceptance()))
    (reports/'FINAL_REPORT.md').write_text('Fixture report: '+str(tmp_path))
    (reports/'unrelated-private.txt').write_text('must remain outside package')
    output = tmp_path/'exports/fixture.zip'
    result = package.build_bundle(campaign, reports, output)
    assert result['observed_records'] == 2 and result['uploaded'] is False
    with zipfile.ZipFile(output) as bundle:
        assert not any('unrelated-private' in name for name in bundle.namelist())
        exported = [json.loads(line) for line in bundle.read('results/fixture.portable.jsonl').splitlines()]
        assert sum(bool(row['exported_record'].get('error')) for row in exported) == 1
        assert all(row['original_record_hash'] == row['exported_record']['record_hash'] for row in exported)
        assert str(tmp_path) not in bundle.read('reports/FINAL_REPORT.md').decode()
    assert original == (campaign/'records/E2_fixture.jsonl').read_bytes()
    with pytest.raises(FileExistsError):
        package.build_bundle(campaign, reports, output)


def test_pending_acceptance_never_becomes_a_final_bundle(tmp_path, monkeypatch):
    monkeypatch.setattr(package, 'runtime_home', lambda: tmp_path)
    reports = tmp_path/'runs/final'; reports.mkdir(parents=True)
    (reports/'ACCEPTANCE.json').write_text(json.dumps(acceptance('PENDING')))
    with pytest.raises(ValueError, match='Unfinished'):
        package.build_bundle(tmp_path/'runs/campaign', reports, tmp_path/'out.zip')
    assert not (tmp_path/'out.zip').exists()
