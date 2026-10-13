"""Offline evidence-integrity fixtures; never actual model results."""
import json
from pathlib import Path
import pytest

from scripts.audit_final_report_v2 import audit_report, report_cells, compare_cells, HEADER
from scripts.audit_generation_numbers_v2 import sha
from ragscope.experiments.generation_campaign import freeze_campaign, run_model, GenerationBudget
from ragscope.experiments.analysis_report import build_analysis, markdown_table
from test_generation_campaign_v2 import sample, IDENTITY, UNIT, OfflineContractClient
from test_analysis_report_v2 import CONFIG


def table():
    return '|'+ '|'.join(HEADER)+'|\n|'+'|'.join(['---']*9)+'|\n'+''.join(
        f'|E2|fixture|{method}|COMPLETE|1/1|0|0.1235|1.0000|12.3457|\n'
        for method in ('R0', 'R1', 'R2'))


def test_display_cells_rounding_and_report_tampering():
    cells = report_cells(table())
    values = {'failed_n': 0, 'token_f1': .12345678,
              'context_full_reference_coverage': 1., 'generation_p50_ms': 12.345678}
    raw = {'all_checks': [{**cell, 'recomputed': values[cell['metric']], 'n_rows': 1,
                          'raw_record_hashes': ['offline-fixture']} for cell in cells]}
    assert all(row['match'] for row in compare_cells(cells, raw))
    changed = report_cells(table().replace('0.1235', '0.9235', 1))
    assert not compare_cells(changed, raw)[1]['match']
    assert cells[0]['report_line'] == 3 and len(cells) == 12


@pytest.mark.parametrize('change', ['duplicate', 'few', 'nan', 'wrong_group'])
def test_missing_or_ambiguous_report_evidence_rejected(change):
    text = table()
    if change == 'duplicate': text += text
    if change == 'few': text = text.replace('|R2|COMPLETE|', '|R2|INCOMPLETE|')
    if change == 'nan': text = text.replace('0.1235', 'NaN', 1)
    if change == 'wrong_group':
        with pytest.raises(ValueError, match='no independently'):
            compare_cells(report_cells(text), {'all_checks': []})
    else:
        with pytest.raises(ValueError): report_cells(text)


def test_actual_report_text_is_bound_to_fresh_raw_audit(tmp_path):
    campaign, corpus = tmp_path/'campaign', tmp_path/'corpus'
    inputs = {method: sample('same-question') for method in ('R0', 'R1', 'R2')}
    groups = [{'experiment_id': 'E2', 'method': method, 'model_ids': ['fixture'],
               'input_ids': [method]} for method in inputs]
    freeze_campaign(campaign, {'fixture': IDENTITY}, groups, inputs)
    run_model(campaign, 'fixture', OfflineContractClient(),
              GenerationBudget(tmp_path/'budget', limit=10, core_reserved=0),
              evidence_units={'u': UNIT}, token_counter=len)
    corpus.mkdir()
    docs = corpus/'documents.jsonl'
    docs.write_text(json.dumps({'doc_id': 'd', 'text': 'source', 'evidence_units': [UNIT]})+'\n')
    (corpus/'SPLIT_MANIFEST.json').write_text(json.dumps({'files': {'documents.jsonl': {'sha256': sha(docs)}}}))
    analysis = build_analysis(campaign, CONFIG)
    analysis_path = tmp_path/'ANALYSIS.json'
    analysis_path.write_text(json.dumps(analysis))
    final = tmp_path/'FINAL_REPORT.md'
    final.write_text(markdown_table(analysis), encoding='utf-8')
    result = audit_report(final, campaign, analysis_path, corpus, tmp_path/'AUDIT.json')
    assert result['status'] == 'PASS' and len(result['random_report_numbers']) == 10
    assert result['report_source']['sha256'] == sha(final)
    assert (tmp_path/'AUDIT.raw.json').exists()
    with pytest.raises(FileExistsError):
        audit_report(final, campaign, analysis_path, corpus, tmp_path/'AUDIT.json')
    final.write_text(final.read_text(encoding='utf-8').replace('|1.0000|', '|0.5000|', 1), encoding='utf-8')
    changed = audit_report(final, campaign, analysis_path, corpus, tmp_path/'AUDIT_changed.json')
    assert changed['status'] == 'FAILED_VALIDATION'
    assert not all(row['match'] for row in changed['all_report_table_checks'])
