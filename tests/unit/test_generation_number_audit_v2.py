"""Independent auditing is exercised only on explicit offline contract fixtures."""
import json
from pathlib import Path
import pytest

from scripts.audit_generation_numbers_v2 import audit, overlap, sha
from ragscope.experiments.generation_campaign import freeze_campaign, run_model, GenerationBudget
from ragscope.experiments.analysis_report import build_analysis
from test_generation_campaign_v2 import sample, IDENTITY, UNIT, OfflineContractClient
from test_analysis_report_v2 import CONFIG
from ragscope.indexing.identity import artifact


@pytest.mark.parametrize('external', [False, True])
def test_reverse_audit_detects_changed_report_without_trusting_saved_scores(tmp_path, external):
    campaign, corpus = tmp_path/'campaign', tmp_path/'corpus'
    inputs = {method: sample('same-question') for method in ('R0', 'R2')}
    groups = [{'experiment_id': 'E2', 'method': method, 'model_ids': ['fixture'], 'input_ids': [method]} for method in inputs]
    freeze_campaign(campaign, {'fixture': IDENTITY}, groups, inputs)
    run_model(campaign, 'fixture', OfflineContractClient(), GenerationBudget(tmp_path/'budget', limit=10, core_reserved=0),
              evidence_units={'u': UNIT}, token_counter=len)
    if external:
        main = tmp_path/'main'
        freeze_campaign(main, {'fixture': IDENTITY}, groups, inputs,
            provenance={'external_campaigns': [{'manifest': artifact(campaign/'EXPERIMENT_MANIFEST.json')}]})
        campaign = main
    corpus.mkdir()
    docs = corpus/'documents.jsonl'
    docs.write_text(json.dumps({'doc_id': 'd', 'text': 'source', 'evidence_units': [UNIT]})+'\n')
    (corpus/'SPLIT_MANIFEST.json').write_text(json.dumps({'files': {'documents.jsonl': {'sha256': sha(docs)}}}))
    report = build_analysis(campaign, CONFIG)
    analysis_path = tmp_path/'ANALYSIS.json'; analysis_path.write_text(json.dumps(report))
    result = audit(campaign, analysis_path, corpus, tmp_path/'AUDIT.json')
    assert result['status'] == 'PASS' and len(result['random_report_numbers']) == 10
    report['groups'][0]['estimates']['token_f1']['mean'] = .75
    analysis_path.write_text(json.dumps(report))
    assert audit(campaign, analysis_path, corpus, tmp_path/'AUDIT_changed.json')['status'] == 'FAILED_VALIDATION'


def test_independent_overlap_keeps_failure_denominator_and_all_references():
    query = {'answerability': 'answerable', 'answer_refs': ['wrong', {'free_form_answer': 'the right answer'}]}
    assert overlap({'answer': 'right answer'}, query) == (1., 1.)
    assert overlap({'answer': 'right answer', 'error': {'kind': 'timeout'}}, query) == (0., 0.)
    query['answerability'] = 'unanswerable'
    assert overlap({'answer': 'right answer'}, query) == (None, None)
