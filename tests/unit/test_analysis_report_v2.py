"""Offline fixtures test analysis arithmetic and integrity, never model quality."""
import copy
import json
from pathlib import Path

import pytest

from ragscope.experiments.analysis_report import build_analysis, flatten_scores, markdown_table
from ragscope.experiments.generation_campaign import GenerationBudget, run_model, freeze_campaign, load_campaign
from ragscope.experiments.ledger import read_records
from ragscope.indexing.identity import signature, artifact
from test_generation_campaign_v2 import campaign, OfflineContractClient, UNIT


CONFIG = json.loads((Path(__file__).resolve().parents[2]/'configs/v2/analysis.json').read_text())


def execute(tmp_path, n=2, outputs=None, max_tasks=None):
    campaign(tmp_path, n=n)
    client = OfflineContractClient(outputs)
    run_model(tmp_path, 'fixture', client, GenerationBudget(tmp_path/'budget', limit=30, core_reserved=0),
              evidence_units={'u': UNIT}, token_counter=len, max_tasks=max_tasks)


def test_missing_denominator_not_promoted_to_final_estimate(tmp_path):
    execute(tmp_path, max_tasks=1)
    report = build_analysis(tmp_path, CONFIG)
    assert report['n_expected_runnable'] == 2 and report['n_observed'] == 1
    group = report['groups'][0]
    assert group['status'] == 'INCOMPLETE' and group['estimates'] is None
    assert group['n_missing'] == 1
    assert '1/2' in markdown_table(report)


def test_failure_zero_and_single_paper_cluster_ci(tmp_path):
    good = '{"answer":"source","citations":["c"],"abstain":false}'
    execute(tmp_path, outputs=[good]*3+[TimeoutError('fixture failure'), good])
    report = build_analysis(tmp_path, CONFIG)
    group = report['groups'][0]
    assert report['status'] == 'COMPLETE'
    assert group['observed_summary']['n_failed'] == 1
    assert group['estimates']['token_f1']['mean'] == .5
    assert group['estimates']['token_f1']['n_clusters'] == 1
    assert group['estimates']['token_f1']['low'] == .5
    assert group['estimates']['token_f1']['n_resamples'] == 2000
    assert group['performance']['first_token_ms'] is None


@pytest.mark.parametrize('change', ['mock', 'context', 'ticket', 'task', 'hash'])
def test_integrity_rejects_plausible_but_invalid_rows(tmp_path, change):
    execute(tmp_path)
    path = tmp_path/'records/E2_fixture.jsonl'
    rows = read_records(path)
    if change == 'mock': rows[0]['real_model'] = False
    if change == 'context': rows[0]['question'] = 'substituted query'
    if change == 'ticket': rows[1]['call_ticket'] = rows[0]['call_ticket']
    if change == 'task': rows[0]['query_id'] = 'wrong'
    for row in rows:
        row.pop('record_hash')
        row['record_hash'] = signature(row)
    if change == 'hash': rows[0]['scores']['token_f1'] = .875
    path.write_text(''.join(json.dumps(row)+'\n' for row in rows), encoding='utf-8')
    with pytest.raises(ValueError): build_analysis(tmp_path, CONFIG)


def test_article_cluster_and_impossible_failure_score():
    row = {'doc_id': 'paragraph-a', 'query_metadata': {'article_cluster_id': 'article'},
           'scores': {'token_f1': 0, 'exact_match': 0, 'rouge_l': 0}, 'error': {'kind': 'timeout'}}
    assert flatten_scores(row)['cluster_id'] == 'article'
    changed = copy.deepcopy(row); changed['scores']['token_f1'] = 1
    with pytest.raises(ValueError, match='zero'): flatten_scores(changed)


@pytest.mark.parametrize('max_tasks', [None, 1])
def test_external_rows_stay_at_original_source_and_missing_stays_missing(tmp_path, max_tasks):
    shard, main = tmp_path/'shard', tmp_path/'main'
    execute(shard, max_tasks=max_tasks)
    manifest, frozen = load_campaign(shard)
    original = (shard/'records/E2_fixture.jsonl').read_bytes()
    inputs = {key: {k: v for k, v in row.items() if k not in {'input_id', 'input_hash'}}
              for key, row in frozen.items()}
    freeze_campaign(main, manifest['models'], manifest['groups'], inputs,
        provenance={'external_campaigns': [{'manifest': artifact(shard/'EXPERIMENT_MANIFEST.json')}]})
    report = build_analysis(main, CONFIG)
    assert report['n_observed'] == (max_tasks or 2)
    assert report['groups'][0]['n_missing'] == (1 if max_tasks else 0)
    assert (shard/'records/E2_fixture.jsonl').read_bytes() == original
    assert not (main/'records/E2_fixture.jsonl').exists()
    assert any(item['path'] == str(shard/'records/E2_fixture.jsonl') for item in report['sources'])
