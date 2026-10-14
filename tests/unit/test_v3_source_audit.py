"""Audit integrity and metric limitations, using functional fixtures only."""
import json
import sqlite3

import pytest

from ragscope.indexing.identity import canonical_json, signature
from ragscope.v3.audit import audit_lora, journal_rows, overlap, projection


def test_multiple_reference_uses_matching_reference_and_reports_lengths():
    scored = overlap('automobile', ['car', 'an automobile'])
    assert scored['legacy_token_f1'] == 1
    assert scored['best_reference_index'] == 1
    assert scored['reference_words'] == [1, 1]


def test_word_f1_does_not_prove_semantic_correctness():
    assert overlap('automobile', ['car'])['legacy_token_f1'] == 0
    assert overlap('The baseline outperforms the new method.',
                   ['The new method outperforms the baseline.'])['legacy_token_f1'] == 1


def test_empty_official_score_is_distinguished_from_legacy():
    result = overlap('', ['the'])
    assert result['legacy_token_f1'] == 1
    assert result['qasper_compatible_answer_f1'] == 0


def test_source_hash_and_partial_tail_fail_without_repairing_source(tmp_path):
    path = tmp_path / 'old.jsonl'
    value = {'record_id': 'q1', 'answer': 'original'}
    value['record_hash'] = signature(value)
    value['answer'] = 'altered'
    path.write_text(canonical_json(value) + '\n', encoding='utf-8')
    before = path.read_bytes()
    with pytest.raises(ValueError, match='Corrupt'):
        list(journal_rows(path))
    assert path.read_bytes() == before
    path.write_bytes(b'{"record_id":"partial"}')
    with pytest.raises(ValueError, match='Incomplete'):
        list(journal_rows(path))
    assert path.read_bytes() == b'{"record_id":"partial"}'


def fixture_row(qid='q1', *, abstain=False, failed=False, answerable=True):
    row = {'record_id': qid, 'query_id': qid, 'question': 'What colour?', 'doc_id': 'p1',
        'answer': 'blue', 'abstain': abstain, 'error': {'kind': 'parse_error'} if failed else None,
        'context_bundle': {'context_hash': 'ctx', 'budget': {'reference_tokens': 20}},
        'prompt': [{'role': 'user', 'content': 'What colour? Material says blue.'}],
        'config_hash': 'cfg', 'record_hash': 'hash', 'real_model': True,
        'backend_metadata': {'eval_count': 5}, 'timings': {'generation_ms': 5},
        'scores': {'token_f1': None if not answerable else 0 if abstain or failed else 1,
                   'exact_match': None if not answerable else 0 if abstain or failed else 1}}
    query = {'query_id': qid, 'question': 'What colour?', 'doc_id': 'p1',
             'answer_refs': ['blue'], 'answerability': 'answerable' if answerable else 'unanswerable'}
    return row, query


def test_failures_remain_zero_unanswerable_stays_na_and_sufficiency_unknown():
    row, query = fixture_row(failed=True)
    result = projection(row, query, {})
    assert result['legacy_token_f1'] == 0 and result['failed']
    assert result['visible_context_sufficient'] == 'UNKNOWN'
    row, query = fixture_row(answerable=False)
    assert projection(row, query, {})['legacy_token_f1'] is None


def test_raw_reported_score_disagreement_is_refused():
    row, query = fixture_row()
    row['scores']['token_f1'] = 0.9
    with pytest.raises(ValueError, match='Independent F1'):
        projection(row, query, {})


def test_frozen_dataclass_document_scope_is_legal_identity():
    row, query = fixture_row()
    del query['doc_id']
    query['scope'] = {'kind': 'document', 'doc_ids': ['p1']}
    assert projection(row, query, {})['doc_id'] == 'p1'


def test_lora_decomposition_preserves_failure_denominator(tmp_path):
    run = tmp_path / 'new'
    run.mkdir()
    root = tmp_path / 'runtime'
    pair = root / 'runs/20261014_v2_lora/train_2b_1024_seed42/paired_test100'
    pair.mkdir(parents=True)
    db = sqlite3.connect(run / 'S0_EVIDENCE.sqlite')
    db.execute('CREATE TABLE queries(query_id TEXT PRIMARY KEY,payload TEXT)')
    for qid in ('q1', 'q2'):
        _, query = fixture_row(qid)
        db.execute('INSERT INTO queries VALUES(?,?)', (qid, canonical_json(query)))
    db.commit()
    db.close()
    for condition in ('baseline', 'adapted'):
        config = {'condition': condition}
        (pair / (condition + '.config.json')).write_text(json.dumps(config), encoding='utf-8')
        rows = []
        for qid in ('q1', 'q2'):
            row, _ = fixture_row(qid, abstain=condition == 'adapted' and qid == 'q1',
                                 failed=condition == 'adapted' and qid == 'q2')
            row['config_hash'] = signature(config)
            del row['record_hash']
            row['record_hash'] = signature(row)
            rows.append(canonical_json(row))
        (pair / (condition + '.jsonl')).write_text('\n'.join(rows) + '\n', encoding='utf-8')
    result = audit_lora(run, root, {})
    assert result['answerable_n'] == 2
    assert result['delta_f1'] == -1
    assert result['group_contributions'] == {'NEW_ABSTENTION': -0.5, 'FORMAT_OR_REQUEST_FAILURE': -0.5}
    assert result['contribution_sum_matches_delta']
