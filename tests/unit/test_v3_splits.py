import json
import pytest

from ragscope.indexing.identity import artifact, atomic_json, canonical_json, signature
from ragscope.v3.splits import assert_inference_safe, public_question, read_split, select_paper_questions


def test_recursive_gold_fields_never_enter_public_input():
    with pytest.raises(ValueError, match='Gold'):
        assert_inference_safe({'context': [{'metadata': {'gold_doc_id_evaluation_only': 'p'}}]})
    q = {'query_id': 'q', 'doc_id': 'p', 'question': 'What?', 'answer_refs': ['private']}
    assert public_question(q) == {'query_id': 'q', 'question': 'What?', 'scope': {'kind': 'document', 'doc_ids': ['p']}}


def test_selection_is_deterministic_and_balanced_by_paper():
    queries = [{'query_id': f'{p}:{i}', 'doc_id': p} for p in ['p1', 'p2', 'p3'] for i in range(3)]
    left = select_paper_questions(queries, 6, 3, 42, 'holdout')
    assert left == select_paper_questions(list(reversed(queries)), 6, 3, 42, 'holdout')
    assert len({q['doc_id'] for q in left}) == 3
    with pytest.raises(ValueError, match='Not enough'):
        select_paper_questions(queries, 6, 4, 42, 'holdout')


def make_split(tmp_path):
    public = tmp_path / 'public.jsonl'
    gold = tmp_path / 'gold.jsonl'
    public.write_text(canonical_json({'query_id': 'q', 'question': 'What?'}) + '\n', encoding='utf-8')
    gold.write_text(canonical_json({'query_id': 'q', 'answer_refs': ['secret']}) + '\n', encoding='utf-8')
    manifest = {'splits': {'H_QA': {'role': 'sealed_test', 'questions': artifact(public), 'gold': artifact(gold)}}}
    manifest['manifest_hash'] = signature(manifest)
    atomic_json(tmp_path / 'SPLITS_V3.json', manifest)
    return manifest


@pytest.mark.parametrize('purpose', ['training', 'development', 'diagnosis', 'final_evaluation'])
def test_holdout_gold_is_denied_before_frozen_final_round(tmp_path, purpose):
    make_split(tmp_path)
    with pytest.raises(PermissionError):
        read_split(tmp_path, 'H_QA', purpose=purpose)
    assert read_split(tmp_path, 'H_QA') == [{'query_id': 'q', 'question': 'What?'}]


def test_holdout_authorization_binds_the_lock_and_single_round(tmp_path):
    manifest = make_split(tmp_path)
    lock = tmp_path / 'CANDIDATE_LOCK.json'
    atomic_json(lock, {'splits_manifest_hash': manifest['manifest_hash']})
    auth = tmp_path / 'FINAL_EVALUATION_AUTHORIZATION.json'
    atomic_json(auth, {'candidate_lock_sha256': artifact(lock)['sha256'], 'evaluation_round': 2})
    with pytest.raises(PermissionError):
        read_split(tmp_path, 'H_QA', purpose='final_evaluation')
    atomic_json(auth, {'candidate_lock_sha256': artifact(lock)['sha256'], 'evaluation_round': 1})
    assert read_split(tmp_path, 'H_QA', purpose='final_evaluation')[0]['answer_refs'] == ['secret']
