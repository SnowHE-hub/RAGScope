from ragscope.v3.semantic_inputs import blinded_context_pieces, validated_calibration_groups
from scripts.v3.rescore_old import calibration_groups, observed_output_tokens


def test_actual_backend_output_tokens_do_not_use_reference_token_counts():
    value = {'backend_metadata': {'actual_output_tokens': 512, 'eval_count': 4}, 'actual_output_tokens': 120}
    assert observed_output_tokens(value) == (512, 'backend.actual_output_tokens')
    assert observed_output_tokens({'backend_metadata': {}})[0] is None


def test_contrasts_are_fixed_before_judging_and_originals_remain_unchanged():
    old = calibration_groups()
    new = validated_calibration_groups(old)
    assert len(new) == 60
    assert 'sold' in old[0]['negative']
    assert 'did not purchase that car' in new[0]['negative']
    assert all(g['judge_result'] == 'NOT_RUN_YET' for g in new)
    assert all(g['human_review_status'] == 'NOT_REVIEWED' for g in new)


def test_blind_context_preserves_source_text_without_model_identity():
    bundle = {'pieces': [{'chunk_id': 'p:c', 'text': 'Evidence text.', 'canonical_start': 4,
                          'canonical_end': 18, 'model_id': 'hidden', 'score': 0.91}]}
    assert blinded_context_pieces(bundle) == [{'chunk_id': 'p:c', 'text': 'Evidence text.',
                                              'canonical_start': 4, 'canonical_end': 18}]
