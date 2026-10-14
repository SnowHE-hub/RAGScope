from ragscope.v3.protocol_diagnostics import recover_envelope


def test_envelope_recovery_preserves_answer_and_does_not_fill_missing_fields():
    raw='```json\n{"answer":"Alpha did not beat Beta.","citations":[],"abstain":false}\n```'
    result=recover_envelope(raw)
    assert result['status']=='RECOVERABLE_ENVELOPE_DIAGNOSTIC_ONLY'
    assert result['original_object']['answer']=='Alpha did not beat Beta.'
    assert raw[result['original_start']:result['original_end']].startswith('{')
    assert recover_envelope('{"answer":"Alpha","citations":[]}')['eligible_objects']==0


def test_multiple_valid_answers_are_not_cherry_picked():
    first='{"answer":"yes","citations":[],"abstain":false}'
    second='{"answer":"no","citations":[],"abstain":false}'
    assert recover_envelope(first+'\n'+second)['status']=='AMBIGUOUS_MULTIPLE_OBJECTS'
