from ragscope.v3.records import canonical_payload,FIELDS


def test_missing_measurement_cannot_become_zero_or_human_review_and_failure_is_preserved():
    record=canonical_payload({'success':False,'error_type':{'kind':'transport_error'},'human_review_status':'REVIEWED',
        'user_scope':{'kind':'document','doc_ids':['visible']},'wall_time_ms':123},campaign_id='c',task_id='t')
    assert set(FIELDS)<=set(record)
    assert record['actual_prompt_tokens'] is None
    assert record['output_tokens'] is None
    assert record['human_review_status']=='NOT_REVIEWED'
    assert record['success'] is False and record['error_type']['kind']=='transport_error'
    assert record['paper_ids']==['visible']
    assert record['wall_time']['seconds']==.123
