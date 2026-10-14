"""Canonical experiment records: unmeasured fields stay null, never fabricated zero."""
FIELDS=tuple('campaign_id task_id question_id paper_ids dataset_role exposure_status task_type_source user_scope config_hash model_identity prompt_hash actual_prompt_tokens context_intervals retriever_candidates rank_scores packing_drops raw_response parsed_response stop_reason output_tokens success error_type wall_time measured_resource metric_version judge_identity proxy_unknowns budget_ticket'.split())


def canonical_payload(payload,*,campaign_id,task_id):
    output={key:None for key in FIELDS}
    output.update(payload,campaign_id=campaign_id,task_id=task_id)
    output['human_review_status']='NOT_REVIEWED'
    scope=output.get('user_scope')
    if output['paper_ids'] is None and isinstance(scope,dict):output['paper_ids']=scope.get('doc_ids')
    if output['wall_time'] is None and output.get('wall_time_ms') is not None:
        output['wall_time']={'seconds':output['wall_time_ms']/1000,'basis':'actual nonstream wall clock; TTFT unavailable'}
    if output['proxy_unknowns'] is None:output['proxy_unknowns']={'unmeasured_fields':[k for k in FIELDS if output[k] is None]}
    return output
