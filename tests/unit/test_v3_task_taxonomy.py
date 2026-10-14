from ragscope.v3.task_taxonomy import label_question


def test_compound_task_is_not_forced_to_one_label_and_sufficiency_is_unknown():
    result=label_question('Summarize the difference between Alpha and Beta.',{'kind':'document','doc_ids':['p']})
    assert result['task_labels']==['compare','summarize']
    assert result['document_scope']=='specified'
    assert result['context_sufficient'] is None and result['confidence_probability'] is None


def test_generic_global_question_cannot_silently_acquire_a_paper():
    result=label_question('What dataset did they use?',{'kind':'global'})
    assert result['document_scope']=='ambiguous'
    assert result['task']=='fact'
    assert result['label_source']=='RULE_PROXY_NOT_HUMAN'
