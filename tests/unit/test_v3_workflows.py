from dataclasses import asdict
from ragscope.v3.workflows import balanced_context,clarification_action,comparison_subqueries,summary_context


def chunk(name,start,text,section):
    return {'chunk_id':name,'doc_id':'paper','text':text,'canonical_start':start,'canonical_end':start+len(text),
        'retrieval_text':f'Title: paper\nSection: {section}\n\n{text}','score':1}


def test_comparison_is_bounded_and_ambiguous_pair_is_not_invented():
    queries=comparison_subqueries('What differs between model A and model B?')
    assert len(queries)==2 and 'model A' in queries[0] and 'model B' in queries[1]
    assert comparison_subqueries('Why does the method work?')==['Why does the method work?']


def test_balanced_evidence_preserves_both_sides_and_source_coordinates():
    a,b=chunk('a',0,'A is faster.','Methods'),chunk('b',100,'B is more accurate.','Results')
    packed,audit=balanced_context([('A',[a]),('B',[b])],scope={'doc_ids':['paper']},max_tokens=200)
    pieces=asdict(packed)['pieces']
    assert {p['chunk_id'] for p in pieces}=={'a','b'}
    assert all(p['text']=={'a':a,'b':b}[p['chunk_id']]['text'] for p in pieces)
    assert all(not g['empty'] for g in audit['groups'])


def test_summary_and_clarify_expose_finite_coverage_and_ambiguity():
    rows=[chunk(str(i),100*i,f'Fact {i}.',s) for i,s in enumerate(('Introduction','Methods','Results','Limitations'))]
    packed,audit=summary_context(rows,scope={'doc_ids':['paper']},max_tokens=300)
    assert len(audit['selected_sections'])==3 and audit['unselected_sections']==['Limitations']
    assert len(asdict(packed)['pieces'])==3
    assert clarification_action('What did they find?',{'kind':'global'})['action']=='CLARIFY_DOCUMENT'
    assert clarification_action('What did they find?',{'doc_ids':['paper']})['action']=='EXECUTE_WITH_SPECIFIED_SCOPE'
