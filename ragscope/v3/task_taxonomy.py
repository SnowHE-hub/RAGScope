"""Visible-input task/scope labels; rule proxies never assert context sufficiency."""
import re


def label_question(question,scope):
    text=question.casefold();tasks=[]
    if re.search(r'\b(compare|contrast|difference|versus|trade.?off)\b|比较|区别',text):tasks.append('compare')
    if re.search(r'\b(summarize|summary|overview|main contributions|main findings)\b|总结|概括',text):tasks.append('summarize')
    if not tasks:
        if re.search(r'^(why|how)\b',text) and not re.search(r'^how (many|much)\b',text):tasks=['explain']
        elif re.search(r'^(what|which|when|where|who|is|are|was|were|do|does|did|can|how many|how much)\b',text):tasks=['fact']
        else:tasks=['other']
    specified=isinstance(scope,dict) and scope.get('kind')=='document' and bool(scope.get('doc_ids'))
    anchored=bool(re.search(r'paper title:|\b(?:paper|document)\s+["\u201c]|题名|论文[《\u201c]',text))
    document_scope='specified' if specified else 'identifiable' if anchored else 'ambiguous'
    evidence_scope='multi_document' if re.search(r'\b(papers|documents)\b|两篇|多篇',text) else 'whole_document' if 'summarize' in tasks else 'multi_section' if 'compare' in tasks else 'local'
    if re.search(r'\babstract\b|摘要',text):evidence_scope='local'
    return {'task':tasks[0],'task_labels':tasks,'evidence_scope':evidence_scope,'document_scope':document_scope,
        'label_source':'RULE_PROXY_NOT_HUMAN','confidence_probability':None,'context_sufficient':None,
        'evidence_scope_status':'HEURISTIC_NOT_PROVEN_BY_QUESTION','human_review_status':'NOT_REVIEWED'}
