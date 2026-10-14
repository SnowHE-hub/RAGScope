"""Visible-query workflow actions and balanced source evidence selection."""
from dataclasses import asdict
import re
from ragscope.generation.context import pack_context,reference_token_count,render_context


def comparison_subqueries(question):
    """Extract explicit operands only; an unclear pair retains the original query."""
    match=re.search(r'\bbetween\s+(.+?)\s+and\s+(.+?)(?:\?|$)',question,re.I)
    if match is None:match=re.search(r'(.+?)\s+(?:versus|vs\.?)\s+(.+?)(?:\?|$)',question,re.I)
    if match:
        operands=[part.strip() for part in match.groups()]
        if all(0<len(part.split())<=18 for part in operands):
            return [f'{question}\nEvidence specifically describing: {part}' for part in operands]
    return [question]


def section_labels(chunk):
    match=re.search(r'^Section:\s*(.+)$',chunk.get('retrieval_text',''),re.M)
    return [s.strip() for s in match.group(1).split('|') if s.strip()] if match else ['UNKNOWN_SECTION']


def balanced_context(groups,*,scope,max_tokens=2048):
    """Equal initial budgets preserve whole source chunks and their coordinates."""
    if not groups:raise ValueError('No evidence groups')
    group_budget=max_tokens//len(groups);pieces=[];reports=[]
    for name,chunks in groups:
        bundle=asdict(pack_context(chunks,max_tokens=group_budget,scope=scope))
        pieces.extend(bundle['pieces'])
        reports.append({'group':name,'allocated_reference_tokens':group_budget,
            'selected_chunk_ids':[p['chunk_id'] for p in bundle['pieces']],
            'drops':bundle['truncations'],'empty':not bundle['pieces']})
    final=pack_context(pieces,max_tokens=max_tokens,scope=scope)
    if reference_token_count(render_context(final))>max_tokens:raise ValueError('Balanced context exceeds budget')
    return final,{'groups':reports,'policy':'equal initial whole-chunk budgets; canonical dedup in final pack',
        'coverage_guarantee':False,'human_review_status':'NOT_REVIEWED'}


def summary_context(ranking,*,scope,max_tokens=2048):
    """Pick at most three distinct section groups within the shared finite budget."""
    sections=[]
    for chunk in ranking:
        for label in section_labels(chunk):
            if label not in sections:sections.append(label)
    selected=sections[:3]
    groups=[(label,[c for c in ranking if label in section_labels(c)]) for label in selected]
    bundle,audit=balanced_context(groups,scope=scope,max_tokens=max_tokens)
    audit.update(available_sections=sections,selected_sections=selected,unselected_sections=sections[3:],
        limitation='Finite section sampling, not exhaustive whole-document coverage')
    return bundle,audit


def clarification_action(question,scope,visible_candidates=()):
    if scope and scope.get('doc_ids'):return {'action':'EXECUTE_WITH_SPECIFIED_SCOPE','doc_ids':scope['doc_ids']}
    return {'action':'CLARIFY_DOCUMENT','message':'请指定要分析的论文或文档。',
        'candidate_documents':list(visible_candidates),'question':question,
        'generation_calls':0,'hidden_gold_used':False}
