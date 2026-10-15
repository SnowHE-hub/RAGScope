"""Task-directed whole-paragraph selection, without answer or reference inputs."""
import re
from ragscope.v3.workflows import balanced_context


def named_comparison_queries(question):
    match=re.match(r'^Compare (.+?) and (.+?) as described in this paper',question)
    if not match:raise ValueError('Comparison requires two explicit visible names')
    return [question+'\nFocus on the named approach: '+name.strip() for name in match.groups()]


def source_paragraphs(document,ranking):
    paragraphs=[]
    sections={s.get('section_id'):s['title'] for s in document['sections']}
    for unit in document['evidence_units']:
        if unit['kind'] in {'title','section_title'} or len(unit['text'].split())<8:continue
        lo,hi=unit['canonical_start'],unit['canonical_end']
        if document['text'][lo:hi]!=unit['text']:raise ValueError('Source paragraph coordinates changed')
        section=sections.get(unit['section_id'],'Abstract' if ':abstract:' in unit['unit_id'] else 'UNKNOWN_SECTION')
        score=max((c['score'] for c in ranking if c['canonical_start']<hi and c['canonical_end']>lo),default=0)
        paragraphs.append({'chunk_id':'source-paragraph:'+unit['unit_id'],'doc_id':document['doc_id'],
            'text':unit['text'],'canonical_start':lo,'canonical_end':hi,'section_title':section,'score':score,
            'metadata':{'title':document['title']},'unit_spans':[{'unit_id':unit['unit_id'],'doc_id':document['doc_id'],
                'start':lo,'end':hi,'chunk_start':0,'chunk_end':hi-lo}]})
    return sorted(paragraphs,key=lambda p:(-p['score'],p['canonical_start']))


def summary_paragraph_context(document,ranking,*,scope,max_tokens=2048):
    paragraphs=source_paragraphs(document,ranking);sections=list(dict.fromkeys(p['section_title'] for p in paragraphs))
    chosen=[]
    for pattern in (r'abstract',r'method|approach|model|architecture|framework|implementation',
        r'experiment|evaluation|result|finding|analysis',r'conclusion|limitation|discussion'):
        matches=[s for s in sections if s not in chosen and re.search(pattern,s,re.I)]
        if matches:chosen.append(matches[0])
    chosen.extend(s for s in sections if s not in chosen)
    chosen=chosen[:4]
    bundle,audit=balanced_context([(s,[p for p in paragraphs if p['section_title']==s]) for s in chosen],scope=scope,max_tokens=max_tokens)
    audit.update(selected_sections=chosen,available_sections=sections,source_unit_policy='complete original paragraphs; no synthesized evidence',
        omitted_sections=[s for s in sections if s not in chosen],gold_read=False)
    return bundle,audit
