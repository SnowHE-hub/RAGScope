"""Final deployment intent and longer, balanced source context for complex tasks."""
from dataclasses import asdict
import re
from ragscope.generation.context import pack_context
from ragscope.evaluation.evidence import merge_intervals
from ragscope.v3.finish_runtime import task_kind,evidence_bundle,answer_messages
from ragscope.v3.generation_protocols import ProtocolClient


def deployment_kind(question,explicit=None):
    if explicit and explicit!='auto':return task_kind(question,explicit)
    # Asking for names of compared baselines is a fact request, not an essay.
    if re.search(r'^\s*(which|what)\b.*\b(models?|baselines?|datasets?|methods?|approaches?)\b',question,re.I) and not re.search(r'\b(difference|better|worse|best|advantages?|disadvantages?)\b',question,re.I):
        return 'fact'
    if re.search(r'^\s*(is|are|was|were|do|does|did|has|have|can)\b',question,re.I):return 'fact'
    return task_kind(question)


class DeploymentClient(ProtocolClient):
    def __init__(self,original,service,num_ctx=8192):
        super().__init__(original,service,protocol='G0')
        self.num_ctx=num_ctx;self.identity['num_ctx']=num_ctx


def deployment_context(engine,question,scope,vector,original,selection,explicit=None,method='hybrid'):
    kind=deployment_kind(question,explicit)
    ids=(scope or {}).get('doc_ids',[])
    complex_task=kind in {'summary','comparison'} or len(ids)>1
    if not complex_task:
        effective='B0' if ids else 'CONTROL'
        options={'candidate_depth':50} if ids else selection['global_selected']['options']
        options={**options,'method':method}
        context,result,_=evidence_bundle(engine,question,scope,vector,condition=effective,retrieval=options,explicit_task=kind)
        return context,result,kind,8192,512,effective
    options=selection['selected' if ids else 'global_selected']['options']
    options={**options,'method':method}
    result=engine.search(question,scope=scope,query_vector=vector,embedding_identity=engine.embedding_identity,balanced=True,**options)
    ranked=result['candidates']
    groups=[]
    if kind=='summary' and ids:
        for pid in ids:
            body=[engine.chunks[i] for i in engine.by_doc[pid]]
            first=[0]
            for pattern in (r'method|approach|framework',r'experiment|result|evaluation',r'conclu|limitation|future'):
                hits=[j for j,c in enumerate(body) if re.search(pattern,c['text'][:250],re.I)]
                first.extend(hits[-1:] if 'conclu' in pattern else hits[:1])
            first.extend([len(body)-1,*range(len(body))])
            groups.append([body[j] for j in dict.fromkeys(first)])
    elif ids:
        groups=[[c for c in ranked if c['doc_id']==pid] for pid in ids]
    else:groups=[ranked]
    candidates=[group[pos] for pos in range(max(map(len,groups),default=0)) for group in groups if pos<len(group)]
    system=answer_messages(question,{'pieces':[]},condition='FINAL',kind=kind)[0]
    def fits(rendered):
        messages=[system,{'role':'user','content':f'Question: {question}\n\nSource excerpts:\n{rendered}\n\nReturn the answer JSON.'}]
        return original.prompt_token_counter(messages)+768<=8192
    context=asdict(pack_context(candidates,max_tokens=6144,scope=scope,model_budget_checks=[fits]))
    coverage={}
    for pid in ids:
        spans=[(p['canonical_start'],p['canonical_end']) for p in context['pieces'] if p['doc_id']==pid]
        total=max(engine.chunks[i]['canonical_end'] for i in engine.by_doc[pid])
        coverage[pid]=sum(b-a for a,b in merge_intervals(spans))/total if total else 0.
    # Extra metadata lives outside ContextBundle so its provenance hash stays valid.
    result['source_character_coverage']=coverage
    return context,result,kind,8192,768,'FINAL'
