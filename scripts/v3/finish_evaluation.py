"""Score every final row, including failures, without another answer-generation call."""
import argparse
import json
from pathlib import Path
import sys
from collections import defaultdict
PROJECT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(PROJECT))
from ragscope.experiments.runtime import configure
from ragscope.indexing.identity import artifact,atomic_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',required=True);parser.add_argument('--run',required=True)
    parser.add_argument('--resident8k',action='store_true')
    args=parser.parse_args();configure(cpu_only=True)
    source,run=Path(args.source),Path(args.run)
    suffix='_RESIDENT8K' if args.resident8k else ''
    import numpy as np
    from ragscope.experiments.ledger import read_records,RecordStore
    from ragscope.v3.splits import read_split
    from ragscope.v3.local_runtime import ManagedOllama
    from ragscope.v3.finish_runtime import ReusableEncoder
    from ragscope.v3.practical_quality import fact_checks
    from ragscope.evaluation.generation_metrics import exact_match,token_f1,rouge_l
    from ragscope.evaluation.evidence import evidence_metrics,merge_intervals
    initial=read_records(run/'FINAL_GENERATION.jsonl')
    deployed=read_records(run/('DEPLOYMENT_FINAL_ANSWERS'+suffix+'.jsonl'))
    records=[r for r in initial if r['condition']=='B0']+deployed
    if len(initial)!=440 or len(deployed)!=220 or len(records)!=440 or not (run/'FINAL_CONFIGURATION.json').exists():
        raise ValueError('Freeze configuration and finish all440 answer records before final gold evaluation')
    if len({(r['query_id'],r['condition']) for r in records})!=440:raise ValueError('Duplicate or missing final question/condition')
    manifest=json.loads((source/'SPLITS_V3.json').read_text('utf8'))
    atomic_json(run/'SPLITS_V3.json',manifest)
    atomic_json(run/'CANDIDATE_LOCK.json',{'splits_manifest_hash':manifest['manifest_hash'],
        'configuration':artifact(run/'FINAL_CONFIGURATION.json'),'deployment_configuration':artifact(run/('DEPLOYMENT_CONFIGURATION'+suffix+'.json')),
        'purpose':'USER_OVERRIDE_FINAL_VALIDATION'})
    atomic_json(run/'FINAL_EVALUATION_AUTHORIZATION.json',{'candidate_lock_sha256':artifact(run/'CANDIDATE_LOCK.json')['sha256'],
        'evaluation_round':1,'authorization':'Explicit user instruction removes original10.1 admission block'})
    gold={q['query_id']:q for q in read_split(run,'H_QA',purpose='final_evaluation')}
    repair=json.loads((source/'H_TASKS_SOURCE_REPAIR.json').read_text('utf8'))
    gold_ref=next(ref for ref in repair['outputs'] if Path(ref['path']).name=='H_tasks.gold_v2.jsonl')
    if artifact(gold_ref['path'])!=gold_ref:raise ValueError('Task references changed')
    tasks={q['query_id']:q for q in (json.loads(line) for line in Path(gold_ref['path']).read_text('utf8').splitlines())}
    needed={pid for r in records for pid in r['scope']['doc_ids']};units={}
    with (run.parents[1]/'corpora/20261014_v2/all_documents.jsonl').open(encoding='utf8') as handle:
        for line in handle:
            doc=json.loads(line)
            if doc['doc_id'] in needed:units.update({u['unit_id']:u for u in doc['evidence_units']})
    inputs=[]
    for row in records:
        g=gold.get(row['query_id'],tasks.get(row['query_id']))
        refs=g.get('answer_refs',[g.get('reference_answer','')])
        p=row['parsed'] or {};answer=p.get('answer','')
        if answer:inputs.append(answer)
        inputs.extend(refs)
        inputs.extend(point['text'] for point in g.get('source_keypoints',[]))
        for piece in (row.get('context') or {}).get('pieces',[]):
            if piece['chunk_id'] in p.get('citations',[]):inputs.append(piece['text'])
    texts=list(dict.fromkeys(text for text in inputs if text.strip()))
    identity=json.loads((run/'FULL_DENSE_INDEX.json').read_text('utf8'))['embedding']
    with ManagedOllama(run,run.parents[1]/'tools/ollama-0.34.4/ollama.exe',num_ctx=4096,startup_ram_bytes=512*1024**2) as service:
        encoder=ReusableEncoder(run,source,service,identity)
        vectors=encoder.embed(texts)
    embeddings=dict(zip(texts,vectors))
    def similarity(a,b):
        return float(np.dot(embeddings[a],embeddings[b])) if a in embeddings and b in embeddings else 0.
    store=RecordStore(run/('FINAL_SCORES'+suffix+'.jsonl'),{'records':artifact(run/'FINAL_GENERATION.jsonl'),
        'deployment_records':artifact(run/('DEPLOYMENT_FINAL_ANSWERS'+suffix+'.jsonl')),
        'implementation':artifact(__file__),'rules':artifact(PROJECT/'ragscope/v3/practical_quality.py'),
        'semantic':'Qwen3 Embedding0.6B Q8_0 normalized answer/reference cosine; no query prefix, not factual correctness',
        'embedding':identity,'human_review':'NOT_REVIEWED'},max_records=440)
    grouped=defaultdict(list)
    for row in records:
        qid=row['query_id'];g=gold.get(qid,tasks.get(qid));refs=g.get('answer_refs',[g.get('reference_answer','')])
        p=row['parsed'] or {};answer=p.get('answer','');context=row.get('context') or {'pieces':[],'unit_spans':[]}
        valid=row['success'] and not p.get('abstain',False);answerable=g.get('answerability','answerable')!='unanswerable'
        values={metric:(function(answer,refs) if valid else 0.) if answerable else None for metric,function in [('exact_match',exact_match),('token_f1',token_f1),('rouge_l',rouge_l)]}
        values['semantic_cosine']=max((similarity(answer,ref) for ref in refs),default=0.) if valid and answerable else (0. if answerable else None)
        visible={piece['chunk_id'] for piece in context['pieces']};citations=p.get('citations',[])
        values.update(citation_visibility=(sum(c in visible for c in citations)/len(citations) if citations else None),
            answer_has_citation=bool(citations),failure=not row['success'],abstain=p.get('abstain',False),
            wrong_abstain=bool(p.get('abstain')) if answerable else None,
            correct_abstain=bool(p.get('abstain')) if not answerable and row['success'] else (False if not answerable else None),
            elapsed_s=row['elapsed_s'],output_tokens=row['backend'].get('eval_count'))
        evidence=evidence_metrics(g.get('evidence_reference_sets',[]),context.get('unit_spans',[]),g.get('answerability','answerable'),units)
        values['evidence_full_coverage']=evidence['full_reference_coverage'];values['evidence_continuous_coverage']=evidence['continuous_coverage']
        points=g.get('source_keypoints',[])
        values['keypoint_semantic_cosines']=[similarity(answer,point['text']) for point in points]
        intervals={pid:merge_intervals([(piece['canonical_start'],piece['canonical_end']) for piece in context['pieces'] if piece['doc_id']==pid]) for pid in {piece['doc_id'] for piece in context['pieces']}}
        values['keypoint_context_coverage']=[float(sum(max(0,min(point['canonical_end'],end)-max(point['canonical_start'],begin)) for begin,end in intervals.get(point['doc_id'],[]))>=point['canonical_end']-point['canonical_start']) for point in points]
        source_coverage=row.get('source_character_coverage',{})
        values['source_character_coverage']=float(np.mean(list(source_coverage.values()))) if source_coverage else None
        cited_texts=[piece['text'] for piece in context['pieces'] if piece['chunk_id'] in citations]
        values['citation_semantic_cosine']=max((similarity(answer,text) for text in cited_texts),default=0.) if valid else None
        rules=fact_checks(row['question'],answer,refs,context['pieces'],citations)
        rid=row['record_id']
        score={'source_record_id':rid,'query_id':qid,'split':row['split'],'condition':row['condition'],'model':row['model'],
            'metrics':values,'literal_fact_checks':rules,'factual_correctness':'UNKNOWN','human_review':'NOT_REVIEWED'}
        if rid not in store.ids:store.add(rid,score)
        grouped[row['split']+':'+row['condition']].append(score)
    summary={}
    for key,rows in grouped.items():
        result={'n':len(rows),'failures':sum(r['metrics']['failure'] for r in rows),'factual_unknown':len(rows)}
        verdicts=defaultdict(int)
        for r in rows:verdicts[r['literal_fact_checks']['rule_verdict']]+=1
        result['literal_rule_verdicts']=dict(verdicts)
        result['number_flags']=sum(bool(r['literal_fact_checks']['numbers_not_in_context']) for r in rows)
        result['yes_no_reference_conflicts']=sum(r['literal_fact_checks']['yes_no_reference_conflict'] is True for r in rows)
        for metric in ('exact_match','token_f1','rouge_l','semantic_cosine','citation_visibility','evidence_full_coverage','evidence_continuous_coverage','wrong_abstain','correct_abstain','output_tokens','source_character_coverage'):
            applicable=[r['metrics'][metric] for r in rows if r['metrics'][metric] is not None]
            result[metric]=float(np.mean(applicable)) if applicable else None;result[metric+'_n']=len(applicable)
        result['p50_seconds']=float(np.median([r['metrics']['elapsed_s'] for r in rows]))
        result['p95_seconds']=float(np.percentile([r['metrics']['elapsed_s'] for r in rows],95))
        points=[value for r in rows for value in r['metrics']['keypoint_context_coverage']]
        result['source_keypoint_context_coverage']=float(np.mean(points)) if points else None
        summary[key]=result
    output={'groups':summary,'records':artifact(run/'FINAL_GENERATION.jsonl'),'scores':artifact(store.path),
        'deployment_records':artifact(run/('DEPLOYMENT_FINAL_ANSWERS'+suffix+'.jsonl')),'final_input_selection':'Original220 B0 rows + deployed220 final rows. All prior variants retained as diagnostics, not silently overwritten.',
        'semantic_definition':'Normalized answer/reference cosine; not BERTScore, not factual accuracy',
        'baseline_qualification':'V2 policy on identical expanded corpus and GPU embedding. Original CPU deployment is not the timing baseline.',
        'generation_latency_interpretation':'FinalQA includes exact-request response reuse. Source-row generation timing is diagnostic; actual online records define deployment speed.',
        'H_tasks_reference_status':'20 AI-constructed tasks with source-verified quotes, not official or human-reviewed gold',
        'human_review':'NOT_REVIEWED','unknown_does_not_block':True}
    atomic_json(run/('FINAL_SUMMARY'+suffix+'.json'),output)
    print(json.dumps(output,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
