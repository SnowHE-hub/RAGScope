"""Full matched workflow utility vectors and preregistered mapping versus fixed/hindsight."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics
import sys
PROJECT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(PROJECT))
from ragscope.experiments.ledger import read_records
from ragscope.generation.pipeline import RAGPipeline
from ragscope.indexing.identity import artifact,atomic_json,canonical_json
from ragscope.v3.audit import overlap,reference_texts
from ragscope.v3.retrieval_diagnostics import coverage
from ragscope.v3.splits import read_split


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign',required=True);a=p.parse_args();run=Path(a.campaign).resolve()
    if run.parent!=Path('D:/RAGScope_Runtime/runs').resolve():raise ValueError('Wrong campaign')
    completion=json.loads((run/'S4_WORKFLOW_GENERATION_COMPLETION.json').read_text('utf8'))
    if completion['status']!='PASS_180_ACTUAL_WORKFLOW_GENERATIONS' or artifact(completion['records']['path'])!=completion['records']:
        raise ValueError('Workflow generation is not terminal complete')
    rows=read_records(Path(completion['records']['path']));manifest=json.loads((run/'S4_WORKFLOW_CONTEXT_MANIFEST.json').read_text('utf8'))
    inputs={e['input_id']:e for e in map(json.loads,Path(manifest['inputs']['path']).read_text('utf8').splitlines())}
    wanted={r['question_id'] for r in rows};labels={}
    for split in ('D_search','D_confirm'):
        labels.update({q['query_id']:q for q in read_split(run,split,purpose='development') if q['query_id'] in wanted})
    needed={u for q in labels.values() for group in q['evidence_reference_sets'] for u in group};units={}
    with Path('D:/RAGScope_Runtime/corpora/20261014_v2/all_documents.jsonl').open(encoding='utf8') as f:
        for line in f:
            for unit in json.loads(line)['evidence_units']:
                if unit['unit_id'] in needed:units[unit['unit_id']]=unit
    vectors=defaultdict(dict);task_labels={}
    for row in rows:
        q=labels[row['question_id']];entry=inputs['S4:'+row['workflow']+':'+row['question_id']]
        if row['actual_prompt_tokens'] is not None and row['actual_prompt_tokens']!=entry['actual_generator_prompt_tokens_preflight']:
            raise ValueError('Frozen/actual token mismatch')
        parsed=row['parsed_response'];failed=bool(row['error_type'])
        if not failed:
            check=RAGPipeline.parse_output(row['raw_response']['content'])
            if {k:check[k] for k in ('answer','citations','abstain')}!=parsed:raise ValueError('Stored/raw parser mismatch')
        answerable=q['answerability']=='answerable'
        word=None if not answerable else overlap(parsed['answer'],reference_texts(q))['legacy_token_f1'] if parsed and not failed and not parsed['abstain'] else 0.0
        correct_abstain=None if answerable else float(not failed and bool(parsed and parsed['abstain']))
        evidence=coverage(entry['context_bundle']['pieces'],q['evidence_reference_sets'],units) if q['evidence_reference_sets'] else None
        vectors[q['query_id']][row['workflow']]={'word_f1':word,'correct_abstention':correct_abstain,'failure':failed,
            'abstain':parsed['abstain'] if parsed else None,'evidence_metrics':evidence,'generation_wall_ms':row['wall_time_ms'],
            'source_record_hash':row['record_hash'],'claim_f1_proxy':None,'support_proxy':None,'citation_completeness_proxy':None,
            'comparison_keypoint_coverage':None,'summary_keypoint_coverage':None,'human_review_status':'NOT_REVIEWED'}
        task_labels[q['query_id']]=entry['task']
    for qid,vector in vectors.items():
        if set(vector)!={'W_local','W_compare','W_summary'}:raise ValueError('Missing workflow output')
        vector['W_clarify']={**vector['W_local'],'delegation':'specified scope -> local workflow',
            'additional_generation_requests':0,'not_an_independent_generated_answer':True}
    def quality(item):return item['word_f1'] if item['word_f1'] is not None else item['correct_abstention']
    summaries={}
    for name in ('W_local','W_compare','W_summary','W_clarify'):
        items=[v[name] for v in vectors.values()];words=[i['word_f1'] for i in items if i['word_f1'] is not None]
        summaries[name]={'n':len(items),'answerable_n':len(words),'word_f1':statistics.mean(words),
            'failure_n':sum(i['failure'] for i in items),'proxy_selection_quality':statistics.mean(quality(i) for i in items),
            'generation_p50_ms':statistics.median(i['generation_wall_ms'] for i in items),
            'latency_kind':'FROZEN_WORKFLOW_GENERATION_NOT_ONLINE_E2E'}
    fixed=max(('W_local','W_compare','W_summary'),key=lambda n:summaries[n]['proxy_selection_quality'])
    mapping={'fact':'W_local','explain':'W_local','compare':'W_compare','summarize':'W_summary','other':'W_local'}
    decisions=[]
    for qid,vector in vectors.items():
        selected=mapping[task_labels[qid]];oracle=max(('W_local','W_compare','W_summary'),key=lambda n:quality(vector[n]))
        decisions.append({'query_id':qid,'paper_id':labels[qid]['doc_id'],'task_proxy':task_labels[qid],
            'prelabeled_mapping':selected,'best_fixed':fixed,'hindsight_oracle':oracle,
            'mapping_proxy_quality':quality(vector[selected]),'fixed_proxy_quality':quality(vector[fixed]),
            'oracle_proxy_quality':quality(vector[oracle]),'full_utility_vector':vector})
    delta=statistics.mean(d['mapping_proxy_quality']-d['fixed_proxy_quality'] for d in decisions)
    result={'status':'PASS_240_UTILITY_ENTRIES_FROM_180_REAL_GENERATIONS','questions_n':60,'groups':summaries,
        'prelabeled_mapping':mapping,'best_fixed_by_short_answer_proxy':fixed,'mapping_minus_best_fixed_proxy':delta,
        'hindsight_oracle_mean':statistics.mean(d['oracle_proxy_quality'] for d in decisions),
        'router_training_gate':'NOT_RUN_GATE_PRIMARY_TASK_UTILITY_UNVERIFIED',
        'gate_reason':'Short QA overlap and generation-only cost do not establish comparison/summary keypoint correctness or online benefit.',
        'task_class_counts':manifest['class_limitations'],'human_review_status':'NOT_REVIEWED',
        'source_generation':artifact(Path(completion['records']['path'])),'source_contexts':artifact(run/'S4_WORKFLOW_CONTEXT_MANIFEST.json'),
        'limitations':['No summary-labeled question in the selected60; executing W_summary on short QA does not validate whole-paper summary quality.',
            'Hindsight oracle uses outcomes and cannot be deployed or treated as classification accuracy.',
            'W_clarify uses no extra generation for already specified documents; ambiguous-scope behavior is a separate test.']}
    output=run/'S4_WORKFLOW_UTILITY_VECTORS.jsonl'
    if output.exists():
        if [json.loads(l) for l in output.read_text('utf8').splitlines()]!=decisions:raise ValueError('Preserve differing utility vectors')
    else:output.write_text(''.join(canonical_json(d)+'\n' for d in decisions),encoding='utf8')
    result['utility_vectors']=artifact(output);atomic_json(run/'S4_WORKFLOW_UTILITY_ANALYSIS.json',result)
    print(canonical_json(result));return 0


if __name__=='__main__':raise SystemExit(main())
