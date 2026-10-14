"""Independently audit matched Oracle controls and remaining factual probes."""
import argparse
from collections import Counter
import json
from pathlib import Path
import statistics
import sys
PROJECT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(PROJECT))
from ragscope.experiments.ledger import read_records
from ragscope.generation.pipeline import RAGPipeline
from ragscope.indexing.identity import artifact,atomic_json,canonical_json
from ragscope.v3.audit import overlap
from ragscope.v3.budget import Budget


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign',required=True);a=p.parse_args();run=Path(a.campaign).resolve()
    if run.parent!=Path('D:/RAGScope_Runtime/runs').resolve():raise ValueError('Wrong campaign')
    path=run/'S2_SUPPLEMENT_RESULTS_verified.jsonl';rows=read_records(path)
    receipt=json.loads((run/'S2_SUPPLEMENT_INPUTS.json').read_text('utf8'))
    inputs={x['input_id']:x for x in read_records(run/'S2_SUPPLEMENT_INPUTS.jsonl')}
    labels={x['query_id']:x for x in read_records(run/'splits/evaluator_only/S2_DIAGNOSTIC_LABELS.jsonl')}
    mainrows={(x['model_id'],x['input_id'].split(':',2)[2],x['context_kind']):x for x in read_records(run/'S2_PROTOCOL_RESULTS.jsonl') if x['protocol']=='G1'}
    pairs=[];mismatch=[];token_mismatch=[];prior=[]
    budget=Budget(run)
    with budget.lock:state=budget._state()
    for row in rows:
        e=inputs[row['input_id']];rid=row['record_id'];raw=row['raw_response'].get('content')
        if rid not in state['reserved'] or rid not in state['finalized']:raise ValueError('Model result lacks completed budget ticket')
        if row['actual_sent_payload']['options']['num_predict']!=512:raise ValueError('Output budget changed')
        if row['actual_prompt_tokens']+512>4096:raise ValueError('Complete prompt exceeded context')
        if row['raw_response'].get('backend_metadata',{}).get('token_count_match') is not True:token_mismatch.append(rid)
        failed=bool(row['error_type']);parsed=None
        if not failed:
            parsed=RAGPipeline.parse_output(raw)
            if {k:parsed[k] for k in ('answer','citations','abstain')}!=row['parsed_response']:raise ValueError('Stored parse differs from actual raw output')
        refs=labels[e['query_id']]['answer_refs'] if e['context_kind']=='oracle_padded' else [e['expected_answer']]
        score=overlap(parsed['answer'],refs)['legacy_token_f1'] if parsed and not parsed['abstain'] else 0.0
        if abs(score-row['legacy_token_f1'])>1e-12:mismatch.append(rid)
        if e['context_kind']=='prior':
            answer=parsed['answer'].casefold().strip() if parsed else ''
            expected=e['expected_answer'].casefold()
            prior.append({'model_id':row['model_id'],'question_id':e['query_id'],'answer':answer,'abstain':parsed['abstain'] if parsed else None,
                'literal_correctness_proxy':expected in answer or expected=='7' and 'seven' in answer,'failure':failed})
        else:
            paired={'model_id':row['model_id'],'query_id':e['query_id'],'oracle_padded':score,'padding_failure':failed}
            for kind in ('oracle_audited','retrieval_frozen'):
                old=mainrows[row['model_id'],e['query_id'],kind];parse=old['parsed_response']
                paired[kind]=overlap(parse['answer'],refs)['legacy_token_f1'] if parse and not old.get('error_type') and not parse['abstain'] else 0.0
            pairs.append(paired)
    expected={('qwen_'+size+'b',kind):n for size in ('2','4','9') for kind,n in [('oracle_padded',12),('prior',2)]}
    counts=Counter((x['model_id'],x['context_kind']) for x in rows)
    if counts!=expected:raise ValueError('Supplement incomplete or duplicate design cells')
    summaries={}
    for mid in ('qwen_2b','qwen_4b','qwen_9b'):
        group=[x for x in pairs if x['model_id']==mid]
        summaries[mid]={kind:statistics.mean(x[kind] for x in group) for kind in ('oracle_audited','oracle_padded','retrieval_frozen')}
        summaries[mid]['padded_minus_clean']=statistics.mean(x['oracle_padded']-x['oracle_audited'] for x in group)
        summaries[mid]['failure_n']=sum(x['padding_failure'] for x in group)
    audits=receipt['padding_audits']
    result={'status':'PASS_INDEPENDENT_SUPPLEMENT_AUDIT' if not mismatch and not token_mismatch else 'FAIL_RECONCILIATION',
        'record_n':len(rows),'padding_calls':len(pairs),'prior_calls':len(prior),'score_mismatch_n':len(mismatch),
        'token_mismatch_n':len(token_mismatch),'groups':summaries,'paired_results':pairs,'prior_results':prior,
        'length_max_relative_difference':max(abs(x['length_difference_reference_tokens'])/x['target_reference_tokens'] for x in audits),
        'retrieval_evidence_position_undefined_n':sum(x['first_evidence_retrieval_index'] is None for x in audits),
        'failures_retained_n':sum(bool(x['error_type']) for x in rows),'records':artifact(path),'inputs':artifact(run/'S2_SUPPLEMENT_INPUTS.json'),
        'human_review_status':'NOT_REVIEWED','budget':budget.snapshot(),
        'limitations':['Exposed diagnostic subset of 12; not a final holdout or population benefit claim.',
            'Padding is same-paper nonannotated low lexical overlap source text; semantic unrelatedness not human verified.',
            'Complete source sentence boundaries kept, resulting lengths approximate rather than exactly equal.',
            'When retrieval has no annotated evidence, padding evidence placed near middle; no matching location exists.',
            'Prior correctness is literal/alias proxy; no-context grounding fixture behavior scored separately.']}
    atomic_json(run/'S2_SUPPLEMENT_ANALYSIS.json',result)
    print(canonical_json({k:result[k] for k in ('status','record_n','groups','prior_results','length_max_relative_difference','retrieval_evidence_position_undefined_n','budget')}))
    return 0 if result['status'].startswith('PASS') else 2


if __name__=='__main__':raise SystemExit(main())
