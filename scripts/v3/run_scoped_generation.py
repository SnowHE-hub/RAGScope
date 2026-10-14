"""Real bounded paired development generation on frozen compact contexts."""
import argparse
import json
from pathlib import Path
import sys
import time
PROJECT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(PROJECT));sys.path.insert(0,str(PROJECT/'scripts'))
from freeze_generation_v2 import model_client_factory
from filelock import FileLock
from ragscope.experiments.ledger import RecordStore,read_records
from ragscope.experiments.runtime import configure,require_ram
from ragscope.generation.pipeline import RAGPipeline
from ragscope.indexing.identity import artifact,atomic_json,canonical_json,signature
from ragscope.v3.audit import overlap,reference_texts
from ragscope.v3.budget import Budget
from ragscope.v3.generation_protocols import ProtocolClient,protocol_messages,protocol_schema
from ragscope.v3.local_runtime import ManagedOllama,counted_chat
from ragscope.v3.records import canonical_payload
from ragscope.v3.splits import read_split


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign',required=True)
    p.add_argument('--split',choices=['D_search','D_confirm'],required=True);p.add_argument('--resume',action='store_true');a=p.parse_args()
    runtime=Path(configure(cpu_only=True)['home']).resolve();run=Path(a.campaign).resolve()
    if run.parent!=runtime/'runs':raise ValueError('Wrong campaign')
    require_ram(2*1024**3)
    with FileLock(str(run/'writer.lock'),timeout=0):
        manifest_path=run/'S3_SCOPED_CONTEXT_MANIFEST.json';manifest=json.loads(manifest_path.read_text('utf8'))
        if manifest['status']!='PASS_FROZEN_COMPACT_CONTEXTS' or artifact(manifest['inputs']['path'])!=manifest['inputs']:raise ValueError('Frozen compact inputs invalid')
        wanted=set(manifest['initial_D_search_query_ids']) if a.split=='D_search' else {q['query_id'] for q in read_split(run,'D_confirm',purpose='inference')}
        entries=[]
        with Path(manifest['inputs']['path']).open(encoding='utf8') as f:
            for l in f:
                e=json.loads(l)
                if e['dataset_role']==a.split and e['query_id'] in wanted:entries.append(e)
        if len(entries)!=2*len(wanted):raise ValueError('Paired development inputs incomplete')
        labels={q['query_id']:q for q in read_split(run,a.split,purpose='development')}
        config={'inputs':artifact(manifest_path),'query_ids':sorted(wanted),'split':a.split,
            'runner':artifact(__file__),'protocol':artifact(PROJECT/'ragscope/v3/generation_protocols.py'),
            'record_schema':artifact(PROJECT/'ragscope/v3/records.py'),'runtime':artifact(PROJECT/'ragscope/v3/local_runtime.py'),
            'temperature':0,'seed':42,'max_output_tokens':512,'num_ctx':4096,'answer_cache':False,
            'resource_session_max_calls':30,'baseline_qualification':manifest['baseline_qualification']}
        journal=run/('S3_SCOPED_GENERATION_'+a.split+'.jsonl')
        if journal.exists() and not a.resume:raise ValueError('Existing generation needs --resume')
        store=RecordStore(journal,config,max_records=len(entries))
        warmups=RecordStore(run/('S3_SCOPED_WARMUPS_'+a.split+'.jsonl'),config,max_records=10)
        pending=[e for e in entries if 'S3:'+a.split+':'+e['input_id'] not in store.ids]
        dev={json.loads(l)['query_id'] for l in (runtime/'corpora/20261014_v2/queries_dev.jsonl').read_text('utf8').splitlines()}
        original,_=model_client_factory(manifest['generator_spec'],dev)
        if original.identity!=manifest['generator_model_identity']:raise ValueError('Frozen generator identity differs')
        budget=Budget(run)
        for start in range(0,len(pending),29):
            with ManagedOllama(run,runtime/'tools/ollama-0.34.4/ollama.exe',num_ctx=4096) as service:
                tasks=[('warmup',pending[start])]+[('formal',e) for e in pending[start:start+29]]
                for kind,e in tasks:
                    rid='S3:'+a.split+':'+e['input_id'] if kind=='formal' else 'S3:warmup:'+a.split+':'+str(len(warmups.ids))
                    require_ram(2*1024**3)
                    client=ProtocolClient(original,service,protocol='G0');messages=protocol_messages(e['question'],e['context_bundle'],'G0')
                    response,parsed,error={},None,None;began=time.perf_counter()
                    try:
                        response=counted_chat(run,service,client,request_id=rid,stage='S3_retrieval_packing',messages=messages,
                            max_tokens=512,temperature=0,output_schema=protocol_schema(e['context_bundle'],'G0'),purpose='S3_scoped_'+kind)
                        output=RAGPipeline.parse_output(response['content']);parsed={k:output[k] for k in ('answer','citations','abstain')}
                        if client.last_metadata.get('token_count_match') is not True:raise RuntimeError('Actual prompt count differs')
                        visible={p['chunk_id'] for p in e['context_bundle']['pieces']}
                        if any(c not in visible for c in parsed['citations']):raise ValueError('Output citation invisible')
                    except Exception as exc:
                        with budget.lock:
                            if rid not in budget._state()['reserved']:raise
                        error={'kind':type(exc).__name__,'message':str(exc)}
                        if not response:response={'raw_backend_response':client.raw_response,'backend_metadata':client.last_metadata}
                    label=labels[e['query_id']];answerable=label['answerability']=='answerable';refs=reference_texts(label)
                    metric=overlap(parsed['answer'],refs) if parsed and not error and not parsed['abstain'] and answerable else None
                    row=canonical_payload({'question_id':e['query_id'],'dataset_role':a.split,
                        'exposure_status':'EXPOSED_DIAGNOSTIC' if a.split=='D_search' else 'DEVELOPMENT_VALIDATION',
                        'task_type_source':'UNLABELED_OFFICIAL_QASPER','user_scope':e['scope'],'condition':e['condition'],
                        'input_id':e['input_id'],'prompt_hash':signature(messages),'model_identity':client.identity,
                        'actual_prompt_tokens':client.last_metadata.get('prompt_eval_count'),
                        'context_intervals':[{'doc_id':p['doc_id'],'start':p['canonical_start'],'end':p['canonical_end']} for p in e['context_bundle']['pieces']],
                        'retriever_candidates':e['retriever_candidates'],'rank_scores':[p['score'] for p in e['retriever_candidates']],
                        'packing_drops':e['context_bundle']['truncations'],'raw_response':response,'parsed_response':parsed,
                        'stop_reason':client.last_metadata.get('done_reason'),'output_tokens':client.last_metadata.get('eval_count'),
                        'success':error is None,'error_type':error,'wall_time_ms':(time.perf_counter()-began)*1000,
                        'measured_resource':client.last_metadata.get('resources'),'budget_ticket':rid,
                        'metric_version':'v3-independent-word-overlap; claim/support/citation semantics pending',
                        'proxy_unknowns':['claim_f1_proxy','support_proxy','citation_completeness_proxy'],
                        'document_answerable':answerable,'legacy_token_f1':None if not answerable else metric['legacy_token_f1'] if metric else 0.0,
                        'word_metrics':metric,'actual_sent_payload':client.sent_payload,'purpose':kind},campaign_id=run.name,task_id='S3_scoped_'+kind)
                    (store if kind=='formal' else warmups).add(rid,row)
                    print(canonical_json({'event':'SCOPED_GENERATION','split':a.split,'formal_records':len(store.ids),
                        'kind':kind,'condition':e['condition'],'failed':bool(error)}),flush=True)
                    if error and error['kind']!='ValueError':raise RuntimeError('Recorded backend/count failure; stop bounded batch')
        result={'status':'PASS_COMPLETE_PAIRED_GENERATION' if len(store.ids)==len(entries) else 'INCOMPLETE',
            'split':a.split,'questions_n':len(wanted),'formal_records':len(store.ids),'warmups':len(warmups.ids),
            'records':artifact(journal),'context_manifest':artifact(manifest_path),'budget':budget.snapshot(),
            'claim_metrics_status':'PENDING_NOT_WORD_F1','baseline_qualification':manifest['baseline_qualification'],'human_review_status':'NOT_REVIEWED'}
        atomic_json(run/('S3_SCOPED_GENERATION_'+a.split+'_COMPLETION.json'),result)
        print(canonical_json({'status':result['status'],'records':len(store.ids),'warmups':len(warmups.ids)}));return 0


if __name__=='__main__':raise SystemExit(main())
