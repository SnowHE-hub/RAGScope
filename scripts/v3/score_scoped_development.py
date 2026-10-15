"""Measure frozen100-question development claim proxies, preserving every failure and UNKNOWN."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import sys
PROJECT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(PROJECT));sys.path.insert(0,str(PROJECT/'scripts'))
from freeze_generation_v2 import model_client_factory
from filelock import FileLock
from ragscope.experiments.ledger import RecordStore,read_records
from ragscope.experiments.runtime import configure,require_ram
from ragscope.indexing.identity import artifact,atomic_json,canonical_json
from ragscope.v3.audit import reference_texts
from ragscope.v3.claim_metrics import retained_failure_metrics,score_claims
from ragscope.v3.gpu import GpuSession
from ragscope.v3.local_runtime import ManagedOllama,counted_chat
from ragscope.v3.nli import WindowedNLI
from ragscope.v3.quote_atoms import QUOTE_SYSTEM,quote_schema,quote_spans
from ragscope.v3.resident_claim_judge import ResidentCachedJudge
from ragscope.v3.splits import read_split


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign',required=True)
    p.add_argument('--phase',choices=['atomize','nli','all'],default='all');p.add_argument('--resume',action='store_true');a=p.parse_args()
    runtime=Path(configure(cpu_only=True)['home']).resolve();run=Path(a.campaign).resolve()
    if run.parent!=runtime/'runs':raise ValueError('Wrong campaign')
    completion_path=run/'S3_SCOPED_GENERATION_D_confirm_COMPLETION.json';completion=json.loads(completion_path.read_text('utf8'))
    if artifact(completion['records']['path'])!=completion['records'] or completion['formal_records']!=200:raise ValueError('Frozen paired100 results changed')
    rows=read_records(Path(completion['records']['path']));labels={q['query_id']:q for q in read_split(run,'D_confirm',purpose='development')}
    manifest=json.loads((run/'S3_SCOPED_CONTEXT_MANIFEST.json').read_text('utf8'))
    if artifact(manifest['inputs']['path'])!=manifest['inputs']:raise ValueError('Frozen contexts changed')
    contexts={e['input_id']:e for e in map(json.loads,Path(manifest['inputs']['path']).read_text('utf8').splitlines()) if e['dataset_role']=='D_confirm'}
    grouped={qid:[r for r in rows if r['question_id']==qid] for qid in labels}
    if len(grouped)!=100 or any(len(rs)!=2 for rs in grouped.values()):raise ValueError('Paired denominator changed')
    require_ram(2*1024**3)
    with FileLock(str(run/'writer.lock'),timeout=0):
        atom_config={'runner':artifact(__file__),'generation':artifact(completion_path),'references_source':artifact(run/'SPLITS_V3.json'),
            'prompt':QUOTE_SYSTEM,'quote_implementation':artifact(PROJECT/'ragscope/v3/quote_atoms.py'),
            'stage':'S3_retrieval_packing','num_ctx':8192,'max_output_tokens':4096,'human_review_status':'NOT_REVIEWED'}
        atoms=RecordStore(run/'S3_D_CONFIRM_CLAIM_ATOMS.jsonl',atom_config,max_records=100)
        if atoms.ids and not a.resume:raise ValueError('Existing atoms require resume')
        pending=[qid for qid in labels if qid not in atoms.ids]
        if pending and a.phase in {'all','atomize'}:
            old=json.loads((runtime/'runs/20261014_v2_generation/EXPERIMENT_MANIFEST.json').read_text('utf8'))
            dev={json.loads(l)['query_id'] for l in (runtime/'corpora/20261014_v2/queries_dev.jsonl').read_text('utf8').splitlines()}
            original,_=model_client_factory(old['provenance']['spec']['models']['qwen_4b'],dev)
            for offset in range(0,len(pending),20):
                require_ram(2*1024**3)
                with ManagedOllama(run,runtime/'tools/ollama-0.34.4/ollama.exe',num_ctx=8192) as service:
                    client=service.client(original,num_ctx=8192)
                    for qid in pending[offset:offset+20]:
                        texts=[{'source_id':'reference_'+str(i),'text':ref} for i,ref in enumerate(reference_texts(labels[qid]))]
                        texts.extend({'source_id':r['condition'],'text':r['parsed_response']['answer']} for r in grouped[qid]
                            if not r['error_type'] and r['parsed_response'] and not r['parsed_response']['abstain'])
                        if not texts:
                            atoms.add(qid,{'segments':[],'error':None,'raw_response':None,'budget_ticket':None,'status':'NO_ASSERTIONS_NO_MODEL_CALL'});continue
                        rid='S3:D_confirm_claim_atoms:'+qid;raw={};segments=[];error=None
                        try:
                            raw=counted_chat(run,service,client,request_id=rid,stage='S3_retrieval_packing',
                                messages=[{'role':'system','content':QUOTE_SYSTEM},{'role':'user','content':json.dumps(
                                    {'query_id':qid,'question':labels[qid]['question'],'texts':texts},ensure_ascii=False)}],
                                max_tokens=4096,temperature=0,output_schema=quote_schema([t['source_id'] for t in texts]),purpose='development_claim_segmentation')
                            expected={t['source_id']:t['text'] for t in texts};returned=json.loads(raw['content'])['sources']
                            if len(returned)!=len(expected) or {r['source_id'] for r in returned}!=set(expected):raise ValueError('Source mapping mismatch')
                            segments=[{'source_id':r['source_id'],'units':quote_spans(expected[r['source_id']],r['units'])} for r in returned]
                        except Exception as exc:error={'kind':type(exc).__name__,'message':str(exc)}
                        atoms.add(qid,{'segments':segments if error is None else None,'error':error,'raw_response':raw,
                            'budget_ticket':rid,'original_texts_unchanged':True,'human_review_status':'NOT_REVIEWED'})
                        print(canonical_json({'event':'D_CONFIRM_CLAIM_ATOMS','groups':len(atoms.ids),'error':error}),flush=True)
                        if error and error['kind']!='ValueError':raise RuntimeError('Atomization backend failure retained; stop bounded batch')
        if a.phase=='atomize':return 0
        if set(atoms.ids)!=set(labels):raise ValueError('Development atomization population incomplete')
        calibrated=json.loads((run/'NLI_GPU_SCORING_CALIBRATION.json').read_text('utf8'))
        if not calibrated['declarative_eligible'] or not calibrated['completion_eligible']:raise ValueError('Engineering calibration failed')
        if any(calibrated['config'][k]!=importlib.metadata.version(k) for k in ('torch','transformers')):raise ValueError('Judge stack changed')
        chosen=next(m for m in json.loads((run/'METRIC_MODEL_MANIFEST.json').read_text('utf8'))['models'] if m['model'].startswith('MoritzLaurer/'))
        config={'runner':artifact(__file__),'generation':artifact(completion_path),'atoms':artifact(atoms.path),'model':chosen,
            'calibration':artifact(run/'NLI_GPU_SCORING_CALIBRATION.json'),'metrics':artifact(PROJECT/'ragscope/v3/claim_metrics.py'),
            'resident_guard':artifact(PROJECT/'ragscope/v3/resident_claim_judge.py'),'threshold':.8,'maximum':512,'dtype':'float32','device':'cuda:0',
            'preload_free_with_reserve_bytes':3*1024**3,'resident_additional_bytes':64*1024**2,'resident_reserve_bytes':1024**3,
            'human_review_status':'NOT_REVIEWED','evaluation_role':'D_CONFIRM_NOT_HOLDOUT'}
        store=RecordStore(run/'S3_D_CONFIRM_CLAIM_SCORES.jsonl',config,max_records=200)
        relations=RecordStore(run/'S3_D_CONFIRM_NLI_RELATIONS.jsonl',config,max_records=30000)
        pending=[r for r in rows if r['record_id'] not in store.ids];amap={r['record_id']:r for r in atoms.records}
        if pending:
            with GpuSession(run) as gpu:
                nli=WindowedNLI(chosen['directory'],device='cuda:0');judge=ResidentCachedJudge(nli,relations,completion_gate=True,gpu=gpu)
                try:
                    for r in pending:
                        q=labels[r['question_id']];e=contexts[r['input_id']];parsed=r['parsed_response'];answerable=q['answerability']=='answerable'
                        if not answerable:
                            metrics={'claim_f1_proxy':None,'claim_f1_proxy_lower':None,'claim_f1_proxy_upper':None,
                                'status':'UNANSWERABLE_NA_NOT_ZERO'}
                        elif r['error_type'] or not parsed or parsed['abstain']:metrics=retained_failure_metrics('FAILED_OR_ABSTAINED_ON_ANSWERABLE')
                        else:
                            atom=amap[q['query_id']]
                            units=next((s['units'] for s in atom['segments'] if s['source_id']==r['condition']),[]) if atom['segments'] else []
                            if not units:units=[{'source_quote':parsed['answer'],'kind':'unclear','start':0,'end':len(parsed['answer'])}]
                            metrics=score_claims(q['question'],parsed['answer'],units,reference_texts(q),e['context_bundle']['pieces'],parsed['citations'],judge)
                        store.add(r['record_id'],{'query_id':q['query_id'],'paper_id':q['doc_id'],'condition':r['condition'],'answerable':answerable,
                            'metrics':metrics,'failure':bool(r['error_type']),'abstain':parsed['abstain'] if parsed else None,
                            'source_response_record_hash':r['record_hash'],'human_review_status':'NOT_REVIEWED'})
                        print(canonical_json({'event':'D_CONFIRM_CLAIM_SCORE','records':len(store.ids),'relations':len(relations.ids)}),flush=True)
                finally:nli.close()
        result={'status':'PASS_200_DEVELOPMENT_CLAIM_SCORE_ROWS','answer_rows_n':len(store.ids),'question_n':100,
            'actual_nli_relations':len(relations.ids),'scores':artifact(store.path),'relations':artifact(relations.path),
            'atomization_failures':sum(r['error'] is not None for r in atoms.records),'human_review_status':'NOT_REVIEWED',
            'natural_judge_accuracy':'NOT_ESTABLISHED_BY_ENGINEERING_CALIBRATION','old_full_scoring_restarted':False}
        atomic_json(run/'S3_D_CONFIRM_CLAIM_COMPLETION.json',result);print(canonical_json(result));return 0


if __name__=='__main__':raise SystemExit(main())
