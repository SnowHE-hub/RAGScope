"""Counted source-preserving atomization then existing calibrated NLI for the small dev task cohort."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import sys
PROJECT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(PROJECT));sys.path.insert(0,str(PROJECT/'scripts'));sys.path.insert(0,str(PROJECT/'scripts/v3'))
from freeze_generation_v2 import model_client_factory
from ragscope.v3.resident_claim_judge import ResidentCachedJudge as CachedJudge
from filelock import FileLock
from ragscope.experiments.ledger import RecordStore,read_records
from ragscope.experiments.runtime import configure,require_ram
from ragscope.indexing.identity import artifact,atomic_json,canonical_json,signature
from ragscope.v3.audit import overlap
from ragscope.v3.claim_metrics import bounds,proposition,retained_failure_metrics,score_claims
from ragscope.v3.gpu import GpuSession
from ragscope.v3.local_runtime import ManagedOllama,counted_chat
from ragscope.v3.nli import WindowedNLI
from ragscope.v3.quote_atoms import QUOTE_SYSTEM,quote_schema,quote_spans


def span_fraction(point,pieces):
    lo,hi=point['canonical_start'],point['canonical_end'];end=lo;covered=0
    for a,b in sorted((max(lo,p['canonical_start']),min(hi,p['canonical_end'])) for p in pieces if p['doc_id']==point['doc_id']):
        if a<b:covered+=max(0,b-max(a,end));end=max(end,b)
    return covered/(hi-lo)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign',required=True);p.add_argument('--phase',choices=['atomize','nli','all'],default='all')
    p.add_argument('--resume',action='store_true');a=p.parse_args();runtime=Path(configure(cpu_only=True)['home']).resolve();run=Path(a.campaign).resolve()
    if run.parent!=runtime/'runs':raise ValueError('Wrong campaign')
    completion=json.loads((run/'S4_DEVELOPMENT_TASK_GENERATION_COMPLETION.json').read_text('utf8'))
    if artifact(completion['records']['path'])!=completion['records']:raise ValueError('Task generation changed')
    rows=read_records(Path(completion['records']['path']));manifest=json.loads((run/'S4_DEVELOPMENT_TASK_CONTEXT_MANIFEST.json').read_text('utf8'))
    contexts={e['input_id']:e for e in map(json.loads,Path(manifest['inputs']['path']).read_text('utf8').splitlines())}
    gold_path=run/'S4_DEVELOPMENT_TASK_GOLD_v2.jsonl';golds={g['query_id']:g for g in map(json.loads,gold_path.read_text('utf8').splitlines())}
    grouped={qid:[r for r in rows if r['question_id']==qid] for qid in golds}
    require_ram(2*1024**3)
    with FileLock(str(run/'writer.lock'),timeout=0):
        atom_config={'runner':artifact(PROJECT/'scripts/v3/score_development_source_tasks.py'),'answers':completion['records'],'references':artifact(gold_path),
            'prompt':QUOTE_SYSTEM,'quote_implementation':artifact(PROJECT/'ragscope/v3/quote_atoms.py'),
            'protocol':'quote-only-v2','num_ctx':8192,'max_output_tokens':4096,'human_review_status':'NOT_REVIEWED'}
        atoms=RecordStore(run/'S4_DEVELOPMENT_TASK_ATOMS.jsonl',atom_config,max_records=len(golds))
        pending=[qid for qid in golds if qid not in atoms.ids]
        if atoms.ids and not a.resume:raise ValueError('Existing atoms need resume')
        if pending and a.phase in {'atomize','all'}:
            old=json.loads((runtime/'runs/20261014_v2_generation/EXPERIMENT_MANIFEST.json').read_text('utf8'))
            dev={json.loads(l)['query_id'] for l in (runtime/'corpora/20261014_v2/queries_dev.jsonl').read_text('utf8').splitlines()}
            original,_=model_client_factory(old['provenance']['spec']['models']['qwen_4b'],dev)
            with ManagedOllama(run,runtime/'tools/ollama-0.34.4/ollama.exe',num_ctx=8192) as service:
                client=service.client(original,num_ctx=8192)
                for qid in pending:
                    texts=[{'source_id':'reference','text':golds[qid]['reference_answer']}]
                    texts.extend({'source_id':r['condition'],'text':r['parsed_response']['answer']} for r in grouped[qid] if r['parsed_response'] and not r['parsed_response']['abstain'])
                    q=contexts[grouped[qid][0]['input_id']]['question'];rid='S4:source_task_atoms:'+qid;raw={};error=None;segments=[]
                    try:
                        raw=counted_chat(run,service,client,request_id=rid,stage='S4_task_workflows_router',
                            messages=[{'role':'system','content':QUOTE_SYSTEM},{'role':'user','content':json.dumps({'query_id':qid,'question':q,'texts':texts},ensure_ascii=False)}],
                            max_tokens=4096,temperature=0,output_schema=quote_schema([t['source_id'] for t in texts]),purpose='source_task_quote_segmentation')
                        returned=json.loads(raw['content'])['sources'];expected={t['source_id']:t['text'] for t in texts}
                        if len(returned)!=len(expected) or {r['source_id'] for r in returned}!=set(expected):raise ValueError('Atom source mapping differs')
                        segments=[{'source_id':r['source_id'],'units':quote_spans(expected[r['source_id']],r['units'])} for r in returned]
                    except Exception as exc:error={'kind':type(exc).__name__,'message':str(exc)}
                    atoms.add(qid,{'raw_response':raw,'segments':segments if error is None else None,'error':error,'budget_ticket':rid,
                        'original_texts_unchanged':True,'human_review_status':'NOT_REVIEWED'})
                    print(canonical_json({'event':'SOURCE_TASK_ATOMS','groups':len(atoms.ids),'error':error}),flush=True)
                    if error and error['kind']!='ValueError':raise RuntimeError('Atomizer failure retained; stop bounded cohort')
        if a.phase=='atomize':return 0
        if set(atoms.ids)!=set(golds):raise ValueError('Task atomization population incomplete')
        calibrated=json.loads((run/'NLI_GPU_SCORING_CALIBRATION.json').read_text('utf8'))
        if not calibrated['declarative_eligible'] or not calibrated['completion_eligible']:raise ValueError('Existing GPU engineering calibration failed')
        if any(calibrated['config'][k]!=importlib.metadata.version(k) for k in ('torch','transformers')):raise ValueError('Judge dependency stack differs')
        chosen=next(m for m in json.loads((run/'METRIC_MODEL_MANIFEST.json').read_text('utf8'))['models'] if m['model'].startswith('MoritzLaurer/'))
        score_config={'runner':artifact(__file__),'answers':completion['records'],'references':artifact(gold_path),'atoms':artifact(atoms.path),
            'judge':chosen,'judge_calibration':artifact(run/'NLI_GPU_SCORING_CALIBRATION.json'),
            'threshold':.8,'maximum':512,'dtype':'float32','device':'cuda:0','metric_implementation':artifact(PROJECT/'ragscope/v3/claim_metrics.py'),
            'human_review_status':'NOT_REVIEWED','evaluation_role':'SMALL_SYNTHETIC_DEVELOPMENT_NOT_HOLDOUT'}
        scores=RecordStore(run/'S4_DEVELOPMENT_TASK_SCORES_resident_v2.jsonl',score_config,max_records=len(rows))
        relations=RecordStore(run/'S4_DEVELOPMENT_TASK_NLI_RELATIONS_resident_v2.jsonl',score_config,max_records=5000)
        atom_map={r['record_id']:r for r in atoms.records};pending_rows=[r for r in rows if r['record_id'] not in scores.ids]
        if pending_rows:
            with GpuSession(run) as gpu:
                nli=WindowedNLI(chosen['directory'],device='cuda:0')
                judge=CachedJudge(nli,relations,completion_gate=True,gpu=gpu)
                try:
                    for r in pending_rows:
                        qid=r['question_id'];entry=contexts[r['input_id']];gold=golds[qid];parsed=r['parsed_response']
                        fractions=[span_fraction(point,entry['context_bundle']['pieces']) for point in gold['source_keypoints']]
                        if r['error_type'] or not parsed or parsed['abstain']:
                            metrics=retained_failure_metrics('FAILED_OR_ABSTAINED_ON_SOURCE_TASK');keypoint_verdicts=['NOT_ENTAILED_PROXY']*len(fractions);word=0.0
                        else:
                            atom=atom_map[qid]
                            units=next((s['units'] for s in atom['segments'] if s['source_id']==r['condition']),[]) if atom['segments'] else []
                            if not units:units=[{'start':0,'end':len(parsed['answer']),'kind':'unclear','source_quote':parsed['answer']}]
                            metrics=score_claims(entry['question'],parsed['answer'],units,[gold['reference_answer']],entry['context_bundle']['pieces'],parsed['citations'],judge)
                            keypoint_verdicts=[judge(proposition(entry['question'],parsed['answer']),proposition(entry['question'],point['text']))['verdict'] for point in gold['source_keypoints']]
                            word=overlap(parsed['answer'],[gold['reference_answer']])['legacy_token_f1']
                        scores.add(r['record_id'],{'query_id':qid,'condition':r['condition'],'task_kind':entry['task_kind'],'word_f1':word,
                            'source_keypoint_context_fractions':fractions,'source_keypoint_context_full_coverage':sum(f==1 for f in fractions)/len(fractions),
                            'source_keypoint_entailment_proxy':bounds(keypoint_verdicts),'source_keypoint_verdicts':keypoint_verdicts,
                            'metrics':metrics,'failure':bool(r['error_type']),'abstain':parsed['abstain'] if parsed else None,
                            'source_response_record_hash':r['record_hash'],'human_review_status':'NOT_REVIEWED',
                            'reference_authorship':'SYNTHETIC_AI_SOURCE_QUOTES_VERIFIED_PARAPHRASE_NOT_HUMAN_REVIEWED',
                            'natural_judge_accuracy':'NOT_ESTABLISHED_BY_ENGINEERING_CALIBRATION'})
                        print(canonical_json({'event':'SOURCE_TASK_SCORE','records':len(scores.ids),'condition':r['condition']}),flush=True)
                finally:nli.close()
        result={'status':'PASS_SMALL_SOURCE_TASK_METRIC_EXECUTION','question_n':len(golds),'answer_n':len(scores.ids),
            'actual_nli_relations':len(relations.ids),'encoder_tokens':sum(r['result']['encoder_tokens'] for r in relations.records),
            'atomization_failure_n':sum(r['error'] is not None for r in atoms.records),'scores':artifact(scores.path),
            'relations':artifact(relations.path),'atoms':artifact(atoms.path),'human_review_status':'NOT_REVIEWED',
            'generalization':'NOT_ESTABLISHED_SMALL_UNBALANCED_AI_AUTHORED_COHORT','new_generator_training_trigger':False}
        atomic_json(run/'S4_DEVELOPMENT_TASK_METRICS_COMPLETION_resident_v2.json',result);print(canonical_json(result));return 0


if __name__=='__main__':raise SystemExit(main())
