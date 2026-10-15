"""Requirement-linked acceptance, including mandatory blocks and actual observed exits."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import re
import sys
PROJECT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(PROJECT))
from ragscope.indexing.identity import artifact,atomic_json,canonical_json
from ragscope.v3.budget import Budget


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign',required=True);a=p.parse_args();run=Path(a.campaign).resolve()
    if run.parent!=Path('D:/RAGScope_Runtime/runs').resolve():raise ValueError('Wrong campaign')
    report=PROJECT/'docs/execution/v3';requirements=[]
    def add(key,status,description,names,command,exit_code,limitations=''):
        refs=[]
        for name in names:
            path=PROJECT/name if name.startswith(('docs/','scripts/','ui/','configs/','tests/','LOCAL_')) else run/name
            if not path.is_file():raise ValueError('Required evidence missing: '+str(path))
            refs.append(artifact(path))
        requirements.append({'id':key,'status':status,'description':description,'command':command,
            'actual_exit_code':exit_code,'evidence':refs,'limitations':limitations})
    root='docs/execution/v3/ROOT_CAUSE_VERDICTS.md';resource='FINAL_RESOURCE_STATE.json'
    add('0.1','BLOCKED','Diagnose, implement and validate benefits without replacing missing unseen comparison.',[root,'CANDIDATE_LOCK.json'],'freeze_candidate_gate.py',0,'Required H_QA pair remains blocked; no confirmed overall gain.')
    add('0.2','PASS','Separate workflow, execution, performance, release and human status.',['docs/execution/v3/QUALITY_SCORECARD.json','CANDIDATE_LOCK.json'],'build_delivery_reports.py',0)
    add('0.3','PASS','Autonomous bounded stage decisions; no approval wait.',['docs/execution/v3/DECISIONS.md','BUDGET.jsonl'],'Recorded campaign execution',None,'Administrative contract; not a model-quality claim.')
    add('0.4','PASS','D runtime, exact ownership, serial GPU, resources released, manual original-training resume.',['FINAL_RESOURCE_STATE.json','S6_ONLINE_COST_COMPLETION.json','S6_APPLICATION_CACHE_VERIFICATION.json'],'audit_final_resources.py',0)
    receipts=[(path,json.loads(path.read_text('utf8'))) for path in run.glob('V2_PRESERVATION_RECHECK_*.json')]
    eligible=[(path,row) for path,row in receipts if row.get('baseline_commit')=='d431f646a91502c2c75f1e991521f4ec0c519a1d' and row.get('tracked_files')==356 and 'observed_utc' in row]
    if not eligible:raise ValueError('No complete exact-baseline preservation receipt')
    preservation=max(eligible,key=lambda pair:pair[1]['observed_utc'])[0]
    if json.loads(preservation.read_text('utf8'))['status']!='PASS_V2_SOURCE_PRESERVED':raise ValueError('v2 source protection failed')
    add('0.5','PASS','Original v2 source, models and five environments preserved; new branch and immutable journals.',[preservation.name,'FINAL_RUNTIME_IDENTITY.json','BUDGET.jsonl','S6_CAMPAIGN_REUSE_FINAL.log'],'audit_v2_preservation.py; recheck_runtime_identity.py; run_campaign.py --resume --stage S6 --until S6',0,'356 tracked files;25 pre-existing line-ending differences.398 runtime identities rehashed,0differences. No byte-identical source claim.')
    add('1.2','PASS','All H01-H12 have evidence, counterevidence and uncertainty.',[root,'HYPOTHESES_CURRENT_VERDICTS.jsonl'],'report_root_cause_verdicts.py',0,'Unresolved causality retained; hypotheses not treated as confirmed bugs.')
    add('2','PASS','Full budget reservations, failed/warmup/judge/UI charges.',['BUDGET.jsonl','S6_ONLINE_COST_COMPLETION.json'],'Budget._state()',0,'NLI/embedding non-generative calls are separate ledgers.')
    add('3.S0.1','PASS','Original source journals, adapters, training data and identity inventory.',['SOURCE_INVENTORY.json','S0_AUDIT_RECEIPT.json'],'audit_sources.py',0)
    add('3.S0.2','PASS','Independent5520/39/32 baseline recount and task semantics.',['BASELINE_RECOUNT.json','RECOUNT_ALL_OLD.json'],'audit_sources.py; rescore_old.py',0)
    add('3.S0.3','PASS','Primary paper-disjoint splits and auxiliary synthetic scope/tasks sealed.',['SPLITS_V3.json','AUXILIARY_SPLIT_VALIDATION.json','H_TASKS_SOURCE_REPAIR.json'],'seal_splits.py; prepare_auxiliary_splits.py --resume; repair_task_reference_whitespace.py',0,'H_tasks AI source-backed, not official QA or operator-blind; H_QA gold stays sealed.')
    add('4.S1.1','PASS','All5720 answer metrics, legitimate multi-reference/yes-no/failed denominators.',['RECOUNT_ALL_OLD.json','S6_FULL_V3_REGRESSION.log'],'rescore_old.py; pytest test_v3_*.py',0)
    add('4.S1.2','BLOCKED','Full original semantic/factual/support/citation scoring.',['BERTSCORE_DEFAULT_RESOURCE_BLOCK.json','CLAIM_SCORING_RESOURCE_BLOCK.json','CLAIM_ALL_OLD_COVERAGE_AND_UNKNOWNS.jsonl','S3_D_CONFIRM_CLAIM_GATE.json'],'close_claim_resource_block.py; audit_development_claim_gate.py',0,'Official default BERTScore resource-blocked; pinned alternative completed. Full old NLI exhausted bounded GPU/CPU attempts. UNKNOWN retains full population.')
    add('4.S1.3','BLOCKED','Engineering calibration and120 natural blinded proxy reviews.',['CLAIM_SCORING_RESOURCE_BLOCK.json','BLIND_REVIEW_PACKAGE.jsonl','BLIND_REVIEW_NLI_RESOURCE_PARTIAL.jsonl','HUMAN_REVIEW_PACKAGE_RECEIPT.json','S6_FULL_V3_REGRESSION.log'],'close_claim_resource_block.py; export_human_review_package.py',0,'Constructed calibrations passed; natural120 review only12 partial scores. Human NOT_REVIEWED.')
    add('4.S1.4','PASS','Document answerability, actual visible sufficiency proxy and abstention separated.',['LORA_23_REFUSAL_CONTEXT_PROXY.jsonl','CONTEXT_SUFFICIENCY_PROXY.jsonl','CONTEXT_SUFFICIENCY_SUMMARY.json',root],'summarize_context_diagnosis.py',0,'117 context-only AI judgements; missing historical TTFT not invented; natural sufficiency truth remains proxy.')
    add('4.S1.5','PASS','LoRA and scale/length/refusal/format decomposition.',['LORA_OLD_RECOUNT.json','docs/execution/v3/LORA_DIAGNOSIS.md',root],'attribute_old_outputs.py; audit_lora_runtime_diagnostics.py',0)
    add('5.S2.1','PASS','72 real direct/pipeline smoke calls,36 equal pairs.',['S2_PROTOCOL_RESULTS.jsonl','S2_PROTOCOL_ANALYSIS.json'],'run_protocol_study.py; audit_protocol_results.py',0,'Fixed fixtures only, not all possible prompts/backends.')
    add('5.S2.2','PASS','864 main fixed-protocol records plus prior/padded diagnostics.',['S2_PROTOCOL_ANALYSIS.json','S2_SUPPLEMENT_COMPLETION.json','S2_SUPPLEMENT_ANALYSIS.json'],'audit_protocol_results.py; audit_protocol_supplements.py',0,'Three seeds are repeated measures; not independent question count.')
    add('5.S2.3','NOT_RUN_GATE','Additional precision/backend/model comparisons only when necessary.',['S2_PROTOCOL_ANALYSIS.json','S3_SCOPED_GENERATION_D_confirm_ANALYSIS.json'],'No new9B BF16 grid',None,'No admitted9B product candidate; fixed main2B confirmation performed. No quantization cause asserted.')
    for key,desc,names in [
        ('6.S3.1','Scope and visible-object conditions',['S3_DOCUMENT_CANDIDATES_D_SEARCH.json','S3_DOCUMENT_DENSE_CANDIDATES.json','S6_H_SCOPE_COMPLETION.json']),
        ('6.S3.2','Actual500 old retrieval funnels and9519 windows',['S3_OLD_RETRIEVAL_FUNNEL.json']),
        ('6.S3.4','Real document sparse/dense, scoped bypass and global BM25 escape',['S3_DOCUMENT_BM25_INDEX.json','S3_DOCUMENT_DENSE_INDEX.json','S3_HIERARCHY_REAL_LOOKUPS.json','S6_H_SCOPE_COMPLETION.json']),
        ('6.S3.5','Natural BGE/Qwen relevance scores and max/top2 confirmation',['S3_NATURAL_RERANK_D_search_ANALYSIS.json','S3_NATURAL_RERANK_D_confirm_ANALYSIS.json']),
        ('6.S3.6','Real core-first control, confirmed100 and negative admission gate',['S3_SCOPED_GENERATION_D_confirm_ANALYSIS.json','S3_D_CONFIRM_CLAIM_GATE.json','CANDIDATE_LOCK.json'])]:
        add(key,'PASS',desc,names,'Independent stage audit and saved actual records',0,'Diagnostic execution; no automatic quality release claimed.')
    add('6.S3.3','BLOCKED','Three chunkers, four sparse variants and selected full chunk Dense indices.',['S3_CHUNK_SPARSE_INDICES.json','S3_CHUNK_DENSE_FINAL_RESOURCE_BLOCK.json','S3_SCOPED_VECTOR_COMPLETION.json'],'build_chunk_dense_index_resource_v3.py',1,'12096 partial rows retained; compact141-paper completion is a qualified alternative, not full-corpus Dense.')
    add('7.S4.1','PASS','Visible-input intent/scope/span proxy labels; no gold features.',['S4_WORKFLOW_CONTEXT_MANIFEST.json','S6_FULL_V3_REGRESSION.log'],'prepare_workflow_labels.py',0,'Rule/AI labels, no fabricated summary class.')
    add('7.S4.2','BLOCKED','Different local/compare/summary actions and broad source-keypoint coverage.',['S4_WORKFLOW_GENERATION_COMPLETION.json','S4_DEVELOPMENT_TASK_METRICS_COMPLETION_resident_v2.json','docs/execution/v3/TASK_KEYPOINT_DIAGNOSIS.md'],'run_workflow_generation.py; score_development_source_tasks_resident_v2.py',0,'180 actual workflow answers complete; only2 valid summaries/1 comparison after bounded source attempts.')
    add('7.S4.3','PASS','All240 utility entries, fixed/mapping/hindsight comparison.',['S4_WORKFLOW_UTILITY_ANALYSIS.json','S4_WORKFLOW_UTILITY_VECTORS.jsonl'],'audit_workflow_utility.py',0,'Hindsight is undeployable upper bound; mapped utility not better.')
    add('7.S4.4','NOT_RUN_GATE','New linear/MLP/LSTM/Laya classification contingent on useful workflows.',['S4_WORKFLOW_UTILITY_ANALYSIS.json','CANDIDATE_LOCK.json'],'No new router training',None,'Workflow prerequisite failed; no classifier-quality claims.')
    add('8.S5.1','PASS','972 mask/EOS numeric checks,20 decodes,8-fixture update/resume chain.',['S5_SUPERVISION_DISTRIBUTION_DIAGNOSIS.json','S5_SUPERVISION_SMOKE_v2/COMPLETION.json'],'audit_lora_supervision_distribution.py; run_lora_supervision_smoke_v2.py',0,'Initial zero-LR one-step failure retained;2-step compatibility tests chain only.')
    add('8.S5.2','BLOCKED','Same24 BF16 base/unmerged runtime, merge gate and fixed-token speed.',['S5_RUNTIME_INDEPENDENT_ANALYSIS.json','S5_MERGE_LOGIT_GATE.json','S5_RUNTIME_base_COMPLETION.json','S5_RUNTIME_unmerged_COMPLETION.json'],'run_lora_diagnostic_sequence.py',2,'Base/unmerged complete; merged function failed prior numeric tolerance, merged generation not run.')
    for key,desc in [('8.S5.3','Formal retraining'),('8.S5.4','Deployment-distribution training dataset'),('8.S5.5','Merged Q4 deployment conversion')]:
        add(key,'NOT_RUN_GATE',desc,['S5_LORA_REPORT_SOURCE_NOTES.json','S5_MERGE_LOGIT_GATE.json'],'No new formal adapter/configuration',None,'Primary judge and verified300 deployment samples prerequisites unmet; no extra training rounds.')
    add('9.S6.1','BLOCKED','Freeze candidates and once-only same unseenH_QA200 comparison.',['CANDIDATE_LOCK.json','SPLITS_V3.json'],'freeze_candidate_gate.py',0,'No candidate passes10.1. No final authorization or gold read; required comparison not completed.')
    add('9.S6.2','BLOCKED','H_scope40 and H_tasks final product tasks.',['S6_H_SCOPE_COMPLETION.json','CANDIDATE_LOCK.json'],'run_scope_validation.py',0,'Scope40 actual passed; final comparison/tasks not admitted and remain blocked.')
    add('9.S6.3','PASS','180 uncached actual live online requests and separate answer cache test.',['S6_ONLINE_COST_ANALYSIS.json','S6_APPLICATION_CACHE_VERIFICATION.json','S6_ONLINE_COST_FINAL_RESUME.log','S6_COLD_START_OBSERVATIONS.json'],'run_online_cost.py --resume; audit_online_cost.py; verify_workbench_application.py',0,'6 warmups excluded and startup metadata separate; first2 interrupted attempts retained; hotGPU bridge not oldCPU deployment. No TTFT invented.')
    add('9.S6.4','BLOCKED','2000 paper-cluster statistics for development/online; required final statistics.',['S3_SCOPED_GENERATION_D_confirm_ANALYSIS.json','S3_D_CONFIRM_CLAIM_GATE.json','S6_ONLINE_COST_ANALYSIS.json','CANDIDATE_LOCK.json'],'Independent paired bootstrap audits',0,'Development/online executed; finalH_QA unavailable. Uncertainty envelope not ordinary point CI.')
    for key in ('10.1','10.2','10.3','10.4'):
        add(key,'PASS','Apply frozen gates without changing judge/thresholds or enabling failed defaults.',['CANDIDATE_LOCK.json','docs/execution/v3/QUALITY_SCORECARD.json'],'freeze_candidate_gate.py',0,'Gate applied and failed; PASS here means decision integrity, never performance success.')
    add('11.1','PASS','Working help/dry-run/resume/stage/until/config protection and actual exit semantics.',['S6_CAMPAIGN_HELP.log','S6_CAMPAIGN_DRY_RUN_FINAL.log','S6_CAMPAIGN_REUSE_FINAL.log','S6_AUDIT_HELP.log'],'run_campaign.py --help/--dry-run/--resume; audit_campaign.py --help',0)
    add('11.2','PASS','All72 v3 regression checks and real application guards.',['S6_FULL_V3_REGRESSION_SCOPE_GUARD.log','S6_APPLICATION_CACHE_VERIFICATION.json','FINAL_RESOURCE_STATE.json'],'pytest test_v3_*.py; verify_workbench_application.py',0,'Mock fixtures prove functionality only; real experimental results are separate. Incomplete Hybrid scope rejects instead of silently generating with sparse fallback.')
    add('11.3','PASS','Canonical fields, hashes, real timing, failure/UNKNOWN and budget identity.',['S6_ONLINE_COST.jsonl','S6_APPLICATION_REQUESTS.jsonl','S6_FULL_V3_REGRESSION.log'],'RecordStore and canonical records regression; independent online audit',0,'Unmeasured values null; old reuse format preserved rather than rewritten as new calls.')
    add('11.4.cases','BLOCKED','At least3 real before/after cases per supported task class.',['BEFORE_AFTER_CASES_RECEIPT.json','docs/execution/v3/BEFORE_AFTER_CASES.md'],'build_delivery_reports.py',0,'Six real word-change QA cases exported; only2 summary/1 compare source pairs. Missing task examples not invented.')
    add('11.4.deliverables','PASS','Named reports, scorecard, human export, startup/rollback and resource receipt.',
        ['docs/execution/v3/EXECUTIVE_SUMMARY_ZH.md',root,'docs/execution/v3/QUALITY_SCORECARD.json','docs/execution/v3/BEFORE_AFTER_CASES.md',
         'HUMAN_REVIEW_PACKAGE.jsonl','docs/execution/v3/HUMAN_REVIEW_PACKAGE.md','LOCAL_RUN_V3.md','FINAL_RESOURCE_STATE.json','configs/v3/dependencies.lock.json','SOURCE_INVENTORY.json'],
        'build_delivery_reports.py; report_root_cause_verdicts.py; audit_final_resources.py',0,'Human package remainsNOT_REVIEWED; raw package stays local and out ofGit.')
    # Preserve every source requirement line verbatim and attach its controlling acceptance item.
    master=PROJECT/'docs/execution/v3/MASTER_PLAN.md';items=[];section='0.1'
    for i,line in enumerate(master.read_text('utf8').splitlines(),1):
        if line.startswith('## '):
            primary=re.search(r'## (\d+)\.',line)
            if primary:section=primary.group(1)
        if line.startswith('### '):
            match=re.search(r'(\d+\.\d+|S\d\.\d)',line)
            if match:section=match.group(1)
        if re.match(r'^(- |\d+\. )',line):
            controllers=[r['id'] for r in requirements if r['id']==section or r['id'].endswith('.'+section) or (section=='11.4' and r['id'].startswith('11.4'))]
            items.append({'master_line':i,'section':section,'requirement_text':line,'controlling_requirement_ids':controllers,
                'evaluation':'Broad section blocks retain incomplete clauses; point evidence and limits listed above.'})
    result={'workflow_finished':True,'execution_complete':False,'performance_status':'INCONCLUSIVE',
        'release_status':'KEEP_V2_DEFAULT','human_review_status':'NOT_REVIEWED','campaign':str(run),
        'accepted_utc':datetime.now(timezone.utc).isoformat(),'requirements':requirements,'source_requirement_lines':items,
        'master_plan':artifact(master),'budget':Budget(run).snapshot(),'blocked_required_items':[r['id'] for r in requirements if r['status']=='BLOCKED'],
        'new_holdout_gold_read':False,'formal_final_answers':0,'local_git_commit_receipt':'Inspect final task Git commit; no push/upload.',
        'interpretation':'Workflow terminal with bounded blocks, not full mandatory execution or a confirmed quality gain. Taskbook0.2 requires execution_complete=false.',
        'independent_acceptance_audit_required':str(run/'INDEPENDENT_ACCEPTANCE_AUDIT.json')}
    for item in result['requirements']:
        item['evidence']=[artifact(run/'S6_APPLICATION_CACHE_VERIFICATION_final_scope_guard.json')
            if Path(ref['path']).name=='S6_APPLICATION_CACHE_VERIFICATION.json' else ref for ref in item['evidence']]
    application_refs=sorted(run.glob('S6_APPLICATION_REQUESTS*.jsonl'))
    for item in result['requirements']:
        if item['id']=='11.3':
            item['evidence']=[ref for ref in item['evidence'] if Path(ref['path']).name!='S6_APPLICATION_REQUESTS.jsonl']+[artifact(path) for path in application_refs]
    atomic_json(report/'ACCEPTANCE_V3.json',result);atomic_json(run/'ACCEPTANCE_V3.json',result)
    print(canonical_json({k:result[k] for k in ('workflow_finished','execution_complete','performance_status','release_status','blocked_required_items','budget')}));return 0


if __name__=='__main__':raise SystemExit(main())
