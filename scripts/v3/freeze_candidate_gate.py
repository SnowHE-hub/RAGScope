"""Freeze admission outcome before any H_QA gold access; negative gates are terminal evidence."""
import argparse
import json
from pathlib import Path
import sys
PROJECT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(PROJECT))
from ragscope.indexing.identity import artifact,atomic_json,canonical_json,signature


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign',required=True);a=p.parse_args();run=Path(a.campaign).resolve()
    if run.parent!=Path('D:/RAGScope_Runtime/runs').resolve():raise ValueError('Wrong campaign')
    target=run/'CANDIDATE_LOCK.json'
    source_names=['S3_D_CONFIRM_CLAIM_GATE.json','S6_ONLINE_COST_ANALYSIS.json','S4_WORKFLOW_UTILITY_ANALYSIS.json','S5_LORA_REPORT_SOURCE_NOTES.json']
    sources={name:artifact(run/name) for name in source_names}
    if target.exists():
        lock=json.loads(target.read_text('utf8'))
        if lock['admission_evidence']!=sources:raise ValueError('Locked candidate evidence changed')
        print(canonical_json({'status':'REUSED_CANDIDATE_LOCK','admitted_v3_n':len(lock['admitted_v3_candidates'])}));return 0
    quality=json.loads((run/source_names[0]).read_text('utf8'));online=json.loads((run/source_names[1]).read_text('utf8'))
    if online['status']!='PASS_180_INDEPENDENT_ONLINE_AUDIT':raise ValueError('Full actual online screen required before terminal admission decision')
    b,c=[online['groups'][name] for name in ['B0_policy_neighbor_bridge','C1_policy_core_first_bridge']]
    ratio50=c['online_p50_seconds']/b['online_p50_seconds'];ratio95=c['online_p95_seconds']/b['online_p95_seconds']
    quality_pass=quality['quality_candidate_admitted']
    efficiency_pass=online['quality_noninferiority']=='PROVED' and ratio50<=.85 and ratio95<=1
    if quality_pass or efficiency_pass:
        raise RuntimeError('Positive admission requires full guard validation and final runner; do not silently skip required holdout')
    frozen=json.loads((run/'S3_SCOPED_CONTEXT_MANIFEST.json').read_text('utf8'))
    split=json.loads((run/'SPLITS_V3.json').read_text('utf8'))
    lock={'status':'NO_V3_CANDIDATE_ADMITTED','splits_manifest_hash':split['manifest_hash'],
        'admission_evidence':sources,'admitted_v3_candidates':[],
        'baseline':{'condition':'B0_policy_neighbor_bridge','policy':'retained2B Q4/R512/hybrid/neighbor/G0',
            'generator_spec':frozen['generator_spec'],'baseline_qualification':frozen['baseline_qualification']},
        'declined_candidate':{'condition':'C1_policy_core_first_bridge','quality_gain_proved':False,
            'primary_claim_f1_delta':quality['claim_f1_delta_mean'],'online_p50_ratio':ratio50,'online_p95_ratio':ratio95},
        'data_and_indices':{name:artifact(run/name) for name in ['SPLITS_V3.json','S3_SCOPED_CONTEXT_MANIFEST.json','S3_DOCUMENT_DENSE_INDEX.json','S3_DOCUMENT_BM25_INDEX.json','S3_CHUNK_SPARSE_INDICES.json','S3_SCOPED_VECTOR_COMPLETION.json']},
        'metric_versions':{name:artifact(PROJECT/name) for name in ['ragscope/v3/claim_metrics.py','ragscope/v3/audit.py','scripts/v3/audit_online_cost.py']},
        'thresholds':{'development_claim_gain':.02,'efficiency_quality_ci_lower':-.01,'efficiency_p50_ratio':.85,'efficiency_p95_ratio':1},
        'H_QA_status':'BLOCKED_REQUIRED_PAIRED_FINAL_COMPARISON_NO_ADMITTED_CANDIDATE',
        'H_tasks_status':'BLOCKED_REQUIRED_FINAL_COMPARISON_NO_ADMITTED_TASK_WORKFLOW',
        'final_evaluation_authorization_created':False,'holdout_gold_read':False,
        'human_review_status':'NOT_REVIEWED','release_status':'KEEP_V2_DEFAULT','execution_complete':False,
        'reason':'Master10.1 forbids failed/unknown development candidates from entering final comparison. Required unseen comparison remains BLOCKED, not completed or silently reduced to baseline only.'}
    atomic_json(target,lock);print(canonical_json({k:lock[k] for k in ('status','declined_candidate','H_QA_status','H_tasks_status','release_status')}));return 0


if __name__=='__main__':raise SystemExit(main())
