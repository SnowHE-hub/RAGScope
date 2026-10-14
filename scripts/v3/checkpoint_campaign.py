"""Observe live workers and budget; save an accurate, resumable v3 checkpoint."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
import psutil
from ragscope.indexing.identity import atomic_json, canonical_json, signature
from ragscope.v3.budget import Budget


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', required=True)
    args = parser.parse_args()
    run = Path(args.campaign).resolve()
    if run.parent != Path('D:/RAGScope_Runtime/runs').resolve() or not run.name.startswith('v3_'):
        raise ValueError('Wrong campaign')
    status_path = PROJECT / 'docs/execution/v3/STATUS.json'
    status = json.loads(status_path.read_text('utf-8'))
    live = []
    for process in psutil.process_iter(['pid', 'name', 'exe', 'cmdline', 'create_time']):
        try:
            info = process.info
            argv = info['cmdline'] or []
            if info['name'].lower() == 'python.exe' and len(argv) > 1 and argv[1].replace('\\', '/') in {
                    'scripts/v3/run_atomization.py', 'scripts/v3/resume_atomization.py',
                    'scripts/v3/calibrate_gpu_claim_judge.py', 'scripts/v3/score_claim_metrics.py',
                    'scripts/v3/run_context_judge.py', 'scripts/v3/run_protocol_study.py',
                    'scripts/v3/build_task_references.py', 'scripts/v3/run_protocol_supplements.py',
                    'scripts/v3/run_protocol_supplements_v2.py', 'scripts/v3/download_reranker.py',
                    'scripts/v3/calibrate_qwen_reranker.py', 'scripts/v3/prepare_retrieval_sources.py',
                    'scripts/v3/smoke_private_embeddings.py', 'scripts/v3/build_document_dense.py',
                    'scripts/v3/bridge_and_prepare_query_vectors.py', 'scripts/v3/build_chunk_sparse_indices.py',
                    'scripts/v3/build_chunk_dense_index.py', 'scripts/v3/build_chunk_dense_index_resource_v3.py',
                    'scripts/v3/prepare_scoped_vectors.py', 'scripts/v3/run_scoped_generation.py',
                    'scripts/v3/run_scoped_generation_resource.py', 'scripts/v3/calibrate_bge_reranker.py',
                    'scripts/v3/run_lora_runtime_diagnostics.py',
                    'scripts/v3/run_lora_supervision_smoke.py', 'scripts/v3/run_lora_fixed_token_benchmark.py',
                    'scripts/v3/run_lora_diagnostic_sequence.py',
                    'scripts/v3/run_lora_supervision_smoke_v2.py',
                    'scripts/v3/run_workflow_generation.py',
                    str(PROJECT/'scripts/v3/run_workflow_generation.py').replace('\\','/'),
                    str(PROJECT/'scripts/v3/run_lora_runtime_diagnostics.py').replace('\\','/'),
                    str(PROJECT/'scripts/v3/run_lora_supervision_smoke.py').replace('\\','/'),
                    str(PROJECT/'scripts/v3/run_lora_fixed_token_benchmark.py').replace('\\','/'),
                    str(PROJECT/'scripts/v3/prepare_scoped_vectors.py').replace('\\','/'),
                    str(PROJECT/'scripts/v3/run_scoped_generation_resource.py').replace('\\','/'),
                    str(PROJECT/'scripts/v3/build_chunk_dense_index_resource_v3.py').replace('\\','/')} and str(run).replace('\\', '/') in [a.replace('\\', '/') for a in argv]:
                live.append(info)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    progress_path = run / 'ATOMIZATION_PROGRESS.json'
    progress = json.loads(progress_path.read_text('utf-8')) if progress_path.exists() else {}
    budget = Budget(run).snapshot()
    now = datetime.now(timezone.utc)
    status.update(updated_utc=now.isoformat(), updated_local=now.astimezone().isoformat(),
                  active_owned_processes=live,
                  active_model_processes=[p for p in live if not p['cmdline'][1].replace('\\','/').endswith(('prepare_retrieval_sources.py','download_reranker.py'))], budget_snapshot=budget,
                  new_llm_requests=budget['requests_reserved'], atomization_progress=progress,
                  last_local_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=PROJECT, text=True).strip())
    live_atom = any(p['cmdline'][1].replace('\\', '/').endswith('atomization.py') for p in live)
    live_scoring = any(p['cmdline'][1].replace('\\', '/').endswith('score_claim_metrics.py') for p in live)
    quotes_complete = progress.get('recorded_groups') == progress.get('expected_groups') == 558 and not live_atom
    status['tasks'].update({
        'S1.2_actual_atomic_claim_scoring': 'RUNNING_FULL_SCORING' if live_scoring else 'RUNNING_SEGMENTATION' if live_atom else 'SEGMENTATION_COMPLETE_WITH_10_RETAINED_FAILURES' if quotes_complete else 'SEGMENTATION_REQUIRES_TERMINAL_AUDIT',
        'S1.3_qa_style_nli_calibration': 'FAIL_QA_WRAPPER_GATE_RETAINED',
        'S1.3_narrow_qa_completion_calibration': 'PASS_80_CONSTRUCTED_TESTS_NOT_NATURAL_ACCURACY',
    })
    status['completed_evidence'].update(quote_groups_recorded=progress.get('recorded_groups', 0),
        quote_groups_valid=progress.get('valid_groups', 0), quote_groups_failed=progress.get('failures', 0),
        nli_qa_style_tests=72, nli_narrow_completion_tests=80)
    status['tasks']['S1.2_claim_support_citation_metrics'] = 'RUNNING_REAL_FULL_SCORING' if live_scoring else 'IMPLEMENTED_PENDING_REAL_FULL_SCORING'
    score_path = run/'claim_scores_cpu_v3/SUMMARY.json'
    if score_path.exists():
        scores = json.loads(score_path.read_text('utf-8'))
        status['completed_evidence'].update(actual_claim_score_answers=scores['answers'],
            actual_claim_score_relationships=scores['relationships'])
        if scores['answers'] == scores['expected'] == 5720:
            status['tasks']['S1.2_claim_support_citation_metrics'] = 'PASS_FULL_SCORING_WITH_UNCERTAINTY'
    if (run/'NLI_GPU_SCORING_CALIBRATION.json').exists():
        calibration = json.loads((run/'NLI_GPU_SCORING_CALIBRATION.json').read_text('utf-8'))
        status['tasks']['S1.3_actual_gpu_stack_calibration'] = calibration['status']
        status['completed_evidence']['actual_gpu_calibration_tests'] = sum(v['n'] for v in calibration['families'].values())
    if (run/'AUXILIARY_SPLIT_VALIDATION.json').exists():
        auxiliary = json.loads((run/'AUXILIARY_SPLIT_VALIDATION.json').read_text('utf-8'))
        if auxiliary['status'] == 'PASS_INDEPENDENT_SCHEMA_CORRECT_VALIDATION' and auxiliary['primary_overlap_n'] == 0:
            status['tasks']['S0.3_scope40_synthetic_holdout'] = 'PASS_FROZEN_SYNTHETIC_NOT_HUMAN_REVIEWED'
            status['tasks']['S0.3_task_source30_papers'] = 'PASS_SOURCE_INPUTS_REFERENCES_PENDING'
            status['completed_evidence'].update(H_scope=40, H_tasks_source_papers=30)
    if (run/'CONTEXT_JUDGEMENT_INPUTS.json').exists():
        context = json.loads((run/'CONTEXT_JUDGEMENT_INPUTS.json').read_text('utf-8'))
        status['tasks']['S1.4_context_only_sufficiency_inputs'] = 'PASS_INPUTS_ONLY_JUDGE_PENDING'
        status['completed_evidence']['context_only_judge_inputs'] = context['unique_question_context_n']
    if (run/'OLD_OUTPUT_ATTRIBUTION.json').exists():
        status['tasks']['S1.5_model_scale_length_and_lora_latency_decomposition'] = 'PASS_OLD_DESCRIPTIVE_DECOMPOSITION'
    if (run/'POLAR_PREFIX_DIAGNOSTIC.json').exists():
        status['tasks']['S1.5_yes_no_prefix_vs_word_score'] = 'PASS_DIAGNOSTIC_NOT_FACTUAL_ACCURACY'
    if (run/'CLAIM_SCORING_RESOURCE_BLOCK.json').exists():
        blocked = json.loads((run/'CLAIM_SCORING_RESOURCE_BLOCK.json').read_text('utf-8'))
        status['tasks']['S1.2_claim_support_citation_metrics'] = 'BLOCKED_RESOURCE_BOUNDED_ATTEMPTS_EXHAUSTED'
        status['tasks']['S1.3_natural_blind_review'] = 'PARTIAL_12_SCORE_ROWS_FULL_REFERENCE_SUPPORT_BLOCKED'
        status['completed_evidence']['actual_claim_score_answers'] = blocked['actual_cpu_partial_answers']
        status['completed_evidence']['actual_claim_score_relationships'] = blocked['attempts'][-1]['relationships']
        status['stages']['S1'] = 'BLOCKED_REQUIRED_FULL_CLAIM_METRICS_OTHER_DIAGNOSTICS_PASS'
    if (run/'CONTEXT_DIAGNOSIS.json').exists():
        diagnosis = json.loads((run/'CONTEXT_DIAGNOSIS.json').read_text('utf-8'))
        status['tasks']['S1.4_context_only_sufficiency_inputs'] = 'PASS_117_ACTUAL_CONTEXT_PROXY_JUDGEMENTS'
        status['tasks']['S1.5_new_refusal_context_assessment'] = 'PASS_23_INDEPENDENT_PROXY_VERDICTS_NOT_HUMAN'
        status['completed_evidence']['lora_refusal_context_counts'] = diagnosis['new_refusal_context_counts']
    if (run/'S2_PROTOCOL_ANALYSIS.json').exists():
        protocol = json.loads((run/'S2_PROTOCOL_ANALYSIS.json').read_text('utf-8'))
        status['completed_evidence'].update(S2_main_matrix_records=protocol['matrix_n'], S2_G0_reused=144,
            S2_actual_smoke_calls=72, S2_new_main_requests=720)
        status['tasks']['S2.1_actual_direct_pipeline_smoke'] = 'PASS_72_CALLS_36_EQUAL_PAIRS_NO_TOKEN_OR_CHANNEL_MISMATCH'
        status['tasks']['S2.2_G0_G3_frozen24_matrix'] = 'PASS_864_RECORDS_OLD_EXPOSED_DIAGNOSTIC_ONLY'
        status['tasks']['S2.2_oracle_padded36'] = 'PENDING'
        status['tasks']['S2.2_prior_extra6'] = 'PENDING_6_PRIOR_CALLS_DONE_6_ADDITIONAL_PLANNED'
        status['stages']['S2'] = 'RUNNING_REQUIRED_SUPPLEMENT_PENDING'
        status['current_stage'] = 'S2_AND_S0_AUXILIARY_REFERENCES'
    if (run/'H_TASKS_CONSTRUCTION.json').exists():
        construction = json.loads((run/'H_TASKS_CONSTRUCTION.json').read_text('utf-8'))
        status['tasks']['S0.3_scope_tasks_synthetic_holdout'] = 'PASS_SYNTHETIC_SOURCE_CONSTRUCTION_WITH_RETAINED_TEACHER_FAILURES'
        status['completed_evidence']['H_tasks_valid_tasks'] = construction['valid_task_n']
        status['completed_evidence']['H_tasks_teacher_failed_or_blocked_papers'] = construction['failed_or_blocked_papers']
    if (run/'H_TASKS_SOURCE_REPAIR.json').exists():
        repaired = json.loads((run/'H_TASKS_SOURCE_REPAIR.json').read_text('utf-8'))
        status['completed_evidence']['H_tasks_valid_tasks'] = repaired['valid_task_n']
        status['completed_evidence']['H_tasks_invalid_task_branches_retained'] = repaired['failed_task_n']
        status['tasks']['S0.3_scope_tasks_synthetic_holdout'] = 'PASS_40_SCOPE_20_SOURCE_VERIFIED_SYNTHETIC_TASKS_FAILURES_RETAINED'
        status['stages']['S0'] = 'PASS_WITH_SYNTHETIC_SOURCE_FAILURES_RETAINED'
    status['next_action'] = ('H_tasks teacher session71248 actually exited0; no model job remains. '
        '20 source-verified synthetic task references frozen in questions_v2/gold_v2, 29 invalid branches preserved. Finish S2 required Oracle padded36 '
        'and six extra prior-knowledge probes, then S3 retrieval funnel and core-first packing. No more old NLI reloads.')
    status['next_command'] = (
        ". ./scripts/runtime_env.ps1\n"
        "& 'D:/RAGScope_Runtime/envs/core/Scripts/python.exe' scripts/v3/checkpoint_campaign.py --campaign '" + str(run).replace('\\', '/') + "'")
    status['next_command_status'] = 'READ_ONLY_OBSERVATION_AND_CHECKPOINT; no model call'
    status['next_tool_call'] = {'tool': 'exec_command', 'args': {'cmd':
        "Get-Content -LiteralPath 'D:/RAGScope_Runtime/runs/v3_20261015_014506/S2_PROTOCOL_INPUTS.jsonl' -TotalCount 1",
        'max_output_tokens': 2200}}
    if (run/'S2_SUPPLEMENT_ANALYSIS.json').exists():
        supplement=json.loads((run/'S2_SUPPLEMENT_ANALYSIS.json').read_text('utf-8'))
        if supplement['status']=='PASS_INDEPENDENT_SUPPLEMENT_AUDIT':
            status['completed_evidence'].update(S2_padding_calls=36,S2_prior_total_calls=12,S2_total_new_calls=840)
            status['tasks'].update({'S2.2_oracle_padded36':'PASS_36_REAL_CALLS_MATCHED_SOURCE_EVIDENCE',
                'S2.2_prior_extra6':'PASS_12_TOTAL_PRIOR_PROBES',
                'S0.3_task_source30_papers':'PASS_20_VALID_SYNTHETIC_TASKS_29_FAILED_BRANCHES_RETAINED'})
            status['stages']['S2']='DIAGNOSTIC_COMPLETE_D_CONFIRM_GENERATOR_STABILITY_PENDING'
    if (run/'S3_OLD_RETRIEVAL_FUNNEL.json').exists():
        funnel=json.loads((run/'S3_OLD_RETRIEVAL_FUNNEL.json').read_text('utf-8'))
        status['tasks']['S3.2_old_retrieval_funnel']='PASS_500_OLD_ROWS_INDEPENDENT_SPAN_RECOUNT'
        status['tasks']['S3.5_old_window_audit']='PASS_9519_WINDOWS_NO_BUDGET_OR_CHAR_COVERAGE_VIOLATION'
        status['tasks']['S3.5_top2_vs_max_old']='PASS_REAL_SCORE_REUSE_ZERO_SELECTED_COVERAGE_GAIN'
        status['completed_evidence'].update(S3_old_funnel_rows=sum(g['record_n'] for g in funnel['groups'].values()),
            S3_old_window_count=funnel['window_checks']['window_n'])
        status['stages']['S3']='RUNNING_NEW_RETRIEVAL_REQUIRED'
        status['current_stage']='S3_RETRIEVAL_AND_S2_D_CONFIRM'
        status['next_action']='Finish any verified-live reranker calibration, then build the three source-only metadata indices and D_confirm retrieval/generation validation. Do not repeat old NLI or completed S2 calls.'
        status['next_tool_call']={'tool':'exec_command','args':{'cmd':
            "Get-Content -LiteralPath 'D:/RAGScope_Runtime/runs/v3_20261015_014506/S3_QWEN_RERANKER_CALIBRATION.log' -Tail 5",'max_output_tokens':1000}}
    if (run/'S3_CORE_FIRST_OLD_SCREEN.json').exists():
        status['tasks']['S3.6_core_first_old_screen']='PASS_400_MATCHED_PACKING_ROWS_MIXED_RESULTS_NOT_PROMOTED'
        status['completed_evidence']['S3_core_first_old_rows']=400
    if (run/'S3_DOCUMENT_BM25_INDEX.json').exists():
        document_index=json.loads((run/'S3_DOCUMENT_BM25_INDEX.json').read_text('utf-8'))
        status['tasks']['S3.4_document_bm25_index']=document_index['status']
        status['completed_evidence']['S3_document_index_papers']=document_index['paper_n']
    if (run/'S3_DOCUMENT_CANDIDATES_D_SEARCH.json').exists():
        document_audit=json.loads((run/'S3_DOCUMENT_CANDIDATES_D_SEARCH.json').read_text('utf-8'))
        status['tasks']['S3.1_scope_document_candidate_diagnostic']='PASS_120_EXPOSED_QUESTIONS_240_REAL_LOOKUPS_INPUT_CONDITIONS_SEPARATE'
        status['completed_evidence']['S3_document_candidate_questions']=document_audit['questions_n']
    if (run/'S3_QWEN_RERANKER_CALIBRATION.json').exists():
        calibration=json.loads((run/'S3_QWEN_RERANKER_CALIBRATION.json').read_text('utf-8'))
        status['tasks']['S3.5_qwen_reranker_12_fixtures']=calibration['status']
        status['completed_evidence']['S3_reranker_fixture_n']=calibration['record_n']
        status['completed_evidence']['S3_reranker_fixture_correct_n']=calibration['correct_n']
    live_source=any(p['cmdline'][1].replace('\\','/').endswith('prepare_retrieval_sources.py') for p in live)
    if live_source:
        status['tasks']['S3.3_three_chunker_source_metadata']='RUNNING_CPU_ONLY_SOURCE_PREPARATION'
        status['next_action']='Observe the live source preparation job (tool session85266; verify PID/exe/argv). Do not start another builder. After terminal exit, verify all per-document artifacts, then build actual BM25/dense indices and D_confirm validation.'
        status['next_tool_call']={'tool':'write_stdin','args':{'session_id':85266,'chars':'','yield_time_ms':1000,'max_output_tokens':1200}}
    if (run/'S3_RETRIEVAL_SOURCE_PREPARATION.json').exists():
        source_prep=json.loads((run/'S3_RETRIEVAL_SOURCE_PREPARATION.json').read_text('utf-8'))
        status['tasks']['S3.3_three_chunker_source_metadata']=source_prep['status']
        status['completed_evidence']['S3_source_document_groups']=source_prep['completed_document_groups']
        status['completed_evidence']['S3_source_groups']=source_prep['groups']
        status['next_action']='Three source-only chunk preparations completed; build real document/chunk BM25 and dense indices, compare bounded candidates on D_search/D_confirm, and verify generator stability. Do not rerun completed model studies.'
        status['next_tool_call']={'tool':'exec_command','args':{'cmd':
            "Get-Content -LiteralPath 'D:/RAGScope_Runtime/runs/v3_20261015_014506/S3_RETRIEVAL_SOURCE_PREPARATION.json' -Raw",'max_output_tokens':1600}}
    if (run/'S3_DOCUMENT_DENSE_INDEX.json').exists():
        dense=json.loads((run/'S3_DOCUMENT_DENSE_INDEX.json').read_text('utf-8'))
        status['tasks']['S3.4_document_dense_index']=dense['status']
        status['completed_evidence']['S3_real_document_vectors']=dense['paper_n']
    if (run/'S3_DEV_QUERY_VECTORS.json').exists():
        queries=json.loads((run/'S3_DEV_QUERY_VECTORS.json').read_text('utf-8'))
        status['tasks']['S3.4_dev_query_embeddings']=queries['status']
        status['completed_evidence']['S3_dev_query_vectors']=queries['question_n']
    if (run/'S3_EMBEDDING_BACKEND_BRIDGE.json').exists():
        bridge=json.loads((run/'S3_EMBEDDING_BACKEND_BRIDGE.json').read_text('utf-8'))
        status['tasks']['S3.4_embedding_backend_bridge']=bridge['status']
        status['completed_evidence']['S3_embedding_bridge_min_cosine']=min(c['cosine'] for c in bridge['cases'])
    if (run/'S3_CHUNK_SPARSE_INDICES.json').exists():
        status['tasks']['S3.3_four_sparse_indices']='PASS_REAL_SPARSE_INDICES'
    if (run/'S3_SPARSE_CANDIDATE_SCREEN.json').exists():
        screen=json.loads((run/'S3_SPARSE_CANDIDATE_SCREEN.json').read_text('utf-8'))
        status['tasks']['S3.3_sparse_candidate_screen']=screen['status']
        status['completed_evidence']['S3_sparse_screen_rows']=sum(x['n'] for x in screen['groups'].values())
        status['completed_evidence']['S3_selected_dense_variants']=screen['next_dense_variants']
    live_chunk_dense=any(p['cmdline'][1].replace('\\','/').endswith(('build_chunk_dense_index.py','build_chunk_dense_index_resource_v3.py')) for p in live)
    if live_chunk_dense:
        status['tasks']['S3.3_selected_chunk_dense_indices']='RUNNING_SERIAL_GPU_EMBEDDING'
        status['next_action']='Observe the verified-live serial dense index job; R512 body first, metadata second. Do not start another GPU model or rerun completed embeddings. After terminal exit, validate matrices and run bounded hybrid/rerank/core packing development tests.'
        handle_path=run/'CURRENT_LIVE_DENSE_JOB.json'
        handle=json.loads(handle_path.read_text('utf-8')) if handle_path.exists() else {}
        status['next_tool_call']={'tool':'write_stdin','args':{'session_id':handle.get('session_id',81708),'chars':'','yield_time_ms':1000,'max_output_tokens':1200}}
    complete_dense=[]
    for variant in ('R512_body','R512_metadata'):
        path=run/('S3_CHUNK_DENSE_'+variant+'.json')
        if path.exists():complete_dense.append(variant)
    if complete_dense:
        status['completed_evidence']['S3_complete_chunk_dense_variants']=complete_dense
    if len(complete_dense)==2 and not live_chunk_dense:
        status['tasks']['S3.3_selected_chunk_dense_indices']='PASS_TWO_REAL_DENSE_INDICES'
        status['next_action']='Validate completed canonical dense/sparse indices, then execute bounded hybrid/rerank/core development comparisons and real D_confirm generation stability. New H_QA gold remains sealed.'
    if (run/'S3_CHUNK_DENSE_FINAL_RESOURCE_BLOCK.json').exists() and not live_chunk_dense:
        status['tasks']['S3.3_selected_chunk_dense_indices']='BLOCKED_RESOURCE_BOUNDED_ATTEMPTS_EXHAUSTED'
        status['next_action']='Dense compatibility attempt is terminal BLOCKED. Preserve all source/cache results; execute independent mandatory BGE/LoRA runtime/workflow diagnostics and final safe alternatives without reloading the blocked operation.'
    if (run/'S3_DOCUMENT_DENSE_CANDIDATES.json').exists():
        status['tasks']['S3.4_document_dense_fusion_diagnostic']='PASS_220_UNRESOLVED_SCOPE_DIAGNOSTICS_NOT_TASK_QUALITY'
    if (run/'S5_SUPERVISION_DISTRIBUTION_DIAGNOSIS.json').exists():
        lora=json.loads((run/'S5_SUPERVISION_DISTRIBUTION_DIAGNOSIS.json').read_text('utf-8'))
        status['tasks']['S5.1_full_training_numeric_and_decode20']=lora['status']
        status['completed_evidence']['S5_train_supervision_checked']=lora['train_n']
        status['completed_evidence']['S5_actual_decoded_targets']=len(lora['decoded_cases'])
        status['stages']['S5']='SOURCE_DIAGNOSIS_DONE_MANDATORY_RUNTIME_MICRO_PENDING'
    if (run/'S3_DENSE_BLOCK_CLOSURE.json').exists():
        dense_block=json.loads((run/'S3_DENSE_BLOCK_CLOSURE.json').read_text('utf-8'))
        status['tasks']['S3.3_full_corpus_dense']='BLOCKED_BOUNDED_ATTEMPTS_EXHAUSTED'
        status['completed_evidence']['S3_partial_dense_vector_rows']=dense_block['v3_verified_vector_rows']
    if (run/'S3_SCOPED_VECTOR_COMPLETION.json').exists():
        compact=json.loads((run/'S3_SCOPED_VECTOR_COMPLETION.json').read_text('utf-8'))
        status['tasks']['S3.4_compact_paper_vectors']=compact['status']
        status['completed_evidence']['S3_compact_papers']=compact['paper_n']
        status['completed_evidence']['S3_compact_vector_chunks']=compact['chunk_n']
    if (run/'S3_SCOPED_CONTEXT_MANIFEST.json').exists():
        contexts=json.loads((run/'S3_SCOPED_CONTEXT_MANIFEST.json').read_text('utf-8'))
        status['tasks']['S3.6_scoped_context_freeze']=contexts['status']
        status['completed_evidence']['S3_scoped_maximum_prompt_tokens']=contexts['maximum_actual_prompt_tokens']
    for split in ('D_search','D_confirm'):
        completion=run/('S3_SCOPED_GENERATION_'+split+'_COMPLETION.json')
        if completion.exists():
            generated=json.loads(completion.read_text('utf-8'))
            status['tasks']['S3.6_scoped_generation_'+split]=generated['status']
            status['completed_evidence']['S3_generation_'+split]=generated['formal_records']
    if (run/'S3_SCOPED_GENERATION_D_confirm_COMPLETION.json').exists():
        status['tasks']['S2.3_D_confirm_generator_stability']='ACTUAL_200_GENERATIONS_COMPLETE_INDEPENDENT_AUDIT_PENDING'
    if (run/'S3_SCOPED_GENERATION_D_confirm_ANALYSIS.json').exists():
        independent=json.loads((run/'S3_SCOPED_GENERATION_D_confirm_ANALYSIS.json').read_text('utf-8'))
        if independent['status']=='PASS_INDEPENDENT_PAIRED_DIAGNOSTIC':
            status['tasks']['S2.3_D_confirm_generator_stability']='PASS_INDEPENDENT_PAIRED_100_QUESTION_AUDIT'
            status['stages']['S2']='DIAGNOSTIC_COMPLETE_D_CONFIRM_AUDITED'
    if (run/'S3_BGE_RELEVANCE_CALIBRATION.json').exists():
        bge=json.loads((run/'S3_BGE_RELEVANCE_CALIBRATION.json').read_text('utf-8'))
        status['tasks']['S3.5_bge_12_fixtures']=bge['status']
        status['completed_evidence']['S3_bge_fixture_correct_n']=bge['correct_max_n']
    if (run/'S4_DEVELOPMENT_TASK_LABELS.json').exists():
        workflow_labels=json.loads((run/'S4_DEVELOPMENT_TASK_LABELS.json').read_text('utf-8'))
        status['tasks']['S4.1_visible_task_labels']='PASS_RULE_PROXY_LABELS_FROZEN_NOT_HUMAN'
        status['completed_evidence']['S4_labeled_development_questions']=workflow_labels['question_n']
        status['completed_evidence']['S4_workflow_selected_questions']=workflow_labels['selected_n']
    live_scoped_generation=any(p['cmdline'][1].replace('\\','/').endswith(('run_scoped_generation.py','run_scoped_generation_resource.py')) for p in live)
    if live_scoped_generation:
        status['current_stage']='S3_COMPACT_ALTERNATIVE_REAL_GENERATION'
        status['next_action']='Observe live D_confirm fixed20-record resource recovery. Original77 successful responses retained; do not reissue. Full corpus Dense remains BLOCKED. After terminal, independently audit and execute BGE/LoRA runtime diagnostics.'
        status['next_tool_call']={'tool':'write_stdin','args':{'session_id':45993,'chars':'','yield_time_ms':1000,'max_output_tokens':1000}}
    live_lora_runtime=any(p['cmdline'][1].replace('\\','/').endswith(('run_lora_runtime_diagnostics.py',
        'run_lora_supervision_smoke.py','run_lora_supervision_smoke_v2.py',
        'run_lora_fixed_token_benchmark.py','run_lora_diagnostic_sequence.py')) for p in live)
    for condition in ('base','unmerged','merged'):
        completion=run/('S5_RUNTIME_'+condition+'_COMPLETION.json')
        if completion.exists():
            runtime_result=json.loads(completion.read_text('utf-8'))
            status['tasks']['S5.2_runtime_'+condition]=runtime_result['status']
            status['completed_evidence']['S5_runtime_'+condition+'_formal_n']=runtime_result['formal_n']
    if live_lora_runtime:
        status['current_stage']='S5_REAL_BF16_RUNTIME_DIAGNOSTICS'
        status['tasks']['S5.2_same_BF16_24_runtime']='RUNNING'
        status['next_action']='Observe verified-live S5 BF16 runtime job; do not start another GPU model. Record actual EOS/output IDs and terminal result, then run pending adapter conditions and mandatory supervision smoke.'
        handle_path=run/'CURRENT_LORA_RUNTIME_JOB.json'
        handle=json.loads(handle_path.read_text('utf-8')) if handle_path.exists() else {}
        status['next_tool_call']={'tool':'write_stdin','args':{'session_id':handle.get('session_id',62708),'chars':'','yield_time_ms':1000,'max_output_tokens':1000}}
    if (run/'S5_RUNTIME_INDEPENDENT_ANALYSIS.json').exists():
        independent=json.loads((run/'S5_RUNTIME_INDEPENDENT_ANALYSIS.json').read_text('utf-8'))
        status['tasks']['S5.2_runtime_independent_recount']=independent['status']
    if (run/'S5_MERGE_LOGIT_GATE.json').exists():
        merge=json.loads((run/'S5_MERGE_LOGIT_GATE.json').read_text('utf-8'))
        status['tasks']['S5.2_merge_function_gate']='PASS' if merge['passed'] else 'FAILED_NUMERICAL_GATE_MERGED_COMPARISON_NOT_RUN'
    smoke_path=run/'S5_SUPERVISION_SMOKE/COMPLETION.json'
    if smoke_path.exists():
        smoke=json.loads(smoke_path.read_text('utf-8'))
        status['tasks']['S5.1_eight_fixture_train_update_resume']=smoke['status']
        status['completed_evidence']['S5_smoke_optimizer_steps']=smoke['optimizer_steps']
    compatible_smoke=run/'S5_SUPERVISION_SMOKE_v2/COMPLETION.json'
    if compatible_smoke.exists():
        smoke=json.loads(compatible_smoke.read_text('utf-8'))
        status['tasks']['S5.1_initial_one_step_smoke']='FAILED_NO_UPDATE_DUE_ZERO_LR_WARMUP_RETAINED'
        status['tasks']['S5.1_eight_fixture_train_update_resume']=smoke['status']
        status['completed_evidence']['S5_smoke_optimizer_steps']=smoke['optimizer_steps']
        status['completed_evidence']['S5_smoke_changed_tensor_n']=smoke['changed_tensor_n']
    if (run/'S4_WORKFLOW_CONTEXT_MANIFEST.json').exists():
        workflow=json.loads((run/'S4_WORKFLOW_CONTEXT_MANIFEST.json').read_text('utf-8'))
        status['tasks']['S4.2_real_workflow_contexts']=workflow['status']
        status['completed_evidence']['S4_workflow_context_inputs']=workflow['formal_inputs_n']
    for condition in ('base','unmerged','merged'):
        completed=run/('S5_FIXED64_'+condition+'_COMPLETION.json')
        if completed.exists():
            fixed=json.loads(completed.read_text('utf-8'))
            status['tasks']['S5.2_fixed_token_'+condition]=fixed['status']
    if (run/'S5_SEQUENCE_COMPLETION.json').exists() and not live_lora_runtime:
        sequence=json.loads((run/'S5_SEQUENCE_COMPLETION.json').read_text('utf-8'))
        status['stages']['S5']=sequence['status']
        status['tasks']['S5.2_serial_diagnostic_sequence']=sequence['status']
        status['current_stage']='S4_WORKFLOW_UTILITY_PENDING'
        status['next_action']='S5 serial sequence is terminal; inspect actual exits and independent diagnostics, preserve failed merge gate. Run the frozen180 S4 workflow generations, then compare full utility vectors before any router training.'
        status['next_command']=". ./scripts/runtime_env.ps1\n& 'D:/RAGScope_Runtime/envs/core/Scripts/python.exe' scripts/v3/run_workflow_generation.py --campaign '"+str(run).replace('\\','/')+"' --resume"
        status['next_command_status']='ACTUAL_FROZEN_S4_WORKFLOW_GENERATION_AFTER_VERIFIED_GPU_RELEASE'
    if (run/'S5_LORA_REPORT_SOURCE_NOTES.json').exists():
        status['tasks']['S5.3_formal_retraining']='NOT_RUN_GATE'
        status['tasks']['S5.5_deployment_conversion']='NOT_RUN_GATE_NO_QUALIFIED_NEW_ADAPTER'
    if (run/'S4_WORKFLOW_GENERATION_COMPLETION.json').exists():
        workflow=json.loads((run/'S4_WORKFLOW_GENERATION_COMPLETION.json').read_text('utf-8'))
        status['tasks']['S4.3_actual_workflows']=workflow['status']
        status['completed_evidence']['S4_actual_workflow_generations']=workflow['formal_n']
    if (run/'S4_WORKFLOW_UTILITY_ANALYSIS.json').exists():
        utility=json.loads((run/'S4_WORKFLOW_UTILITY_ANALYSIS.json').read_text('utf-8'))
        status['tasks']['S4.3_full_utility_vectors']=utility['status']
        status['tasks']['S4.4_new_router_training']=utility['router_training_gate']
        status['stages']['S4']='MATCHED60_DIAGNOSTIC_DONE_TASK_KEYPOINT_COVERAGE_PENDING'
        status['current_stage']='S3_REMAINING_NATURAL_RERANK_AND_S4_KEYPOINT_COVERAGE'
        status['next_action']='Do not rerun completed180 workflow answers or train a new router. Finish natural reranker comparison and source-verified comparison/summary development keypoints, then freeze admissible conditions for S6; H_QA gold remains sealed.'
        status['next_command']="Get-Content -LiteralPath 'scripts/v3/calibrate_qwen_reranker.py' -Raw"
        status['next_command_status']='READ_EXISTING_PINNED_RERANKER_SOURCE_BEFORE_NATURAL_DEVELOPMENT_RUN'
        status['next_tool_call']={'tool':'mcp__codegraph__codegraph_context','args':{
            'task':'Natural reranker development evaluation reusing calibrated Qwen3Reranker and BGE window scorers with compact scoped candidate inputs and rerank pair budget',
            'maxNodes':8,'includeCode':True}}
    live_workflows=any(p['cmdline'][1].replace('\\','/').endswith('run_workflow_generation.py') for p in live)
    if live_workflows:
        status['current_stage']='S4_ACTUAL_MATCHED_WORKFLOW_GENERATION'
        status['tasks']['S4.3_actual_workflows']='RUNNING_SERIAL_FRESH20_BLOCKS'
        status['next_action']='Observe the verified-live S4 matched60/three-workflow actual generation. Preserve failures; audit full per-question utility vectors after terminal completion before any router training.'
        status['next_command']=". ./scripts/runtime_env.ps1\n& 'D:/RAGScope_Runtime/envs/core/Scripts/python.exe' scripts/v3/checkpoint_campaign.py --campaign '"+str(run).replace('\\','/')+"'"
        status['next_command_status']='OBSERVE_RUNNING_S4_DO_NOT_RESTART'
        status['next_tool_call']={'tool':'write_stdin','args':{'session_id':27341,'chars':'','yield_time_ms':1000,'max_output_tokens':1000}}
    status['unresolved']=['S1 full old claim/support/citation metrics BLOCKED after bounded resource attempts',
        'S3 natural new reranker and hierarchical production path; full corpus chunk Dense BLOCKED',
        'S4 comparison/summary independent keypoint coverage and sparse task classes; new router NOT_RUN_GATE',
        'S5 merged function numerical gate failed; formal retraining and conversion NOT_RUN_GATE',
        'S6 candidate lock, once-only holdout, online180, UI, cases, final reports and acceptance pending']
    status.pop('resume_command_after_verified_exit', None)
    observation = {'observed_utc': now.isoformat(), 'live_processes': live, 'budget': budget,
                   'atomization': progress, 'process_observation_not_terminal_exit_receipt': True}
    observation['observation_hash'] = signature(observation)
    with (run / 'CHECKPOINT_OBSERVATIONS.jsonl').open('a', encoding='utf-8', newline='\n') as handle:
        handle.write(canonical_json(observation) + '\n'); handle.flush(); os.fsync(handle.fileno())
    for destination in (status_path, run / 'STATUS.json'):
        atomic_json(destination, status)
    recovery = ('# v3 当前恢复点\n\n' + now.isoformat() + '\n\n'
        + 'Campaign `' + str(run) + '`；原训练 USER_MANUAL_RESUME，不自动恢复。\n\n'
        + '当前实际预算：' + str(budget['requests_reserved']) + ' 次请求；输出已扣除/保留 '
        + str(budget['output_tokens_charged_or_reserved']) + ' tokens；以 BUDGET.jsonl 为准。\n\n'
        + '已核验运行进程：`' + canonical_json(live) + '`。句柄未知时以 PID、创建时间、exe、argv 联合核验，不能因观察超时重启。\n\n'
        + '旧 V2 分割保存 83 组（含 3 个失败）；V3 实际 187 请求、183 成功、4 契约失败，另 371 组资源预检阻断而未发请求。V4 只续跑未发请求的组，所有原记录保留。\n\n'
        + 'NLI QA 校准：完整句 24/24，短片段 6/24，问题答案包装 12/24；后两者不得承担自动事实门槛，人工 NOT_REVIEWED。\n\n'
        + '参考答案不可见的窄 QA 完句规则实际 80/80 构造校准通过；其余语法/短片段仍 UNKNOWN，不宣称自然语料准确率。严禁与分割同时加载 CPU 裁判，以免再次耗尽内存余量。\n\n'
        + '下一条准确命令：\n```powershell\n' + status['next_command'] + '\n```\n\n'
        + 'V4实际终端session57682已exit2：558真实请求齐全，548有效、10契约失败保留；进程已不存在，私有服务卸载/退出。不得重跑分割。\n\n'
        + '实际GPU裁判210构造检查全部通过，session46755 exit0。评分12条冒烟因均UNKNOWN没有关系调用；原空缓存汇总bug session29339 exit1，已保存源与日志，修复后复用12条且actualexit0。\n\n'
        + 'GPU全量session80992/66941均actualexit1，分别在可用内存3153125376/3206303744字节触发3GiB守卫；其部分结果保留。流式修复后仍不足，GPU全量停止有界重试。\n\n'
        + 'CPU兼容session24620也actualexit1；335实际评分/230关系判断保留。全量旧NLI已到有界阻断，不再加载。所有5720条覆盖/UNKNOWN及盲评120条部分包保存，未执行行不冒充模型结果。\n\n'
        + '40范围题已封存，30任务来源论文已冻结且与主分割241篇零交叉；20有效H_tasks参考/29失败任务分支保留，全部人造AI标签。117个独立问题/上下文输入已冻结。\n\n'
        + '9B上下文独立校准59/60。session21014在104/117后资源exit1，6771只续跑13条actualexit0；117条完成。23新增拒答：YES8/NO12/UNKNOWN3，均为proxy，不是人评。\n\n'
        + 'S2输入冻结24题×2上下文，G0真实144复用。72新direct/pipeline调用session65647 exit0，36配对完全一致、无token/思考通道差异。主矩阵session84289 exit0，864完整记录（新720）独立计分验证；含6prior探针。\n\n'
        + 'H_tasks整篇来源session71248 actualexit0，30请求全部完成；初版16有效任务/20失败论文。一次零生成成本的空白格式修复和独立任务验证后，20有效任务、29失败分支保留；H_tasks.questions_v2/gold_v2为当前辅助人造任务集，不是官方QASPER或操作者盲测。无后台模型进程。\n\n'
        + '当前下一项：'+status['next_action']+'\n\n'
        + 'S2_SUPPLEMENT_RESULTS_verified.jsonl是42条补充结果的当前校验投影。v2封装首行遗留旧record_hash参与新hash的缺陷有确证，原两份journal保留，逐个raw请求/输出完全相同；不得重发调用。\n\n'
        + 'S3旧500记录漏斗与9519窗口核验、400同预算core-first对照已落盘；仅旧诊断，不能当成新D_confirm或holdout。Goal active，execution_complete仍false。\n')
    for destination in (PROJECT / 'docs/execution/v3/RECOVERY.md', run / 'RECOVERY.md'):
        destination.write_text(recovery, encoding='utf-8')
    print(canonical_json(observation))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
