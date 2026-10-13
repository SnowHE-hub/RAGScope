# Current generation / training handover

Updated after completed E0 CPU handover, 2026-10-14 local date. This file contains current state only; historical measurements remain in TRAINING_REPORT.md and immutable runtime logs.

## Current resource ownership

E3 is COMPLETE: 1200/1200 real formal rows, three models × 400, zero request errors, 1200 actual token-count parity checks passed, and nine separate warmups. Both 4B exec78438 and 9B exec39846 exited 0.

Own Ollama11435 was explicitly unloaded and stopped. PID14564 is absent, port11435 is closed, gpu.lock was reacquired and released, GPU0 MiB / 0%, and host available RAM2,989,690,880bytes at 2026-10-13T17:21:27Z. GPU_RELEASE_RECEIPT.json records that Windows service terminate returned1 after successful explicit unload; this is separate from the successful formal generation processes.

COMPLETION_RECEIPT.json seals the original manifest, frozen inputs, every record/config/summary file and all1200 row hashes. Global budget is1421 total /1200 core. Root owns full E3 analysis in fresh analysis_complete; preserve the older2B analysis snapshot unchanged.

CPU embedding11436 and retrieval matrix remain active under data_preparation. No project generation or training model is resident. Original workloads remain USER_MANUAL_RESUME; never restart them or their daily jobs.

Production acceptance is implemented and executed: scripts/accept_generation_v2.py --scope e3 returns COMPLETE1200/1200; --scope main returns PENDING because the final main manifest is not frozen. Ten contract tests passed. Integrity failures use FAILED_VALIDATION, evidence-backed resource blocks use BLOCKED. The audit itself makes zero generation calls. Root completed analysis_complete and an independent30-number check; do not overwrite the old2B snapshot.

Next task, assigned by root: implement and freeze the S3.5 4K/8K context-window resource diagnostic, only2B Q4, at most20 dev questions selected by source feasibility, three warmups per condition, global budget. Context must actually approach each true-template window budget, with512 output reserved, canonical source coordinates and no injected answer labels. Do not start GPU until data_preparation has normally finished its11-group matrix plus bounded CPU batch benchmark and explicitly released the resource window. Then run only these two conditions, retain errors, unload and stop own service. Do not add this diagnostic to main quality rankings.

Diagnostic implementation and realCPUfreeze are now done: scripts/run_context_stress_v2.py, ragscope/experiments/context_stress.py, six offline testsPASS. Frozen run 20261014_v2_context_stress selected8/100dev because92 papers are too short for near8K input; all exclusions and a100-question canonical length audit are retained. The16paired input counts are exactly3584/7680, each plus512output. Manifest_hash e87d506a9eab594eac0f088317d094df315c82097bc3eaa3ce0a56cbb04589c9. Do not modify these frozen implementation files before execution. No GPU call yet. Data has completed11/11 and batch benchmark session33240 is active; wait its explicit completion/freshRAM message, then safely archive only own generation.stop after confirming prior service EXITED, start serve_generation.py -Gpu, and run stress --mode run --window4096 then unload/check then window8192. Exact spaced commands are in CONTEXT_STRESS.md. Source-scope engineering guard failure was retained and fixed before successful freeze; it was not a model call.
## E3 actual core experiments

Shard directory D:/RAGScope_Runtime/runs/20261014_v2_generation_e3.
Manifest_hash 8c6a71031d25f0a0218a85a3abbd9d8dc35d3c0f56dafa046e8249beeffb5599.
Exactly1200tasks: original200QASPERtest×oracle/no_context×3Qwen. Same final promptv2.2, model identities, reference2048+all3true-template budget and512output reserve as futuremain E3. No retrieval selection or test-output-based protocol change.

2B COMPLETE400/400 formalrows,0errors; exec81336 exit0,3separatewarmups. Unload confirmed APIpsempty/GPU0MiB in RELEASE_2B_FOR_E0.json. Do not rerun2B.
For182answerable questions, OracleF1=.5785604023 andNoContextF1=.0414930593;18unanswerable separate. This is a source-context diagnostic, not a retrieval-method gain.
Root's independent2B numeric audit10/10PASS is in e3/analysis/ANALYSIS.json, GENERATION_TABLES.md, INDEPENDENT_NUMBERS_2B.json. Preserve this snapshot; later completeE3 analysis must use a new analysis_complete directory.

Current budget after2B:615total/400core; later4Bcalls increment same D:/RAGScope_Runtime/runs/20261014_v2/generation_budget.json. All pilots,warmups,failures,mainmatrix,routerlabels,judge,LoRAeval count. OOM1/consecutivebackend3 circuitbreak, no silent retries/replacements. Actualprompttokens use backend_metadata.actual_input_tokens/token_count_match; RAGResult.prompt_tokens is legacy reference-token statistics.

Main manifest will explicitly reference the immutable shard via provenance.external_campaigns [{manifest:artifact}]. generation_campaign.iter_record_sources yields {directory,manifest,inputs,owned_task_ids}; it validates all task/input/model/generation/prompt/metric/seed identities, forbids nested/cycles/duplicateownership. Main runner skips every external-owned task, including unfinished ones; external missingrows remain missing. Original records/config/manifest hashes are never rewritten. Root analysis and independent reverse auditor support these sources.

## Completed BF16 LoRA training

D:/RAGScope_Runtime/runs/20261014_v2_lora/train_2b_1024_seed42 has TRAIN_LOOP_COMPLETE and PASS_REAL_TRAINING_AUDIT:
122steps/972train/1epoch,59dev,allloss/gradientfinite; actual114languageattentionLinear targets,3,691,008LoRAparameters.
Three independent seed42 initializations have identical adapterSHA c730a4fa702c2a45717a7d810f472fecd2e90a78f4b4f76b7eb4ca93cf5d685f.
Real5→10 and10→122 recovery verifies adapter/optimizer/scheduler/allRNG/order/progress exactly.
Best=lateststep122,devloss.1356631101,3completeownedcheckpoints retained.
FinaladapterSHA a532dc22942d84e4dceabacb64b936c191b37ebdfa9fda092f3f87464d0bbae4.
Exec72907 exit0,trainingprocess absent,gpu.lock reacquired/released,0MiBGPU confirmed in GPU_RELEASE_RECEIPT.json. Training is finished; do not restart it.

Final data: prepared_2b_1024_v2 train972/1000 (841answerable/131unanswerable),dev59/60 (53/6),whole-length rejects28+1,no evidence/answer truncation. MASK_AND_SPLIT_AUDIT.json verifies actualtokenarrays/masks and paperdisjointness.
Earlier unseeded-initialization10steps and preload RAM failures are retained diagnostic history only, not part of the final curve. Training resource figures, exactconfig and allhistorical failures are in TRAINING_REPORT.md. No inference that trainingloss meansQA improvement.

## Pending same-precision comparison and online work

Main E2/R2 finalcontexts remain pending completion of the active retrieval matrix and dev selection. Paired100BF16 evaluation has NOT run. RealCPUcheck-only wrote paired_test100/READINESS.json WAITING_FOR_FROZEN_GENERATION,weights_loaded=false,gpu_used=false,generation_calls0.
When finalcontexts and a newGPUhandover are ready:

```powershell
& D:/RAGScope_Runtime/envs/train/Scripts/python.exe scripts/evaluate_lora_v2.py --training-run D:/RAGScope_Runtime/runs/20261014_v2_lora/train_2b_1024_seed42 --campaign-dir D:/RAGScope_Runtime/runs/20261014_v2_generation --model-dir D:/RAGScope_Runtime/models/Qwen3.5-2B/15852e8c16360a2fea060d615a32b45270f8a8fc
```

--check-only uses corePython and neverloadsmodel/GPU. Actualrun has2GiBpreloadRAMguard,3separatewarmups/condition/session,globalcounting,INCOMPLETEsummary on circuitbreak. BothBF16conditions share protocol/precision; their prompt-only JSON differs from Q4schema-constrained mainmatrix, so no precision-isolated crossprotocolclaim.

Online runner: scripts/run_online_routing_v2.py --campaign <main> --routing-dir <routing> --mode freeze|run. All7policies' decisions freeze beforetestscoreaccess; onlyfull60devselectsbestfixed/nonfixed, same30dev onlinepair IDs. Actualchoose→retrieve→2048pack→sharedallQwencounterrepack→2Bgenerate;3warmups eachsession/policy,alternatingpairedorder,resume/globalbudget. Perpolicyactualgeneration_attempted andpregenerationfailures count separately; fewer30actualrequests preventsdeployment. Full60dev quality gate and30onlineP50 gate reported separately. CPU BGE sharedverifiedloader and2GiBRAMguard. Selectedlearnedrouter must use real_generation_proxy andexacttrain/dev IDs. Noactualonline rows yet;9offlinecontracttestsPASS.

Supportjudge implemented,notrun: scripts/run_support_judge_v2.py --campaign-dir <main> freezes≤200stratifiedE2IDs beforejudging; --run requires completeE2 andpinned4B/9B. Rawprompt/outputs/hashes preserved,failuresunknown,selfmodelbiasflagged,humanreviewNOT_REVIEWED.

Currentsoftware verification:36generation-campaign/training/online testsPASS,pluslatest9onlineguardtestsPASS. Root unifiedcore suite414PASS/2explicit skips. Allunitfixtures stay outside actualexperimentrecords.
