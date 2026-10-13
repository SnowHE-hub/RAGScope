# Current generation / training handover

Updated after completed E0 CPU handover, 2026-10-14 local date. This file contains current state only; historical measurements remain in TRAINING_REPORT.md and immutable runtime logs.

## Active resource ownership

Owned Ollama11435 lifecycle is exec54636, childPID14564/create1791909897.4709003. It is READY with LLAMA_ARG_CACHE_RAM=0 and owns D:/RAGScope_Runtime/runs/gpu.lock.
Only **4B E3** is now running, exec78438, log D:/RAGScope_Runtime/logs/generation_e3_qwen_4b.log.
Command already started; do not duplicate:

```powershell
. ./scripts/runtime_env.ps1
& D:/RAGScope_Runtime/envs/core/Scripts/python.exe scripts/run_generation_campaign.py --campaign-dir D:/RAGScope_Runtime/runs/20261014_v2_generation_e3 --model-id qwen_4b
```

Before4B start: APIps was empty and hostavailable3,255,685,120bytes. The project's CPU embedding11436 and retrieval matrix have resumed under data_preparation ownership. E0 CPU encoding/retrieval is COMPLETE; chunking_core confirmed session37101 exit0 and actualworker37108 absent after full22,254encoding+200Dense+200Hybrid and independent original-index SHA audit. No E0/Laya/BGE-base model remains loaded by that agent.

Next operation: poll exec78438; inspect its retained rows and completion summary. On completion unload only own4B via own11435/api/generate keep_alive0, verify APIps empty and GPU resources released, take freshRAM measurement. Then run the same command with --model-id qwen_9b and log generation_e3_qwen_9b.log. After9B and all GPU work, stop ownservice through generation.stop and verify release. Original user workloads remain **USER_MANUAL_RESUME**; never restart original jobs/daily jobs.

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
