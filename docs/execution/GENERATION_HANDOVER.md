# Current generation / training handover

Updated for the reported host restart at 2026-10-14 07:03 Australia/Sydney. This timestamp is a parent-provided restart notice, not a restart performed by this task. Exact commands and time-estimate assumptions: [GENERATION_RESUME_20261014.md](GENERATION_RESUME_20261014.md).

## Current execution boundary

Deadline update: user requires completion by 2026-10-14T10:29:06.606796Z (21:29 Sydney), recorded in EXECUTION_DEADLINE.json. E2 now has an independent unchanged2400-task core shard; see [E2_CORE_SHARD.md](E2_CORE_SHARD.md). After R3 TEST200 is sealed and root grants resources, stress22→E2 may run in one serial GPU window while the remaining R3 CPU4 work continues. This supersedes waiting for all660 R3/auxiliary sources before core generation. No GPU/model has been started by this engineering change.

No generation model or service has been started by this recovery preflight. Do not treat any pre-restart PID, service receipt, port state, GPU measurement or ACQUIRED lease as current authority. Original workloads remain USER_MANUAL_RESUME; never restart them or their daily jobs. WSL configuration remains unchanged.

Parent coordinates current resource ownership. Await explicit R3 TEST200 readiness and GPU/RAM handoff, a freshly validated lease and the existing effective3GiB host-RAM guard before stress/E2. R3 auxiliary CPU4 work may continue only under the agreed concurrent resource conditions. Laya and remaining SQuAD/simple sources precede the full main freeze, but do not block E2. Do not poll/operate data worker handles or consume network recovery attempts.

Current-boot validation now rejects old leases at training, service, formal HTTP/UI, pilot and PowerShell GPU entry points. Required new fields are host_platform=sys.platform, host_boot_time_unix=psutil.boot_time(), validated_at=time.time(); finite numeric timestamps, boot difference ≤2 s, validation at/after boot and not future. Root owns read-only safety inspection, old-lease archival and fresh issue. This agent did not change the actual lease.

Root subsequently archived the old lease and issued a current-boot receipt, SHA884715ec4556624cd069b461b1b9cfc3483f0bab32f6f77c6cf2165f6d6b0cac. Its actual validator check passed; evidence is main run/GPU_LEASE_REVALIDATION_20261014.json. The new receipt records NOT_RUNNING_OBSERVED_WSL_OFF and USER_MANUAL_RESUME, linking the old lease as historical artifact rather than carrying old PIDs forward as active. This does not authorize starting stress before the separate resource handoff.

Recovery audit directory: D:/RAGScope_Runtime/runs/20261014_v2/recovery/generation_post_reboot_20261014. PREFLIGHT.json verifies 18 E3 sealed artifacts and 1200 original row hashes/tickets, the complete best LoRA checkpoint, candidate/context linkage, and frozen stress implementation/source hashes. GPU_LEASE_IMPLEMENTATION_AUDIT.json binds the 50 passing CPU tests and real OLD_LEASE_REJECTION.json. The old real lease was rejected with CLI exit2; PowerShell cleared its CUDA variable and rejected it; lease SHA stayed unchanged. No model requests, service starts or downloads.

E3 is COMPLETE1200/1200 (three models ×400, zero request errors, token parity1200, nine separate warmups). Do not rerun it. Main default_spec automatically references its immutable manifest; an in-memory E3-only parent import exercise passed exact task/input/model ownership checks. Full main import remains pending until its real manifest is frozen; this audit does not claim main completion. Preserve both old2B analysis and complete E3 analysis artifacts.

The read-only budget snapshot remains1421/10000 used,1200 core used:8579 total calls available and2400 protected for E2. Maximum remaining main formal calls4320 plus12 initial model warmups; stress22, pairedBF16206, online66 and judge≤200 leave3753 calls before additional restarts/retries, if every planned group is feasible. All further attempts remain counted; no budget was reserved by this audit.

The 2B 4K/8K stress remains frozen and unrun. Eight dev questions ×two windows plus six warmups =22 calls; exact prompt counts3584/7680 plus512 outputs. Manifest e87d506a9eab594eac0f088317d094df315c82097bc3eaa3ce0a56cbb04589c9. Both frozen stress source files are unchanged by lease fixes. Historical single pre-load RAM rejection remains distinct from model OOM. Keep it PENDING, not hardware-BLOCKED merely because it is queued.

LoRA formal training is COMPLETE122steps/972train/59dev; do not train again. Real CPU check-only reverified beststep122 checkpoint and returned exit2 WAITING_FOR_FROZEN_GENERATION. Paired100+100 BF16 evaluation has not run and waits for final main E2/R2 contexts plus its own resource handoff.

## Completed E3 and historical resource measurements

Shard directory D:/RAGScope_Runtime/runs/20261014_v2_generation_e3.
Manifest_hash 8c6a71031d25f0a0218a85a3abbd9d8dc35d3c0f56dafa046e8249beeffb5599.
Exactly1200tasks: original200QASPERtest×oracle/no_context×3Qwen. Same final promptv2.2, model identities, reference2048+all3true-template budget and512output reserve as futuremain E3. No retrieval selection or test-output-based protocol change.

2B COMPLETE400/400 formalrows,0errors; exec81336 exit0,3separatewarmups. Unload confirmed APIpsempty/GPU0MiB in RELEASE_2B_FOR_E0.json. Do not rerun2B.
For182answerable questions, OracleF1=.5785604023 andNoContextF1=.0414930593;18unanswerable separate. This is a source-context diagnostic, not a retrieval-method gain.
Root's independent2B numeric audit10/10PASS is in e3/analysis/ANALYSIS.json, GENERATION_TABLES.md, INDEPENDENT_NUMBERS_2B.json. Preserve this snapshot; later completeE3 analysis must use a new analysis_complete directory.

Historical2B completion budget was615total/400core; finalE3 is1421total/1200core in the same D:/RAGScope_Runtime/runs/20261014_v2/generation_budget.json. All pilots,warmups,failures,mainmatrix,routerlabels,judge,LoRAeval count. OOM1/consecutivebackend3 circuitbreak, no silent retries/replacements. Actualprompttokens use backend_metadata.actual_input_tokens/token_count_match; RAGResult.prompt_tokens is legacy reference-token statistics.

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

Dev selections are already frozen: simple F512_O0, structured R512, both neighbor expansion. Main E2/R2 final contexts remain pending completion of full-corpus retrieval materialization and R3; this is not pending dev selection. Paired100BF16 evaluation has NOT run. RealCPUcheck-only wrote paired_test100/READINESS.json WAITING_FOR_FROZEN_GENERATION,weights_loaded=false,gpu_used=false,generation_calls0.
When finalcontexts and a newGPUhandover are ready:

```powershell
& D:/RAGScope_Runtime/envs/train/Scripts/python.exe scripts/evaluate_lora_v2.py --training-run D:/RAGScope_Runtime/runs/20261014_v2_lora/train_2b_1024_seed42 --campaign-dir D:/RAGScope_Runtime/runs/20261014_v2_generation --model-dir D:/RAGScope_Runtime/models/Qwen3.5-2B/15852e8c16360a2fea060d615a32b45270f8a8fc
```

--check-only uses corePython and neverloadsmodel/GPU. Actualrun requests additional2GiB plus1GiB reserve (effective3GiB available),3separatewarmups/condition/session,globalcounting,INCOMPLETEsummary on circuitbreak. BothBF16conditions share protocol/precision; their prompt-only JSON differs from Q4schema-constrained mainmatrix, so no precision-isolated crossprotocolclaim.

Online runner: scripts/run_online_routing_v2.py --campaign <main> --routing-dir <routing> --mode freeze|run. All7policies' decisions freeze beforetestscoreaccess; onlyfull60devselectsbestfixed/nonfixed, same30dev onlinepair IDs. Actualchoose→retrieve→2048pack→sharedallQwencounterrepack→2Bgenerate;3warmups eachsession/policy,alternatingpairedorder,resume/globalbudget. Perpolicyactualgeneration_attempted andpregenerationfailures count separately; fewer30actualrequests preventsdeployment. Full60dev quality gate and30onlineP50 gate reported separately. CPU BGE sharedverifiedloader and effective3GiB available RAM guard (additional2GiB plus1GiB reserve). Selectedlearnedrouter must use real_generation_proxy andexacttrain/dev IDs. Noactualonline rows yet;9offlinecontracttestsPASS.

Supportjudge implemented,notrun: scripts/run_support_judge_v2.py --campaign-dir <main> freezes≤200stratifiedE2IDs beforejudging; --run requires completeE2 andpinned4B/9B. Rawprompt/outputs/hashes preserved,failuresunknown,selfmodelbiasflagged,humanreviewNOT_REVIEWED.

Current focused recovery verification: 50 GPU-lease/training/online CPU contract tests passed in 7.45 s. Root maintains full-suite status separately. These fixtures are functional tests and never actual experiment results.
