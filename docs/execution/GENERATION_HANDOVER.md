# Current generation / training handover

Updated for the reported host restart at 2026-10-14 07:03 Australia/Sydney. This timestamp is a parent-provided restart notice, not a restart performed by this task. Exact commands and time-estimate assumptions: [GENERATION_RESUME_20261014.md](GENERATION_RESUME_20261014.md).

## Latest queue override

The latest explicit user authorization permits GPU at any phase and states no competing user GPU job. The current 4B run completed all 800 formal tasks and 3 warmups; its model and owned 11435 service have been released for the bounded GPU R3 window. HOLD 9B pending root's fresh resource handoff. Root is preparing a revised queue that prioritizes main qwen_2b local-owned routing/auxiliary tasks once the main manifest is frozen, followed by other main models and the remaining E2 9B work. Judge and same-BF16 paired evaluation remain pending that queue. No model is currently owned or loaded by this agent. Preserve frozen E2 and CPU R3 TEST200 sources unchanged.

## Current execution boundary

Latest at 2026-10-14 04:43 UTC: E2 is FROZEN with manifest_hash `6a2862e3fed696fb0224d23abc78437c5eaa318efb190c13b2422724ca3246e5`. Both qwen_2b and qwen_4b completed 800 formal tasks plus 3 warmups each, with zero errors and all 803 requests per model passing actual token parity. Both record/task/input/ticket audits passed. E2 is therefore 1600/2400 complete; qwen_9b remains pending. Global budget is 3049/10000 used, including 2800 core calls.

The 4B worker session 36734 exited 0. Owned service PID 3644/create 1791948671.6012218 is verified exited after a normal requested stop; API was empty before stop, port 11435 is closed and bindable, and gpu.lock plus generation-client.lock were successfully reacquired and released. Fresh release observation: 4,221,313,024 host-available bytes, GPU 12 MiB/0%. `generation_e2/RELEASE_4B_SERVICE_FOR_GPU_R3.json` SHA `8aa39f2d3b707f85cd3cc0a4d7fec204c68c78d972f9776047615f20d67752f3` preserves the service receipt and model audit. The supervisor's Windows exit 1 is an intentional requested service termination, confirmed by EXITED/request/unloaded=true; it is not a generation error.

`generation_e2/OPERATOR_CHECKPOINT.json` records WAITING_ROOT_GPU_HANDOFF_AFTER_R3. Do not restart while the GPU belongs to R3. When root grants a new generation window, verify the old owned service exited, port 11435 is available, its own stop marker is archived, and the original resource/current-boot lease guards pass. Use the pinned portable binary:

```powershell
. ./scripts/runtime_env.ps1 -Gpu
& D:/RAGScope_Runtime/envs/core/Scripts/python.exe scripts/serve_generation.py --binary D:/RAGScope_Runtime/tools/ollama-0.34.4/ollama.exe
```

The initial E2 conditions were host RAM 5,001,850,880 bytes, R3 CPU4 active, auxiliary worker/service exited and no active pytest. R3 CPU exited under its own memory guard during the 4B run; `PARALLEL_CONDITION_CHANGE_R3_EXIT.json` preserves that latency/resource boundary. E2 timing and progress were monitored without reading test quality means or changing the protocol.

## Historical recovery and authorization sequence

The following paragraphs retain earlier resource observations and source waits. They are historical, superseded by the completed 2B/4B and released-service state above.

Deadline update: user requires completion by 2026-10-14T10:29:06.606796Z (21:29 Sydney), recorded in EXECUTION_DEADLINE.json. E2 now has an independent unchanged2400-task core shard; see [E2_CORE_SHARD.md](E2_CORE_SHARD.md). After R3 TEST200 is sealed and root grants resources, stress22→E2 may run in one serial GPU window while the remaining R3 CPU4 work continues. This supersedes waiting for all660 R3/auxiliary sources before core generation. No GPU/model has been started by this engineering change.

Update at 03:27 UTC: root authorized the stress GPU window using fresh parallel-workload evidence. Own service started empty and exposed a real system Ollama auto-update to0.40.3, conflicting with frozen0.34.4; it was normally stopped with zero model requests. Official0.34.4 portable restoration to D:/RAGScope_Runtime/tools/ollama-0.34.4 passed archive SHA and82-file verification; executable SHA exactly matches the old pilot/E3 service. Own11435 is READY using explicit --binary and all22 frozen stress calls completed after fresh3.872GB RAM guard. Both windows were unloaded; service remains READY empty while waiting for the sealed R3 TEST200 source notice. Restoration acceptance: runtime downloads/ollama-0.34.4/RESTORATION_ACCEPTANCE.json. System installation and frozen experiment identities are preserved. Do not treat any pre-restart PID, service receipt, port state, GPU measurement or ACQUIRED lease as current authority. Original workloads remain USER_MANUAL_RESUME; never restart them or their daily jobs. WSL configuration remains unchanged.

Parent coordinates current resource ownership. Stress is authorized now after portable backend restoration and the existing effective3GiB host-RAM guard; E2 still awaits the explicit R3 TEST200 sealed-source notice. A freshly validated current-boot lease is required for both. R3 auxiliary CPU4 work may continue only under the agreed concurrent resource conditions. Laya and remaining SQuAD/simple sources precede the full main freeze, but do not block E2. Do not poll/operate data worker handles or consume network recovery attempts.

Current-boot validation now rejects old leases at training, service, formal HTTP/UI, pilot and PowerShell GPU entry points. Required new fields are host_platform=sys.platform, host_boot_time_unix=psutil.boot_time(), validated_at=time.time(); finite numeric timestamps, boot difference ≤2 s, validation at/after boot and not future. Root owns read-only safety inspection, old-lease archival and fresh issue. This agent did not change the actual lease.

Root subsequently archived the old lease and issued a current-boot receipt, SHA884715ec4556624cd069b461b1b9cfc3483f0bab32f6f77c6cf2165f6d6b0cac. Its actual validator check passed; evidence is main run/GPU_LEASE_REVALIDATION_20261014.json. The new receipt records NOT_RUNNING_OBSERVED_WSL_OFF and USER_MANUAL_RESUME, linking the old lease as historical artifact rather than carrying old PIDs forward as active. This does not authorize starting stress before the separate resource handoff.

Recovery audit directory: D:/RAGScope_Runtime/runs/20261014_v2/recovery/generation_post_reboot_20261014. PREFLIGHT.json verifies 18 E3 sealed artifacts and 1200 original row hashes/tickets, the complete best LoRA checkpoint, candidate/context linkage, and frozen stress implementation/source hashes. GPU_LEASE_IMPLEMENTATION_AUDIT.json binds the 50 passing CPU tests and real OLD_LEASE_REJECTION.json. The old real lease was rejected with CLI exit2; PowerShell cleared its CUDA variable and rejected it; lease SHA stayed unchanged. No model requests, service starts or downloads.

E3 is COMPLETE1200/1200 (three models ×400, zero request errors, token parity1200, nine separate warmups). Do not rerun it. Main default_spec automatically references its immutable manifest; an in-memory E3-only parent import exercise passed exact task/input/model ownership checks. Full main import remains pending until its real manifest is frozen; this audit does not claim main completion. Preserve both old2B analysis and complete E3 analysis artifacts.

Historical snapshot immediately after completed stress22: the budget was1443/10000 used,1200 core used:8557 total calls available and2400 protected for E2. The earlier recovery-only snapshot1421 is historical. Maximum remaining main formal calls4320 plus12 initial model warmups, pairedBF16206, online66 and judge≤200 leave3753 calls before additional restarts/retries, if every planned group is feasible; stress22 is already included in the used1443. All further attempts remain counted; no budget was reserved by this audit.

The 2B 4K/8K stress is COMPLETE:8formal+3warmup per window,22/22 real calls with zero errors and exact token parity3584/7680 plus512 output reserve. Manifest e87d506a9eab594eac0f088317d094df315c82097bc3eaa3ce0a56cbb04589c9 and both frozen stress source files remain unchanged. COMPLETION_RECEIPT.json validates every source/record/config artifact and unique budget tickets1422–1443. Historical pre-load RAM rejection and empty-service version mismatch are retained, neither is a model OOM. Resource report: CONTEXT_STRESS_REPORT.md. All stress models are unloaded, API ps empty, GPU12MiB/0%, service READY empty pendingE2.

LoRA formal training is COMPLETE122steps/972train/59dev; do not train again. Real CPU check-only reverified beststep122 checkpoint and returned exit2 WAITING_FOR_FROZEN_GENERATION. Paired100+100 BF16 evaluation has not run and will permanently bind the E2 shard R2 contexts, plus its own later resource handoff.

## Authorized continuation after the R3 TEST200 source notice

Root granted continuous execution on2026-10-14: E2 three-model2400 + production acceptance/seal, then E2-bound support judge200/Qwen4B, then normal unload/stop owned11435 and exact PID/port/gpu.lock release, then best122 same-BF16 paired100+100 after its core check-only and fresh original resource/lease guard. No phase approval wait, no new combinations, no retraining. Root may reprioritize when main sources become ready. Each segment records actual concurrent CPU4 process identities and resource conditions, retains failures/tickets, and reports startup identity/time estimate. Runtime continuation receipt: GENERATION_CONTINUATION_20261014T0341.json.

Judge uses `scripts/run_support_judge_v2.py --campaign-dir D:/RAGScope_Runtime/runs/20261014_v2_generation_e2 --judge-model qwen_4b --sample-size 200` to freeze, then the same command with `--run`; it must follow complete E2 and freshresource checks. Paired evaluation uses the E2 command below, first corePython `--check-only`, then isolatedtrainPython after11435exit. The same paired directory must remain bound to E2 permanently.

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

Dev selections are frozen: simple F512_O0, structured R512, both neighbor expansion. E2/R2 final contexts are now frozen in the E2 shard; paired BF16 evaluation will remain permanently bound to that shard. Paired100BF16 evaluation has NOT run. The earlier CPU check-only WAITING_FOR_FROZEN_GENERATION receipt is historical and must be rerun against the existing E2 manifest before actual evaluation.
After the separately authorized BF16 GPU handover and fresh checks (do not later switch the same paired output directory to the main manifest):

```powershell
& D:/RAGScope_Runtime/envs/train/Scripts/python.exe scripts/evaluate_lora_v2.py --training-run D:/RAGScope_Runtime/runs/20261014_v2_lora/train_2b_1024_seed42 --campaign-dir D:/RAGScope_Runtime/runs/20261014_v2_generation_e2 --model-dir D:/RAGScope_Runtime/models/Qwen3.5-2B/15852e8c16360a2fea060d615a32b45270f8a8fc
```

--check-only uses corePython and neverloadsmodel/GPU. Actualrun requests additional2GiB plus1GiB reserve (effective3GiB available),3separatewarmups/condition/session,globalcounting,INCOMPLETEsummary on circuitbreak. BothBF16conditions share protocol/precision; their prompt-only JSON differs from Q4schema-constrained mainmatrix, so no precision-isolated crossprotocolclaim.

Online runner: scripts/run_online_routing_v2.py --campaign <main> --routing-dir <routing> --mode freeze|run. All7policies' decisions freeze beforetestscoreaccess; onlyfull60devselectsbestfixed/nonfixed, same30dev onlinepair IDs. Actualchoose→retrieve→2048pack→sharedallQwencounterrepack→2Bgenerate;3warmups eachsession/policy,alternatingpairedorder,resume/globalbudget. Perpolicyactualgeneration_attempted andpregenerationfailures count separately; fewer30actualrequests preventsdeployment. Full60dev quality gate and30onlineP50 gate reported separately. CPU BGE sharedverifiedloader and effective3GiB available RAM guard (additional2GiB plus1GiB reserve). Selectedlearnedrouter must use real_generation_proxy andexacttrain/dev IDs. Noactualonline rows yet;9offlinecontracttestsPASS.

Support judge is implemented and not run. The already authorized command uses --campaign-dir D:/RAGScope_Runtime/runs/20261014_v2_generation_e2 --judge-model qwen_4b --sample-size 200; it freezes the original 200 stratified E2 IDs before judging. --run requires complete E2 and the pinned judge model. Rawprompt/outputs/hashes preserved,failuresunknown,selfmodelbiasflagged,humanreviewNOT_REVIEWED.

Current focused recovery verification: 50 GPU-lease/training/online CPU contract tests passed in 7.45 s. Root maintains full-suite status separately. These fixtures are functional tests and never actual experiment results.
