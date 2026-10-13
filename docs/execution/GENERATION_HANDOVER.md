# Generation / training handover state

Last verified software state: generation/evaluation/campaign/freeze/training combined82 tests passed (latest coverage94% including campaign95%); subsequent focused changes33 tests passed. No model generation calls or GPU allocation have been performed by this subagent. Root owns live pilots and GPU lease/service lifecycle.

Completed implementation: generation ContextBundle/JSON/error/cache corrections; evidence metrics and failure denominators; Streamlit retrieval/replay/readiness; BF16 training data/checkpoint/runner/paired evaluation; full-matrix freeze and single-model generation CLI. Actual generator models are not declared complete by these code changes.

Current dependency hold: root is finishing4B/9B (and planned LFM) pilots; do not load Torch/tokenizer or models concurrently. Trainenv is ready at D:/RAGScope_Runtime/envs/train/Scripts/python.exe. Wait for root's explicit training/preparation handover; original user tasks are now manual resume by latest human instruction, no automatic restoration by this subagent.

Next exact CPU tokenizer command after root handover:

```powershell
. ./scripts/runtime_env.ps1
$env:USE_TORCH='0'
& 'D:/RAGScope_Runtime/envs/train/Scripts/python.exe' -m ragscope.training.prepare --corpus-dir D:/RAGScope_Runtime/corpora/20261014_v2 --tokenizer-dir D:/RAGScope_Runtime/models/tokenizers/Qwen3.5-2B --output-dir D:/RAGScope_Runtime/runs/20261014_v2_lora/prepared_2b_1024 --split train --max-seq-len 1024
& 'D:/RAGScope_Runtime/envs/train/Scripts/python.exe' -m ragscope.training.prepare --corpus-dir D:/RAGScope_Runtime/corpora/20261014_v2 --tokenizer-dir D:/RAGScope_Runtime/models/tokenizers/Qwen3.5-2B --output-dir D:/RAGScope_Runtime/runs/20261014_v2_lora/prepared_2b_1024 --split dev --max-seq-len 1024
```

These commands use real official tokenizer and assistant-only labels, no weights/GPU. Check accepted/rejected manifests and exact loss-mask boundary, then run real GPU preflight only after explicit root handover and project service release. BF16 snapshot: D:/RAGScope_Runtime/models/Qwen3.5-2B/15852e8c16360a2fea060d615a32b45270f8a8fc .

Generation freeze command: core Python scripts/freeze_generation_v2.py . It must require final CANDIDATE_SELECTION and CONTEXT_SELECTION matching hashes, completed actual source records, current-prompt20dev successful pilots with exact backend token parity. Defaults cover E2/E3/E4/E5/E6/E7 and ROUTE_TRAIN200/ROUTE_DEV60. Missing work stays pending; explicit blocks require evidence. Output D:/RAGScope_Runtime/runs/20261014_v2_generation/FREEZE_READINESS.json is currently pending; no formal EXPERIMENT_MANIFEST should exist yet.

Formal generation command when fully frozen and owned service alreadyREADY: core Python scripts/run_generation_campaign.py --campaign-dir D:/RAGScope_Runtime/runs/20261014_v2_generation --model-id qwen_2b . Other model IDs qwen_4b/qwen_9b/lfm serially. Root-owned service itself holds global GPU lock. Client checks lease/service PID each call and uses one cross-campaign client lock. Exact budget file D:/RAGScope_Runtime/runs/20261014_v2/generation_budget.json includes pilots, warmups, failures, router labels and LoRA comparisons. No infinite retries; transport failures circuit-break after3, OOM after1. Completed marker summaries/MODEL_<id>.json requires n_pending=0 and matching manifest_hash; failure rows remain scored failures, explicit blocked groups remain blocked.

Learning doc06 contains time-stamped actual BM25 dev/test/E8 figures and earlier pilot limitation snapshot, not final results. Root later2BpilotPASS20 message supersedes that snapshot; retain distinction.

Formal resource instrumentation added: `MeasuredClient` wraps each real request withoutTorch, sampling at1s. Fields explicitly mean sampled whole-device VRAM, service-tree aggregateRSS and host-availableRAM; it records raw samples/instrument errors and backend`/api/ps` resident allocation separately. No allocator peak or layer-offload count is invented. `client_request_wall_ms` and `telemetry_finalize_ms` separate post-request monitoring overhead; pipeline generation_ms includes wrapper overhead. Runtime identity now includes `{prompt_ram_cache_mib:0,prompt_cache_policy:backend_history_disabled}`. Freeze requires LLAMA_ARG_CACHE_RAM=0 in actual ownedservice receipt, snapshots receipt, and excludes pilot latencies from formal timing summaries.

Resource/freeze/campaign/training focused regression37 passed5.16s, still entirely offline. Latest CPU preprocessor data.py additionally rejects any packing that removes a required gold evidence unit. Next actual tokenizer command remains on hold untilroot handover (other modelpilots currentlyrunning).

当前资源交接更新（2026-10-14）：root已完成所有v2.1+cachepolicy模型20dev预检，随后因opaque引用ID复制失败而固定最终v2.2枚举schema协议；76项离线回归已过。root正在执行唯一一次最终enum预检，GPU未交给本subagent。BGE CPU agent明确其进程24532/26484均已退出。

实际训练准备：首次进程因应用依赖diskcache缺失退出；trainenv补装diskcache5.6.3/loguru0.7.3/tiktoken0.14.0。第二次完整1000train处理发现TF5.19 apply_chat_template默认return_dict=True而被严格类型守卫全部拒绝（prepared_2b_1024保留）。显式return_dict=False修复后，官方tokenizer真实单例prefix验证通过，prompt15+assistant4tokens；13项训练契约测试通过。TF5.19实际仍导入Torch，即使USE_TORCH=0；探针退出，CUDA mask为空未初始化模型。现在等待root最终enum pilots结束再串行准备，避免内存叠加。

下一条准确CPU命令：`. ./scripts/runtime_env.ps1; & D:/RAGScope_Runtime/envs/train/Scripts/python.exe -m ragscope.training.prepare --corpus-dir D:/RAGScope_Runtime/corpora/20261014_v2 --tokenizer-dir D:/RAGScope_Runtime/models/tokenizers/Qwen3.5-2B --output-dir D:/RAGScope_Runtime/runs/20261014_v2_lora/prepared_2b_1024_v2 --split train --max-seq-len 1024`（再相同dev）。新目录防止覆盖原拒绝证据。训练尚未加载权重。训练循环已加真实resume adapter/optimizer/scheduler/RNG/data顺序/进度精确核验receipt与每step有限梯度、GPUallocator峰值和每microbatch主机RAM观测；GPU授权交回后做5→10实际恢复。
