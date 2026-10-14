# v3 当前恢复点

2026-10-14T17:26:36.485169+00:00；campaign `D:\RAGScope_Runtime\runs\v3_20261015_014506`，分支codex/ragscope-v3-diagnostics。全部本轮模型/下载/审计句柄已真实终态；最新BERT替代session63985 actualexit0、NLI60651 actualexit0；默认BERT17135/31866/97392 actualexit1资源阻断，不能再无限重启。

先读STATUS/CAMPAIGN_IDENTITY/BUDGET和ROOT_CHECKPOINT，不bootstrap、不重封存H_QA、不重跑旧生成矩阵。H_QA200/100论文仍封存。新LLM请求0；人工NOT_REVIEWED；当前只是S1的一部分，Goal未完成。

准确可执行输入恢复校验命令：
```powershell
. ./scripts/runtime_env.ps1
& 'D:/RAGScope_Runtime/envs/core/Scripts/python.exe' scripts/v3/prepare_atomization_groups.py --campaign 'D:/RAGScope_Runtime/runs/v3_20261015_014506' --resume
```

下一实际实现：Implement v3 owned Ollama lifecycle, generic counted client and quote-only claim segmentation. Do not start old hardcoded serve_generation against v2 run. Then run S1 actual facts/support/citation/context judgements before S2.

已读取原LocalLLMClient.chat(messages,max_tokens,temperature,output_schema)，纯tokenizers的Qwen35PromptCounter；model_client_factory可以复用原源manifest/tokenizer/pilot核验，再新身份修改context8192。预算须接v3 Budget而非旧GenerationBudget。原server硬编码20261014_v2/11435/旧lease，新增v3 wrapper不得改或启动旧run。GPU锁仍单模型串行、原训练USER_MANUAL_RESUME、不开WSL不改系统；只有精确自身servicePID可正常卸载退出。完成事实评分和context充分性后才进S2，原训练不自动恢复。
