# UI 资源候选配置

2026-10-14 04:01:12Z，`scripts/configure_ui_v2.py` 已实际 exit0，输出 runtime 候选，尚未发布项目配置。root 负责后续 GUI/GPU QA 与发布。本入口没有启动服务、调用模型推理或进入主 retrieval writer。

候选：`D:/RAGScope_Runtime/runs/20261014_v2/ui_candidate/ui.json`，SHA `73ef6704d26afeba4eb961057ad0a86e2d71d43c0bd21b7e2116746f78e5ff81`。核验收据：同目录 `ui.verification.json`，SHA `eca75ef12397ee7643e8d7ed9574dcaf34ca094e6936196da609461bd2856d9f`。项目 `configs/v2/ui.json` 保持 SHA `02b5413d502cd0982f84034719d60f6708c55fda1435e188a9504fa781c948b6`，没有冒称正式 UI 已完成验收。

实际核验内容：

- 冻结 dev 选择及 context provenance；全509篇 R512 的6551块 dense manifest 与全部产物 SHA，chunk manifest/结构坐标产物，独立 tokenizer bytes 与原 embedding identity。BM25、Dense、Hybrid、Hybrid+Rerank 都绑定同一完整索引，默认 neighbor/reference2048。
- BGE reranker 本地固定 checkpoint 全部文件与真实 dev100/0失败身份，CPU4/windowed_max/50→10；本入口只验文件，没有加载 BGE，不能据此声称并发 RAM 足够或模型驻留。
- `UI_MODEL_PACKAGE_PINS.json` 的 LFM、Qwen2B/4B/9B 四份真实权重 bytes/hash、Ollama manifests、tokenizer/template 和构造出的客户端 identity。模型默认意图为可运行主线中最小的 Qwen2B，未使用 test 分数选型。
- 只发送 GET `/health`、`/api/version`、`/api/tags`、`/api/ps`。实际11435/0.34.4的四个 tags 均匹配，只有 E2 的2B驻留；本入口未加载/卸载任何模型。11436返回 URLError，embedding_ready=false。generation-client.lock 被 E2 持有，四个交互模型 ready=false；前端只展示可用的 BM25/仅检索选项。
- 默认 fixed Hybrid，非fixed routers为空，不能以未绑定的离线速度/质量统计启用自适应路由。现有 `routing_options.deployment_readiness` 仍要求完整 dev/真实 paired online 门槛与运行时绑定，并拒绝 cold UI 与 resident 实验配置混用。

源码接线修正仅两处 `ui/chat_app.py`：候选有 manifest_sha256 时验证整个 manifest 固定哈希；ScopedRetriever 使用独立 tokenizer_id/tokenizer_revision，不再访问真实 embedding identity 中不存在的 model 字段。没有改模型或检索参数。`test_configure_ui_v2.py` 与原 `test_ui_v2.py` 合计13项 PASS（6.83秒）。这些是软件功能 fixture，不计真实质量实验。

第一次实际入口 session98001 exit1：期望的 `E1_R512_dense_neighbor.jsonl` 不存在，因为 dev Dense 只执行过 none。保留 `INITIAL_VALIDATION_FAILURE.json`；改为回放实际完成的 `E1_R512_dense_none.jsonl` 后 session1127 exit0。应用默认上下文仍遵循冻结 neighbor；没有补造记录/新增实验。

后续刷新候选的准确命令（只在 root 的资源/QA 窗口执行，非本子任务自动启动服务）：

```powershell
. ./scripts/runtime_env.ps1
& 'D:/RAGScope_Runtime/envs/core/Scripts/python.exe' scripts/configure_ui_v2.py --output 'D:/RAGScope_Runtime/runs/20261014_v2/ui_candidate/ui_qa_refresh.json'
```

输出强制限制在 runtime/ui_candidate 下，不能直接覆盖项目配置。刷新会再次检查服务/锁；当前收据只是当时状态。可通过 `RAGSCOPE_UI_CONFIG` 指向候选做 QA，发布由 root 在实际验收后完成。原四模型选择、单卡串行锁、生成预算、GPU lease、包字节核验、真实输入预算与原始请求落盘均由既有 UI 执行路径继续强制。
