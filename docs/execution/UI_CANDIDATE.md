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


## 实际离线应用启动QA（2026-10-14 04:49Z）

`scripts/verify_ui_candidate_v2.py` 已用 models 环境中的真实 Streamlit AppTest 执行并exit0。它校验现有候选及其冻结source artifact，实际加载全509篇/6551块索引，在候选默认论文 `qasper:1503.00841` 上提出工程烟测问题“ What problem does this paper address? ”，执行BM25前10→neighbor打包，得到4片、2022个reference tokens；随后从真实E1 BM25文件校验并回放全部100条dev记录。问题由agent编写仅检验工程通路，不属于质量标签或新增实验组合。

收据 `ui_candidate/OFFLINE_STARTUP_QA.json` SHA `19cb41475bd19e2643bfa1ba94b4a4d3413c1a863819711bb95a869f498630d2`，逐次真实检索结果在同目录 `OFFLINE_STARTUP_QA.retrieval.json`。HTTP attempts=0、model calls=0、Torch未导入、未启动服务器、未获取主campaign锁、项目ui.json未变。候选是04:01服务不可用的历史快照，本核验不把其状态冒称04:49服务实时就绪；未验证神经检索/生成，也没有浏览器像素检查。最终仍须root按新鲜服务窗口刷新候选、做浏览器与真实模型QA再发布。

第一次QA session63667 exit1发生在AppTest构造时：全局socket.connect拦截误挡Windows asyncio的本地socketpair自管道，尚未render应用或调用模型，证据保存在 `OFFLINE_STARTUP_QA_FIRST_FAILURE.json`。修正为拒绝应用实际使用的requests/urllib两种HTTP运输、允许框架本地自管道后，真实QA正常通过。2项定向行为测试PASS（1.40秒），分别验证HTTP仍被拒绝以及Windows本地管道能正常工作；无新模型结果。

准确复验命令必须使用新的输出收据路径，入口拒绝覆盖旧QA。示例：

```powershell
. ./scripts/runtime_env.ps1
& 'D:/RAGScope_Runtime/envs/models/Scripts/python.exe' scripts/verify_ui_candidate_v2.py --output 'D:/RAGScope_Runtime/runs/20261014_v2/ui_candidate/OFFLINE_STARTUP_QA_after_changes.json'
```

只在源码或候选变化后需要复验；不要为了等待反复扫描或重跑。当前所有本代理进程已退出，无需再保留CPU模型。
