# 生成与评测代码核验（2026-10-14）

本页记录当前分支的代码与离线验证；不代表任何真实模型质量实验已完成。

| 主任务书问题 | 本机事实 | 修改与证据 |
|---|---|---|
| S1.2-9 缓存身份与错误缓存 | 已确认旧键仅含问题、检索器类名与 k，且错误会缓存 | `ragscope/generation/pipeline.py` 键包含模型/权重/后端、索引、实际上下文与实际 prompt 哈希、解码和 scope；正式评测禁用；错误及无效引用不缓存 |
| S1.2-10 JSON 与引用 | 已确认旧解析失败自动引用首块；验证范围为原始检索列表 | 原始输出完整保留；失败答案为空且显式 parse_error；引用仅在 ContextBundle 中验证，未发送 ID 列为 invalid_citations |
| S1.2-11 分母偏差 | 已确认 benchmark 异常后 continue，完全失败配置也被丢弃 | 每个预选问题均保留一行；失败计零，总体和成功条件均值分开；初始化失败也保留配置 |
| S1.2-12 Recall/HitRate | 单相关文档时理论上相同，旧 CSV 未修改 | 集合召回/命中分别实现；重复文档不重复得分，nDCG 不超过 1 |
| S1.2-13 重排候选深度 | 已确认旧 benchmark 10→10 | 修正默认 50→10；离线测试捕获实际候选数 |
| S1.2-14 长文证据与生成指标 | 旧仅文档指标及单参考答案重合 | 新增 canonical 区间合并、连续/完整单元/完整参考集合覆盖；多参考 EM/F1/ROUGE-L；拒答与引用指标；论文聚类 bootstrap 2000 次 |
| S1.2-15 import 下载 | 本模块移除 rouge-score/NLTK import 链；tiktoken 延迟获取 | 模块导入无需联网/模型；reference tokenizer 缓存显式 D 盘；离线测试使用明确字符计数器验证预算逻辑 |

`pack_context` 将重叠切块按规范化全文坐标去重，片段文字是原片段精确切片；切片引用 ID 带新坐标。父块按最佳子块排名首次出现，同父只发送一次；相邻/父块跨论文链接拒绝。标题、引用标签和分隔符均计入预算；每个拒绝片段留原因。模型预算回调接收完整候选上下文，可闭包绑定每个模型真实模板与问题以构造共同 ContextBundle。`validate_bundle` 拒绝篡改。

统一本地客户端仅允许环回 HTTP、禁止重定向及环境代理，不读取 .env 或密钥。默认项目 Ollama 端口 11435；CPU 请求显式 num_gpu=0，CPU-only 进程禁止 GPU 配置；llama.cpp 在 CPU-only 进程必须另有已核验 CPU 服务标记。构造客户端不启动服务、不下载模型。正式生成要求真实模型身份、revision/权重哈希/后端版本及真实 chat-template token 计数器。

技术 JSON 修复策略冻结为零次自动修复（允许上限一次，不是必须一次）：避免修改事实内容；原始 parse_error 保留。非流式首 token 时间填空，不从总时长推算。

v2 ROUGE-L 定义为与 EM/F1 相同规范化下的无词干化词级 LCS F1。旧 scorer 使用 stemming，与旧报告不可直接混合。证据多参考主规则冻结为按“完整集合→完整单元→连续覆盖”字典序选择最好合法集合，平局选原始首个；每个集合的结果全部保留。不可回答的证据指标和答案重合指标为 N/A，拒答单独统计。不会把引用可见率称为 faithfulness。

验证命令（CPU-only，实际执行）：

```powershell
. ./scripts/runtime_env.ps1
$env:COVERAGE_FILE='D:/RAGScope_Runtime/runs/generation_coverage.data'
& 'D:/RAGScope_Runtime/envs/core/Scripts/python.exe' -m pytest tests/unit/test_generation_v2.py tests/unit/test_metrics_v2.py --cov=ragscope.generation --cov=ragscope.evaluation --cov-report=term --cov-report=json:D:/RAGScope_Runtime/runs/generation_coverage.json -q -p no:cacheprovider
```

结果：50 passed，9.92 秒；合并行覆盖 97%，pipeline 98%、context 95%、本地客户端 99%、答案指标 100%、证据指标 99%、统计与汇总 95%。这些数字来自离线软件测试，不是问答质量。未进行 GPU 分配、模型服务请求或付费 API 调用。

官方接口核验（仅查文档，未调用模型）：[Ollama chat](https://docs.ollama.com/api/chat) 支持 schema、think、keep_alive 与真实计时字段；[llama.cpp 官方 JSON schema 示例](https://github.com/ggml-org/llama.cpp/blob/master/examples/json_schema_pydantic_example.py) 采用 json_object + schema。参数传递不等于已证明某固定模型关闭 thinking/遵守 JSON；该结论仍须真实模型预检。

## UI 接入与实际验证

`ui/chat_app.py` 保留 Streamlit，移除导入/打开页面即加载全部模型与自动 Mock 回退。`configs/v2/ui.json` 只登记就绪索引与模型；现场请求前检查索引产物哈希和模型本地服务，输入预算使用后端真实模板。支持指定论文 scope、直接/相邻/父块上下文、可展开实际片段和来源坐标、错误/拒答、分段计时、资源与 offload 记录、不可见引用警示、历史实验回放。记录浏览器只接受配置哈希和记录哈希完整的 JSONL，不会修复或改写实验文件。

UI 离线验证：`tests/unit/test_ui_v2.py` 7 passed，13.17 秒。使用 Streamlit AppTest 执行组件逻辑，无浏览器截图声明。低可用 RAM 条件下没有为截图额外启动浏览器/GPU模型。

真实数据界面冒烟：实际读取 `E1_F512_O0_bm25_none` 的355块索引及100条已完成dev记录；问题 `What are the three regularization terms?`，scope `qasper:1503.00841`，返回10块、实际发送3片段，异常0。这里只执行真实检索，不调用生成模型。证据：`D:/RAGScope_Runtime/runs/ui_real_data_smoke.json`。该索引登记为工程预览；尚未依据未完成的全矩阵声称dev最优。

## 冻结生成执行器（后续增加）

`ragscope/experiments/generation_campaign.py` 提供全局10000次调用账本（默认优先保留E2/E3的3600次）、不可变输入/模型/配置manifest、按模型串行执行、独立3次warmup、错误行保留和哈希校验续跑。预算耗尽时未请求的任务保持pending，不制造模型失败结果。每个方法单独汇总，不能把R0–R3混成一个质量结论。

`GenerationBudget` 与 `CountedClient` 同时供预检、路由、微调前后对照和自动裁判使用；同一全局路径 `D:/RAGScope_Runtime/runs/20261014_v2/generation_budget.json`。已预留但中断的请求保守计入，不自动返还。LoRA paired evaluation现在强制传入同一个budget。

`scripts/freeze_generation_v2.py` 从正式Hybrid dev选型、已完成检索记录、固定语料与每模型20个dev预检构造E2/E3、E4–E7及路由train200/dev60清单。逐条验证实际后端输入token等于已核验模板计数；所有适用模型共同预算检查后只打包一次。未完成的前置条件写FREEZE_READINESS，不写正式manifest；显式受阻必须有可读取的证据文件。E7只按来源与预算可行性预选，保持三种排列相同片段与reference token长度；干扰段来自同论文、排除所有标注证据集合，其语义无关性尚无人审，单独披露。

新增11个纯离线执行器测试通过；包括断点续跑、全局预算保留、失败分母、源证据Oracle、位置文本守恒与counter/backend token差值。与生成/指标/训练测试合计71 passed，9.77秒。此计数不包含后续freeze脚本测试；这些均不构成真实模型实验结果。

后续完整矩阵、共同上下文和训练边界回归共82项通过，19.42秒。最新合并行覆盖94%，其中 generation_campaign 95%、context98%、pipeline99%；报告仍为 `D:/RAGScope_Runtime/runs/generation_coverage.json`。覆盖范围现在包含新执行器，不能将该94%与此前较小范围的97%作质量涨跌比较。真实后端token counter由根任务另行用逐条pilot核验，离线测试不冒充模型预检。

冻结器已要求 `CANDIDATE_SELECTION.json` 和 `CONTEXT_SELECTION.json` 的候选哈希一致，所有方法按开发集选定的none/neighbor/parent策略使用同一默认上下文。Qwen采用text-only官方语言权重别名；LFM采用独立已核验模板计数器。两者都必须通过同身份、当前prompt版本的20个真实dev预检才能进入正式manifest。

最终 dev 协议修正：rag-json-v2.2 在文字问题/证据内容不变的情况下，给本地后端传入每个 ContextBundle 专属的 JSON schema。citations.items.enum 仅含实际发送片段ID；空上下文采用 maxItems=0。结果保留完整 output_schema、output_schema_hash、citation_constraint，配置及缓存身份携带 visible-citation-enum-v1。此后引用ID可见性由约束保证，不能据此宣称模型支持性或faithfulness提高。没有修补事实内容或事后伪造引用。旧v2.1预检记录保留，最终冻结只接受v2.2对应schema验证后的20dev预检。

2026-10-14最新生成/冻结/资源/训练契约/支持性判分合计76项回归通过（7.16s）；不包含真实训练或judge效果结论。
