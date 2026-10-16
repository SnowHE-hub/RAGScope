### 13.4 真实运行入口与正常退出

以下是在本机现有资产上使用的命令，不是要求重装环境或重跑实验。工作目录固定为项目根。第一终端只读检查可核验索引、模型包和源码身份；实际服务命令会加载模型，因此本次文档核验没有重新执行后者。

```powershell
Set-Location -LiteralPath 'D:/MCS_26S1课程资料/个人项目/RAGScope'
. ./scripts/runtime_env.ps1
& D:/RAGScope_Runtime/envs/core/Scripts/python.exe scripts/v3/serve_full_workbench.py --check-only
# 需要交互问答时再启动：
& D:/RAGScope_Runtime/envs/core/Scripts/python.exe scripts/v3/serve_full_workbench.py
```

第二终端启动UI：

```powershell
Set-Location -LiteralPath 'D:/MCS_26S1课程资料/个人项目/RAGScope'
. ./scripts/runtime_env.ps1
./scripts/run_ui_v3.ps1 -CheckOnly
./scripts/run_ui_v3.ps1
```

浏览器访问 `http://127.0.0.1:8502`。工作台不会自动下载模型，也不会在UI未连接服务时偷偷用Mock。UI默认请求11437；虽然服务脚本有 `--port`，改端口不会自动同步UI中的硬编码URL，这是当前配置技术债。

正常退出：两个终端各按Ctrl+C；后端上下文管理器卸载自有模型、释放自有进程与GPU锁。也可按照当前receipt写精确stop文件：

```powershell
$ragConfig = Get-Content -LiteralPath configs/v3/default.json -Raw | ConvertFrom-Json
$ragReceipt = Join-Path $ragConfig.run 'FULL_WORKBENCH_SERVICE.json'
$ragService = Get-Content -LiteralPath $ragReceipt -Raw | ConvertFrom-Json
if ($ragService.status -eq 'READY') {
    Set-Content -LiteralPath $ragService.stop_file -Value 'Own service exit requested' -Encoding utf8
}
```

这是对**这次服务**的退出请求，不是批量结束ollama/python。原训练采用USER_MANUAL_RESUME，由用户手动续跑，文档任务不操作训练、日更、定时器或WSL。最后交付的资源释放证据是V3FINAL `RESOURCE_RELEASE_FINAL.json`，不将历史显存快照当本秒机器状态。

### 13.5 CPU进程怎样调用GPU

PowerShell环境脚本默认把CUDA mask清空、CPU线程设4；API和UI本身仍可以是CPU进程。真正GPU计算在单独拥有的私有Ollama子进程，GPU使用通过当前boot租约、锁、PID/create_time及服务身份重新验证，HTTP客户端拿到明确授权才请求 `num_gpu=999`。所以“core没有torch”与“最终GPU生成”不矛盾。

模型后端允许小编码器和一个2B生成器驻留，所有请求串行；单卡串行的要求不是多个生成模型并发。首次冷加载、驻留prefill/decode、模型卸载、向量/答案缓存是不同性能条件。服务启动额外请求512MiB，加共用1GiB reserve形成约1.5GiB可用内存门槛；驻留编码又有512MiB底线。`[代码]` `OnlineOwnedOllama`、`ManagedOllama`、`require_ram`。

### 13.6 测试层次、日志与复算

| 验证层 | 当前已有证据 | 不能由此推出 |
|---|---|---|
| 静态与单测 | V3FINAL的 `FINAL_PYTEST_DELIVERY.xml`：82通过 | 全库答案都正确；仓库全部当前测试均重新跑过 |
| 历史V2回归 | 交付记录中的core710通过/2skip、models24通过，去重711通过/唯一权限skip等各有口径 | 把所有批次简单相加成一次全量成绩 |
| 真实模型 | 正式响应保存raw、real_model、权重、实际payload和token | 输出schema合法即事实正确 |
| 真实API | 9个输入/范围/总结比较/缓存场景通过 | 高并发、鉴权和互联网部署可用 |
| 组件集成 | Streamlit AppTest连接真实API，真实回答和引用expander存在 | 浏览器视觉像素验证通过 |
| 浏览器视觉 | 最后v3应用访问被权限拒绝，未绕过 | 没有截图也不能伪称验收通过 |

当前轻量核验命令仅重新读取已有结果，不调用模型：

```powershell
& D:/RAGScope_Runtime/envs/core/Scripts/python.exe docs/_architecture/audit_repository.py
& D:/RAGScope_Runtime/envs/core/Scripts/python.exe docs/_architecture/audit_experiment_evidence.py
& D:/RAGScope_Runtime/envs/core/Scripts/python.exe docs/_architecture/build_architecture.py --check-only
```

历史V2任务状态对账入口为：

```powershell
. ./scripts/runtime_env.ps1
& D:/RAGScope_Runtime/envs/core/Scripts/python.exe -m ragscope.experiments.run --profile full --campaign 20261014_v2 --resume --cpu-only --reconcile-only
```

该命令会在运行目录做状态对账，不是重新生成答案；本次没有因写文档而执行。V3模型实验入口 `finish_experiments.py`、`finish_deployment_answers.py`、`finish_global.py`、`finish_online.py`保留，重跑会受冻结哈希与原账本约束。当前已完成的campaign不应以删除账本方式强行重跑；研究复现要明确复用记录与新调用的界限。

JSONL账本保存config_hash、record_hash、raw response和error；持久层使用文件锁、追加写与恢复校验。它提供审计和断点，不是数据库级多用户事务。`observability/`空包不代表无日志，但也不能把这些离线账本描述成完整生产可观测性平台。

## 14. 当前真实运行路径

<!--DIAGRAM:online-->

### 14.1 从点击“生成答案”开始逐步追踪

| 顺序 | 实际函数/文件 | 输入、关键判断与输出 |
|---:|---|---|
| 1 | `ui/full_workbench.py` 的 `main` | `/api/state`取得论文catalog；用户最多选10篇、输入问题、选任务/方法；答案缓存默认false |
| 2 | 同文件 `api` | loopback urllib、禁代理和重定向，POST `/api/answer`，最长等待180秒 |
| 3 | `scripts/v3/serve_full_workbench.py` 的 `Handler.do_POST` | JSON body≤65536字节；仅接受指定路径；同步交给work.answer |
| 4 | `ragscope/v3/full_workbench.py:15` `validate_full_request` | 字段白名单、问题≤8000字符、独立论文≤10、method/task枚举、禁止gold/答案字段混入推理 |
| 5 | `FullWorkbench.answer:50` | 未知论文拒绝；有限英文指代要求选论文；按配置+payload识别可选内存答案缓存 |
| 6 | `ReusableEncoder.query/embed` | 即使选择BM25，当前工作台也先计算/复用query向量；完整identity缓存、4096embedding窗口 |
| 7 | `deployment.py:10,25` | 判定任务与复杂度；选择普通B0/全库CONTROL/复杂FINAL策略和预算 |
| 8 | `FullCorpusRetriever.search` | 论文范围限制行；body BM25+Dense、可选metadata及doc先验；多篇可轮询平衡 |
| 9 | `pack_context:97` / `pack_core_first:6` | scope、坐标去重、邻块、参考和实际模型预算；输出ContextBundle |
| 10 | `answer_messages` / `visible_output_schema` | G0或任务系统提示；仅枚举实际片段ID；多篇限制引用及答案字符数 |
| 11 | `DeploymentClient.chat` → `LocalLLMClient.chat:113` | 固定Q4 2B、8192窗口、512/768输出、非流式；private11435执行GPU推理 |
| 12 | `RAGPipeline.parse_output:165` / FullWorkbench | JSON类型、可见引用、实际prompt token相符；异常记error，不伪造答案 |
| 13 | `RecordStore.add` / UI显示 | 应用记录独立落盘；返回answer/citations/abstain、实际证据、耗时、覆盖和record_id |

注意：当前FullWorkbench直接复用Pipeline的静态解析器，**没有调用旧Pipeline.answer来包办检索与缓存**。把旧类架构图直接贴到当前UI链路，会错过任务打包、HTTP层和不同缓存语义。

### 14.2 当前默认清单

| 项目 | 当前真实值 | 证据 |
|---|---|---|
| 发布指针 | V3FINAL，release_status=VERIFIED | `configs/v3/default.json` |
| 默认生成模型 | Qwen3.5-2B Q4_K_M，tag `ragscope-qwen3.5:2b-q4_K_M` | GENERATION_SELECTION、original_client |
| embedding | Qwen3-Embedding-0.6B Q8_0，1024维、L2、GPU空间 | FULL_DENSE_INDEX.embedding |
| 默认检索 | hybrid；单篇普通alpha.5，复杂指定alpha.75，全库加metadata.25/doc1.0 | RETRIEVAL_SELECTION、deployment_context |
| 当前索引 | 1585论文、19673块；body mmap Dense+CompressedBM25 | FULL_DENSE_INDEX、FullCorpusRetriever |
| reference budget | 普通2048；复杂6144 | deployment.py |
| 后端窗口/输出 | 全部当前应用8192；输出512/768 | DeploymentClient、deployment_context |
| 任务规则 | fact/explanation/comparison/summary，可手选；规则主要面向英文 | deployment_kind、task_kind |
| 答案缓存 | UI默认false；启用后进程内按配置+payload精确命中 | FullWorkbench.cache |
| 查询向量缓存 | 应用允许复用；正式在线专项禁用 | ReusableEncoder / UncachedOnlineEmbedding |
| CrossEncoder / 学习路由 / LoRA | 默认均不启用 | 当前调用链不含这些执行步骤 |
| 自动下载/云API | 当前启动及回答不启用 | 本地包/模型校验、LocalLLMClient |

### 14.3 已启用不等于已优化到位

UI的“关键词检索”仍会先编码query，这是当前统一执行路径的冗余；文档没有为此修改功能。自动任务和指代规则主要识别英文，中文用户宜明确选择任务/论文；这不是已经验证完善的多语言意图模型。当前API同步串行，没有请求队列和取消协议的生产级验证。

全库可检索是**数据覆盖能力**，实际找对材料是**检索质量**，得到真实正确结论是**生成与事实质量**。1585篇完整覆盖、17.65%全库完整证据、约.4065可答F1和10%正确拒答分别回答不同问题，任何一项都不能替另一项作证明。

## 15. 关键技术决策

下列每项分别写决定、实现、原理、证据、理由来源、替代与代价及状态。历史设计动机不从代码倒推成事实。

### 15.1 结构切块与可追溯坐标

**决定/实现**：保留canonical原文坐标，当前选择StructureRecursive R512。**原理**：结构边界减少无意义切断，区间投影可跨切块配置比较证据。**证据**：chunkers与E1/E6；E6的100题2B F1 .4426→.4837。**理由来源**：选型规则在任务/冻结配置中记录；“更符合论文自然语义”只是 `[推断]`。**替代/代价**：fixed、sentence、parent-child均保留；结构启发式会误判，未实现embedding边界切块。**状态**：当前启用，局部收益而非通用最优。

### 15.2 保留Hybrid而不是看test最高分就改Dense

**决定/实现**：普通指定论文仍Hybrid alpha.5；全库使用额外先验。**原理**：词项与语义信号互补，RRF降低分数量纲冲突。**证据**：开发选型、RETRIEVAL_SELECTION；V2 test中Dense F1高于Hybrid亦保留记录。**理由来源**：`[文档/代码]` dev先选和禁止按test换默认。**替代/代价**：BM25/Dense在UI可选；Hybrid多一路编码/融合，最终文档Hit@10未全面改善。**状态**：当前默认，不称所有题最优。

### 15.3 默认2B

**决定/实现**：已有2B Q4、G0保留给普通问答。**原理**：本机质量与成本取舍，避免仅按参数量定型。**证据**：真实dev20的baseline2B优于试过的新提示/4B/9B，V2同协议矩阵亦有尺度倒挂。**理由来源**：GENERATION_SELECTION.policy_reason明确记录；根因仍 `[未知]`。**替代/代价**：已有4B/9B真实运行，但速度和显存成本较高；开发协议不完全相同，不能证明2B普遍胜出。**状态**：当前默认，可核验而非理论定律。

### 15.4 默认关闭重排

**决定/实现**：当前调用链无CrossEncoder。**原理**：候选相关排序不是完整证据组合优化。**证据**：V2 R3与V3 D_confirm覆盖/回答退步；候选证据仍在而打包丢失的漏斗已分层核验。**理由来源**：实验负结果与部署选择。**替代/代价**：BGE/Qwen重排研究留存；不排除新训练/预算算法在其他任务有益。**状态**：当前停用，非删除模块。

### 15.5 不将旧LoRA放进默认模型

**决定/实现**：Q4默认不加载adapter，也无已通过数值门槛的合并模型。**原理**：训练loss与真实推理是不同目标，未合并adapter有额外decode开销。**证据**：同BF16 100题F1 .464889→.426683，P50增至约3.62倍；新24题方向不同、合并probe失败。**理由来源**：真实质量/成本和前置条件，不是证明LoRA无效。**替代/代价**：保留adapter与恢复证据；没有新增正式LoRA训练。**状态**：研究完成若干检查，默认停用。

### 15.6 总结/比较提供更多来源材料

**决定/实现**：复杂任务最多6144参考token、章节/论文平衡、768输出。**原理**：问题需跨段信息时，仅最相似局部段落不足。**证据**：20题标注关键点上下文覆盖57.45%→100%，F1和语义余弦上升。**理由来源**：源覆盖诊断与开发workflow；不是凭空把窗口调大。**替代/代价**：更慢、更长，6题F1仍退步，未做人审。**状态**：当前启用，小样本固定任务验证。

### 15.7 分清单篇范围与全库定位

**决定/实现**：用户选范围就严格限制rows；全库才算文档先验。**原理**：避免用其他论文回答“这篇”，也避免把已知doc_id的题与未知论文检索混评。**证据**：search与API输入场景、全库1200对照。**理由来源**：scope显式设计与数据标注语义。**替代/代价**：自动消歧仅有限规则；全库问题仍很难。**状态**：默认启用，非完整会话实体解析。

### 15.8 多环境与本地模型

**决定/实现**：core/models/train/reports及评价兼容环境分离，本机loopback推理，D盘资产。**原理**：轻UI与重模型依赖分开，避免CPU/不同Transformers冲突；本地模型无需付费云调用。**证据**：pyvenv.cfg、锁文件、.pth、runtime_env与LocalLLMClient。**理由来源**：本地免费与D盘由任务书明确要求；将版本冲突作为拆环境收益有直接版本差证据，最初全部选型动机仍不完整。**替代/代价**：单大环境、容器、远程API属于通用替代，未做公平基准；metrics环境共享.pth不具备全物理隔离。**状态**：实际采用，不改系统Python。

### 15.9 固定8K驻留与保留失败

**决定/实现**：当前生成窗口统一8192；每次失败和精确请求复用都可追踪。**原理**：窗口切换会改变后端驻留实例；不可通过缓存或删除失败改写性能。**证据**：load_duration、线上P95 10.132→1.988；旧响应保留。**理由来源**：明确实测故障修复。**替代/代价**：统一窗口占用更多KV预算，FINAL仍比同轮B0的1.479秒P95慢。**状态**：卡顿已修复，非全面速度胜利。

## 16. 已知问题、技术债和局限

| 问题/状态 | 证据 | 影响 | 已做处理与当前余项 |
|---|---|---|---|
| 全库准确性不足，仍存在 | Hybrid文档Hit@10 .405→.38，完整证据仅.17647 | 用户不指论文时容易定位错误或缺材料 | 完整索引/先验已做；没有证明全库高可靠 |
| 正确拒答很弱，仍存在 | H_QA不可答3/30，两方案相同 | 材料不足时仍可能勉强回答 | prompt与abstain实现存在；行为未解决 |
| 引用可见不等于支持，仍存在 | schema枚举+规则事实真值UNKNOWN | 看起来有引用的错误更难察觉 | 保存所有实际证据；未完成全面人工审核 |
| 自动事实评估不充分 | 旧claim5042/5720未知；NLI资源受限 | 不能给出可信总体事实准确率 | 增语义和字面检查，未知保留；不是完美裁判 |
| 总结比较样本小且参考非人审 | 20题、14summary/6compare、AI参考 | 不能推广成一般论文任务优胜 | 源引句/关键点校验完成；独立人评缺失 |
| 最终部署有迭代重评 | 首次矩阵后改长任务上下文/驻留窗口 | 最终成绩不是一次性盲测 | 全部版本/复用链保留，明确验证性质 |
| 学习路由未带来部署收益 | dev/test多数类选择，在线P50仅快5.04%且P95变慢 | 分类成本无法稳定换收益 | 默认固定；不能外推新语义分类目标无效 |
| LoRA机制未因果定位 | 100题负向、24题反向、merge gate失败 | 不能安全发布微调收益或合并版本 | 保存adapter/断点/数值诊断，无新增训练 |
| 多时期实现与过期文档 | README仍以v2/旧图为主；早期v3写未完成 | 新读者容易沿旧入口理解当前系统 | 本手册与LOCAL_RUN_V3明确时态，旧资料不删除 |
| 当前API本地串行、无鉴权 | stdlib HTTPServer、loopback | 不等于生产并发服务 | 本地单用户可用；并发、取消、队列未验 |
| 英文启发式与端口硬编码 | task_kind/指代guard与UI URL | 中文或换端口时自动行为有限 | 手动任务/范围可用；本轮不改功能 |
| BM25也先query编码 | FullWorkbench统一encoder调用 | 纯关键词请求仍有模型编码成本 | 已记录；不是检索器本身需要Dense |
| 索引与缓存多套并存 | Chroma/ArrayIndex/CompressedBM25、CPU/GPU identity | 学习和存储管理更复杂 | 哈希与指针阻止误混；未清理用户旧资产 |
| 8K切换重载，已修复 | 原FINAL P95 10.132→1.988 | 原先有明显卡顿 | 当前统一8192；冷启动仍有成本 |
| 多篇JSON截断，局部修复 | 应用保留1错误，限制引用后复测9场景通过 | 极长或复杂输出仍可能失败 | 不声称彻底消除，错误留账本 |
| 全面浏览器视觉未验证 | 原问答页面权限拒绝 | 组件成功不等于浏览器布局完美 | 组件/API替代已过；本手册HTML另做文档渲染核验 |

模型预训练污染、语义近重复遗漏、少数GPU短峰未被采样捕捉、未暴露的实际offload层数都属于证据边界。`RequestResourceSampler`每0.5—1秒采样显卡总量和自有进程RSS；显卡总量包含其他进程、RSS可能重复计共享页，不能用这组数据倒推出精确逐请求VRAM峰值。

可取的项目叙述是“我把各层指标、错误与默认配置分清楚，并据实保留无效方向”；不应把负结果都包装成胜利，也不应把工程交付归零。面试应同时讲出**实现了什么、验证到什么、仍无法证明什么**。
