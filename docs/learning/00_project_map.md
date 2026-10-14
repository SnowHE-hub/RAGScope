# RAGScope 项目地图

## 最终运行结果（2026-10-14）

主矩阵已完整执行5,520条，39个分析组，32条解析失败全部保留；E2的2,400条与E3的1,200条均完成且无请求错误。四份正式子报告位于`reports/v2/`，最终汇总见`FINAL_REPORT.md`。下文有时间标记的准备状态保留为历史，不能代替本段完成事实。

同BF16的LoRA前后各100题和真实在线60题、6次预热均完成。LoRA降低了答案文字重合F1并增加了延迟；规则路由提速5.04%，未达10%部署门槛，因此应用继续使用dev预选的2B、固定Hybrid与neighbor。最终浏览器实际完成两次模型问答、引用展开、资源页与历史回放；这两题是功能验证，不是质量benchmark。所有人工语义审核仍为`NOT_REVIEWED`。

模型与本项目CPU/GPU/UI服务均已释放。原训练由用户自行续跑，见`RECOVERY.md`；本项目不自动重启原训练。启动应用的准确命令见项目根目录`LOCAL_RUN_V2.md`。

RAGScope v2 把问答拆成可以独立检查的四步：找资料、组织上下文、生成答案、核对结果。项目仍使用 Python 包和 Streamlit，本次没有另建一套网站后端。

旧入口 `scripts/build_indexes.py`、`ragscope/generation/pipeline.py` 与 `ui/chat_app.py` 保留。旧实现会在打开界面时加载 embedding 和重排模型；界面默认使用 Mock，模型连接失败也会退回 Mock。v2 打开页面只读资源配置，模型不就绪时明确说明；仅在点击“运行”后使用已经核验的本地资源。

数据流程：

```text
公开原始数据 → 结构保留清洗 → Document 与稳定 EvidenceUnit
→ 带 canonical 坐标的 Chunk → 带身份签名的索引
→ 指定 scope 的排序结果 → ContextBundle
→ 固定 prompt + 本地模型 → 原始输出 / 解析结果 / 错误
→ 每题证据与答案指标 → 不可变 JSONL + 配置与记录哈希 → 汇总与报告
```

`ragscope/schema.py` 定义结构。EvidenceUnit 是切块前已确定的原文段落，Chunk 是某一种切分方案的片段，因此改变块大小会改变 Chunk，但不应改掉金标准证据单位。

`ragscope/ingestion/` 负责文本与证据位置，`ragscope/indexing/array_index.py` 存储可校验的索引，`ragscope/retrieval/` 排序候选。`ragscope/generation/context.py` 决定实际发给模型哪些文字；`pipeline.py` 保存实际 prompt、原始输出和解析错误；`ragscope/evaluation/evidence.py`、`generation_metrics.py`、`v2.py` 分别计算证据、答案重合与统计。

`ragscope/experiments/ledger.py` 是运行账本。每个正式配置都有哈希，每题写完马上保存；中断后只恢复未完成部分。模型、语料、缓存与实验结果放在 `D:/RAGScope_Runtime`，代码保留在当前项目位置。`docs/execution/STATUS.md` 是当前进度，`MASTER_PLAN.md` 是任务依据，`reports/v2/ACCEPTANCE.json` 是最终逐项验收。

界面分“资料问答”“实验记录”“资源状态”。历史记录是回放，不代表重新运行后的实时耗时；检索模式只有证据，没有生成答案。配置来自 `configs/v2/ui.json`，索引经过 manifest 和内容哈希验证，模型必须已有明确版本和就绪状态。

工程验收例子见 `tests/unit/test_ui_v2.py`。AppTest 实际执行 Streamlit 组件逻辑，测试模型未就绪、论文 scope、真实片段展开和记录哈希拒绝。这里的人工微型语料只证明功能正确，不计入真实问答成绩。真实运行目录和各阶段状态以执行账本为准。

## 2026-10-14 06:06 UTC 的更新

下方04:05表格保留为历史快照。现在辅助上下文已经补齐：simple的BM25、Dense、Hybrid各200题，共600条；SQuAD的Hybrid100题也完成，均保留原问题与分母。simple复用同身份的真实查询向量，运行时零HTTP；其中三条历史缓存没有后端usage，仍为N/A。SQuAD只新增100个查询向量，1212个正文块不重编码。证据见`20261014_v2_retrieval/SIMPLE_CACHE_ONLY/HISTORICAL_NA_REUSE/RESULT.json`与`SQUAD_FINAL/EXIT_INDEPENDENT_VALIDATION.json`。这说明检索输入齐了，不等于这些条件的生成问答全部完成。

R3的有效五组现在共660条：CPU的dev100＋test200，以及GPU的global100＋ROUTE train200/dev60。后360条在用户追加GPU授权后运行；原CPU global只完成的29条仍完整保留，不与GPU100条混算。Laya也已实际得到1080条原始路由概率，另有3次预热；这些概率尚不能证明路由提高质量。线性、MLP、LSTM三种学习路由已经真实训练，七种规定策略的dev60/test100决策全部冻结；三种学习路由在dev最终都选了BM25，不能为了得到更多路线而改标签或重训。

此时主生成矩阵仍在执行，LoRA同BF16配对和真实在线路由的结论尚未签收。所有人审仍为`NOT_REVIEWED`。浏览器已实际完成第一阶段BM25查询、证据展开与资源状态检查，见`ui_candidate/BROWSER_PHASE1_QA.json`；页面服务正常请求退出后实际Windows退出码为1并已释放端口。它是第一阶段UI证据，不能代替实时模型生成和最终发布验收。

## 2026-10-14 04:05 UTC 的历史快照

读项目时可以把“数据已备好”“真实请求已执行”“整项验收通过”分开。下面是这一时点的状态，不是未来完成承诺：

| 模块 | 已经发生的事情 | 还不能据此得出的结论 |
|---|---|---|
| 主检索 | R512 全509篇、6551块完整索引已落盘；structured test/global 的 BM25/Dense/Hybrid 共900条及 ROUTE 的 BM25/Hybrid 共520条来源已核验 | 不能说所有重排、路由与生成实验都完成 |
| 辅助索引 | SQuAD1212块＋simple777块，共1989向量，全部从磁盘读回并逐项对齐缓存 | E4/E6 的最终检索上下文与问答尚未物化，预编码没有新增质量实验行 |
| R3 重排 | dev100和test200共300条已完整签收 | 其余global100、ROUTE260仍在执行，不能把300写成660 |
| E2 核心问答 | 原定2400任务的四种检索来源已冻结；按2B→4B→9B单卡串行实跑中 | FROZEN表示输入不再变，不表示2400次输出已经齐全 |
| E3 对照 | 三模型的Oracle/No Context共1200条已经封存 | 后续主清单引用原分片，不重新跑，也不把引用算成新增请求 |
| 压力与训练 | 4K/8K共22次真实调用已完成；LoRA的972例、122步训练已完成 | 压力诊断不等于全数据质量实验；LoRA的同BF16配对问答尚未执行 |

这些运行共同使用原来的10,000次生成调用总预算，预热、失败、正式评测与后续交互均按同一本账记录。索引编码的向量条数不等于生成调用次数。此时人工语义审核全部仍为 `NOT_REVIEWED`，自动hash检查与代理指标不能替代人工判断。

新入口 `scripts/configure_ui_v2.py` 把完整索引和四个固定模型包整理为 `D:/RAGScope_Runtime/runs/20261014_v2/ui_candidate/ui.json`，先供核验，不直接改正式配置。04:01的实际只读核验中，CPU embedding的11436服务已关闭，11435生成服务被E2占用且持有调用锁；因此候选虽已核验文件，神经检索和交互生成仍标为不可用。默认意图是2B、固定Hybrid、neighbor，不能因文件存在就绕过服务状态和锁，也没有启用未经在线门槛验证的自适应路由。详见 `docs/execution/UI_CANDIDATE.md`；真正GUI/实时生成验收仍由后续资源窗口完成。
