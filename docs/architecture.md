# RAGScope — 完整项目架构说明与技术证据手册

> 面向个人学习、简历准备和技术面试。先讲系统实际怎样工作，再讲每个技术为什么可能有用，以及实验究竟证明了什么。工程已交付、某个实验已完成、答案已经可靠，是三个不同的判断。

## 0. 阅读说明与证据分级

### 0.1 核验范围与版本

核验日期：2026-10-15（Australia/Sydney）。代码基线是 `codex/ragscope-v3-diagnostics`，HEAD 为 `493dbdad37cd7f372c3121b8332a055d6faa8f7a`。本手册编写期间只新增文档、图示及轻量核验工具，不训练模型、不重建索引、不启动GPU实验。本文的“当前”指这份代码与 `configs/v3/default.json` 指向的部署，不代表后台服务此刻正在运行。

项目根目录：`D:/MCS_26S1课程资料/个人项目/RAGScope`。大资产根目录：`D:/RAGScope_Runtime`。下文缩写 **RUNTIME** 指后者；**V2RUN** 指 `RUNTIME/runs/20261014_v2`；**V3EARLY** 指 `RUNTIME/runs/v3_20261015_014506`；**V3FINAL** 指 `RUNTIME/runs/v3_finish_20261015_1415`。这些缩写只为阅读方便，附录提供完整路径与核验方式。

参考材料为用户提供的 `D:/MCS_26S2课程资料/quantmind/docs/architecture.md` 与同目录HTML。借鉴的是证据标记、真实入口与历史实现的分离、决策记录和面试问答组织方式，不移植其业务事实。根目录原有 `architecture.md/html` 是2026-10-08的RAGScope V1说明，原件保留，不能当作最新状态。

| 标记 | 定义 | 使用边界 |
|---|---|---|
| `[代码]` | 当前源码、配置或可直接读取的实现事实 | 有函数不等于运行过；默认是否调用另行说明 |
| `[实测]` | 原始逐条记录、模型响应、测试产物或本次轻量复算支持 | 必须能定位数据、样本量、条件与分母；本次没有重跑模型实验 |
| `[文档]` | README、任务书、历史报告、提交消息中的说法 | 不自动升级为事实；与代码冲突时并列展示 |
| `[推断]` | 根据实现与原理提出的解释 | 不写成当时的设计动机或已验证因果 |
| `[未知]` | 当前证据不足 | 保留未知，不用合理猜测补齐 |



实验章节使用的额外路径简称（均为读取既有结果）：

| 简称 | 完整路径 |
|---|---|
| V2 | `D:/RAGScope_Runtime/runs/20261014_v2` |
| L0 | `D:/RAGScope_Runtime/runs/20261014_v2_legacy_reconstructed` |
| R2 | `D:/RAGScope_Runtime/runs/20261014_v2_retrieval` |
| G2 | `D:/RAGScope_Runtime/runs/20261014_v2_generation` |
| G2E2 | `D:/RAGScope_Runtime/runs/20261014_v2_generation_e2` |
| G2E3 | `D:/RAGScope_Runtime/runs/20261014_v2_generation_e3` |
| LORA | `D:/RAGScope_Runtime/runs/20261014_v2_lora` |
| TRAIN / T2 | `D:/RAGScope_Runtime/runs/20261014_v2_lora/train_2b_1024_seed42` |
| ROUTE | `D:/RAGScope_Runtime/runs/20261014_v2_routing` |
| LAYA | `D:/RAGScope_Runtime/runs/20261014_v2_laya` |
| V3D / V3EARLY | `D:/RAGScope_Runtime/runs/v3_20261015_014506` |
| V3F / V3FINAL | `D:/RAGScope_Runtime/runs/v3_finish_20261015_1415` |

### 0.2 怎样使用这本手册

初次阅读按第1、4、14、12、16章建立全貌；遇到术语回第3章；准备面试时用第17章逐题自测。代码引用采用“相对路径:行号 + 符号”；行号绑定本次HEAD，未来提交后可能移动。实验结论以原始结果路径为依据，而非聊天记忆。

能力状态采用七种明确口径：**已实现 / 单测通过 / 真实模型运行 / 曾用于实验 / 当前默认启用 / 历史或研究支线 / 未完成或未知**。例如LoRA属于“已实现、实训、有对照、默认停用”，并不是“从未做过”，也不是“当前回答都经过微调”。

### 0.3 当前工程规模

`[实测]` 本次用 `git ls-files` 统计HEAD受控文件，逐文件按物理行计数；包含空行和注释，不含环境、模型、缓存、未跟踪的个人资料以及本手册新增辅助脚本。

| 范围 | Python文件 | 物理行 |
|---|---:|---:|
| `ragscope/` | 120 | 14732 |
| `scripts/` | 186 | 19631 |
| `tests/` | 90 | 8459 |
| `ui/` | 6 | 1010 |
| `reports/`内Python | 3 | 443 |
| 合计 | **405** | **44275** |

HEAD有84次提交、共572个受控文件。测试源码含487个静态 `test_` 函数定义，这不是参数化展开后的用例数，也不是新跑了487项。最后一轮v3定向回归产物是82项通过；V2有独立的更大回归记录，不能相加宣传为一次全量测试。证据与复查脚本：`docs/_architecture/repository_snapshot.json`、`docs/_architecture/audit_repository.py`。

## 1. RAGScope 是什么

### 1.1 通俗版介绍

RAGScope是一套在自己电脑上运行的论文阅读与问答实验系统。它先把论文整理成能追溯到原文的小段，收到问题后找出相关段落，再让本地语言模型据此作答，同时把引用原文展示出来。它也保存实验中每道题真正看到了什么、模型回答了什么、花了多久，方便判断问题出在检索、上下文还是生成。

适合的输入是已有论文库中的问题，例如“这篇论文用了哪些数据集”“方法A和B有什么区别”“概括这篇论文的贡献和局限”。输出包含回答、引用ID、是否拒答、实际上下文、耗时和记录编号。它不是默认联网的资料搜索器，也不是已经验证可靠的学术事实裁判。

### 1.2 工程师版介绍

这是从短文本检索benchmark演进为长论文RAG与本地实验平台的Python项目：QASPER论文规范化后保持Unicode原文坐标，结构切块形成R512语料；文档层BM25与Dense、块层压缩BM25与1024维归一化向量共同参与检索；指定范围约束与任务工作流决定证据打包；私有Ollama后端执行量化Qwen生成，JSON schema约束可见引用，逐条JSONL账本保存输入、响应、配置身份和错误。`[代码]` `ragscope/v3/full_retrieval.py:11`、`ragscope/v3/full_workbench.py:34`、`ragscope/generation/context.py:97`。

与传统搜索相比，它多了“根据证据组织答案”；与普通LLM聊天相比，它多了外部语料与可检查的引用；与一次性RAG演示相比，它增加了冻结数据、同题对照、失败分母、模型身份和恢复记录。后者是可复核性上的工程成果，不直接等于回答更准确。

### 1.3 当前能力与未具备能力

| 能力 | 真实状态 | 主要证据 |
|---|---|---|
| 全库/指定论文/多篇范围检索 | 已实现、真实运行、默认可用 | `FullCorpusRetriever.search`；V3FINAL的 `API_REAL_ACCEPTANCE.json` |
| 当前目标论文覆盖 | 原509篇缺失0；完整库1585篇/19673块 | `TARGET_509_COVERAGE.json`、`FULL_DENSE_INDEX.json` |
| 问答、解释、总结、比较 | 已实现；有真实生成与小规模任务验证 | `deployment_context`、最终220条回答 |
| 引用原文、上下文和历史回放 | 当前工作台实现且组件/API实测 | `ui/full_workbench.py`、`UI_REAL_COMPONENT_INTEGRATION.json` |
| 任意PDF上传、OCR、表格图像理解 | 当前默认入口没有这些能力 | UI无上传入口；QASPER以结构化文本为主 |
| 自动事实正确率、可靠拒答 | 未解决；拒答实测偏弱 | 正确拒答3/30；规则事实真值仍UNKNOWN |
| 企业级并发、鉴权、多租户、部署监控 | 本轮未实现/未验证 | 本地同步HTTPServer；`observability/`仍为空包 |
| LoRA、学习路由、重排 | 研究实现与历史实验保留，当前默认不启用 | 第10—12章 |

`[实测]` 这里的“可运行”有模型请求和组件集成证据；最终v3浏览器视觉验收因权限拒绝未完成，不以组件测试冒称浏览器截图验收。

## 2. V1—V3 演进历程

<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 800" role="img" aria-labelledby="ragscope-05-evolution-title ragscope-05-evolution-desc">
<title id="ragscope-05-evolution-title">V1 → V2 → V3：工程演进与能力边界</title>
<desc id="ragscope-05-evolution-desc">时间线区分工程扩展和测得质量；不同数据与指标不能跨版本直接比较。</desc>
<defs><marker id="ragscope-05-evolution-arrow" markerWidth="9" markerHeight="9" refX="8" refY="4.5" orient="auto"><path d="M0,0 L9,4.5 L0,9 Z" fill="#77776f"/></marker></defs>
<rect width="100%" height="100%" rx="16" fill="#fdfcf9"/>
<text x="30" y="38" fill="#252522" font-size="24" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="700" text-anchor="start">V1 → V2 → V3：工程演进与能力边界</text>
<rect x="35" y="100" width="340" height="300" rx="10" fill="#f0f0ee" stroke="#74746b" stroke-width="1.6" stroke-dasharray="7 5"/>
<text x="50" y="129" fill="#74746b" font-size="19" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="700" text-anchor="start">V1：检索研究</text>
<text x="50" y="155" fill="#252522" font-size="15" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="400" text-anchor="start">SQuAD / arXiv 历史实验</text>
<text x="50" y="178" fill="#252522" font-size="15" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="400" text-anchor="start">短文本 chunking 对照</text>
<text x="50" y="201" fill="#252522" font-size="15" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="400" text-anchor="start">BM25 / Dense / Hybrid</text>
<text x="50" y="224" fill="#252522" font-size="15" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="400" text-anchor="start">BGE / Chroma / Cross-Encoder</text>
<text x="50" y="247" fill="#252522" font-size="15" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="400" text-anchor="start">检索指标：Recall / MRR / nDCG</text>
<text x="50" y="270" fill="#252522" font-size="15" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="400" text-anchor="start">高召回不等于生成高 F1</text>
<text x="50" y="293" fill="#252522" font-size="15" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="400" text-anchor="start">旧实现与报告继续保留</text>
<rect x="430" y="100" width="340" height="300" rx="10" fill="#f0f0ee" stroke="#74746b" stroke-width="1.6" stroke-dasharray="7 5"/>
<text x="445" y="129" fill="#74746b" font-size="19" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="700" text-anchor="start">V2：长论文与生成研究</text>
<text x="445" y="155" fill="#252522" font-size="15" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="400" text-anchor="start">长文本、结构切块与证据坐标</text>
<text x="445" y="178" fill="#252522" font-size="15" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="400" text-anchor="start">真实本地生成模型对照</text>
<text x="445" y="201" fill="#252522" font-size="15" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="400" text-anchor="start">检索、生成、路由、LoRA 实验</text>
<text x="445" y="224" fill="#252522" font-size="15" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="400" text-anchor="start">模型身份与配置哈希</text>
<text x="445" y="247" fill="#252522" font-size="15" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="400" text-anchor="start">冻结题目 / 失败保留 / 断点</text>
<text x="445" y="270" fill="#252522" font-size="15" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="400" text-anchor="start">更多模块未自动带来更高质量</text>
<text x="445" y="293" fill="#252522" font-size="15" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="400" text-anchor="start">负结果构成已有证据</text>
<rect x="825" y="100" width="340" height="300" rx="10" fill="#edf7f2" stroke="#257455" stroke-width="1.6"/>
<text x="840" y="129" fill="#257455" font-size="19" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="700" text-anchor="start">V3：全库与任务工作台</text>
<text x="840" y="155" fill="#252522" font-size="15" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="400" text-anchor="start">1585 篇 / 19673 块完整 Dense</text>
<text x="840" y="178" fill="#252522" font-size="15" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="400" text-anchor="start">范围检索、全库文档先验</text>
<text x="840" y="201" fill="#252522" font-size="15" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="400" text-anchor="start">总结比较的章节与分组证据</text>
<text x="840" y="224" fill="#252522" font-size="15" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="400" text-anchor="start">统一 8K 驻留窗口，修复重载</text>
<text x="840" y="247" fill="#252522" font-size="15" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="400" text-anchor="start">真实问答 / 全库 / 在线测试</text>
<text x="840" y="270" fill="#252522" font-size="15" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="400" text-anchor="start">保留 2B；停用无充分收益组件</text>
<text x="840" y="293" fill="#252522" font-size="15" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="400" text-anchor="start">完整本地 UI / API / 结果追溯</text>
<polyline points="375,245 430,245" fill="none" stroke="#77776f" stroke-width="1.8" marker-end="url(#ragscope-05-evolution-arrow)"/>
<polyline points="770,245 825,245" fill="none" stroke="#77776f" stroke-width="1.8" marker-end="url(#ragscope-05-evolution-arrow)"/>
<rect x="115" y="500" width="970" height="130" rx="10" fill="#fff" stroke="#a6a297" stroke-width="1.6"/>
<text x="130" y="529" fill="#a6a297" font-size="19" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="700" text-anchor="start">可证明的最终范围</text>
<text x="130" y="555" fill="#252522" font-size="15" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="400" text-anchor="start">同批题、同评测条件：普通问答 F1 0.3904 → 0.4065；任务 F1 0.4274 → 0.4582</text>
<text x="130" y="578" fill="#252522" font-size="15" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="400" text-anchor="start">全库 Hybrid 完整证据覆盖 11.18% → 17.65%，但文档命中 40.5% → 38.0%</text>
<text x="130" y="601" fill="#252522" font-size="15" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="400" text-anchor="start">不可答正确拒答仍为 3/30；语义相似度、引用合法性不能替代人工事实判定</text>
<rect x="30" y="700" width="250" height="40" rx="10" fill="#edf7f2" stroke="#257455" stroke-width="1.6"/>
<text x="45" y="729" fill="#257455" font-size="15" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="700" text-anchor="start">当前默认链路</text>
<rect x="300" y="700" width="250" height="40" rx="10" fill="#eff2fc" stroke="#534ab7" stroke-width="1.6"/>
<text x="315" y="729" fill="#534ab7" font-size="15" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="700" text-anchor="start">共享数据 / 索引</text>
<rect x="600" y="700" width="250" height="40" rx="10" fill="#fff7e8" stroke="#a86a19" stroke-width="1.6" stroke-dasharray="7 5"/>
<text x="615" y="729" fill="#a86a19" font-size="15" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="700" text-anchor="start">独立实验 / 非默认</text>
<rect x="900" y="700" width="250" height="40" rx="10" fill="#f0f0ee" stroke="#74746b" stroke-width="1.6" stroke-dasharray="7 5"/>
<text x="915" y="729" fill="#74746b" font-size="15" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="700" text-anchor="start">历史实现 / 非默认</text>
</svg>

### 2.1 V1：先回答“能不能把原段落找回来”

V1的语料是SQuAD段落和arXiv摘要，数据清洗后约22029篇，fixed512/overlap64产出22054块；检索使用BGE-small、Chroma、BM25、加权RRF与BGE重排。主要评估是200道检索题上的文档Recall/MRR/nDCG，不是最终答案正确率。`[文档]` 根目录旧架构说明；`[实测]` 历史CSV与E0重建分别见第12章，不将两者视为相同模型版本。

原文大多短于切块阈值，几乎一篇对应一块，导致切块比较很难说明长文策略效果。生成客户端与UI当时已有，但“有生成链路代码”不意味着当时已经做过生成质量实验。

### 2.2 V2：把数据、上下文与生成评估补起来

V2转向QASPER长论文：选定509篇、34615证据单元，冻结train1000/dev100/test200和独立全库题；加入规范化坐标、结构/父子切块、Qwen embedding、本地2B/4B/9B及LFM模型、冻结上下文、主实验矩阵、路由与LoRA研究。实验记录开始保存模型权重、实际token、错误、耗时和资源身份。

主矩阵5520条、39组的规模代表**问题被系统性测试了**，不是5520条都正确：32条生成失败保留，LoRA与路由未达到默认启用要求。V2的主要进步是把“检索到了论文”继续拆成“是否包含完整证据、模型是否利用证据、输出格式是否可解析”。`[实测]` 第12章及 `docs/_architecture/experiment_audit.json`。

### 2.3 V3早期诊断与V3最终交付是两个时点

V3EARLY研究了提示协议、证据原子化、NLI/claim评估、重排、分层检索与任务工作流。部分向量只完成141篇，因此早期报告限制功能、封存H_QA/H_tasks。它是当时真实状态，不能拿来否定后续工作，也不能被后续“已完成”覆盖掉。

V3FINAL复用了已有向量，补齐到1585篇/19673块，完成H_QA200/H_tasks20与全库/在线对照，发布新的FullWorkbench。默认仍保留开发集上更稳妥的2B、普通问答旧G0策略；总结和比较采用更广证据，模型后端统一8K以避免切换重载。`[代码]` `configs/v3/default.json`；`[实测]` V3FINAL的原始评分、模型响应、索引与验收产物。

### 2.4 为什么加了技术，质量却没有成比例提高

首先题目变难、指标变了：短段落“找原文”不能直接与长论文“写答案”比较。其次检索、打包和生成是串联环节；任何一环损失关键证据都会限制总效果。更完整的上下文也可能引入干扰或让小模型混淆，更多参数没有自动修复任务适配问题。`[推断]` 这些是机制解释，不是单一因果的实验定论。

能直接证明的是：大模型、新提示、重排、LoRA和学习路由在已测条件下并未一致获益；任务型证据组织在小规模总结/比较题上改善较明显。工程规模从旧版约3800行增至本次受控Python44275行，主要新增了实验、验证和运行约束。**代码更多是工程覆盖扩大；答案能力改善必须另看同题实验。**

## 3. 完整术语与技术原理

本章先解释通用原理，再注明项目落点。理论没有统一标成项目实现；未采用的技术明确写出。一般RAG与低秩适配原理参见 [RAG原论文](https://arxiv.org/abs/2005.11401)、[LoRA原论文](https://arxiv.org/abs/2106.09685)、[QLoRA原论文](https://arxiv.org/abs/2305.14314)。这些论文不证明RAGScope自己的效果。

### 3.1 数据、切块与表示

**检索增强生成（Retrieval-Augmented Generation，RAG）**：像开卷考试，先找材料再回答。外部检索补充模型参数之外的信息，但模型仍可能误读材料。项目当前由 `FullWorkbench.answer` 组合检索与生成，而非直接调用旧 `RAGPipeline.answer` 全流程；旧Pipeline的解析器仍复用。`[代码]` `ragscope/v3/full_workbench.py:50`。

**语料库（Corpus）**：可供搜索的资料集合。例子是1585篇QASPER论文。它是检索范围，不是“1585篇都拿去训练过”。项目存于 `RUNTIME/corpora/20261014_v2/all_documents.jsonl`。

**文档（Document）**：具有稳定身份的一篇资料。V1“一段SQuAD”也叫文档；V2/V3“一篇长论文”才是文档，因此两个版本的文档数不能直接比规模。项目 `ragscope/schema.py:38` 保存文本、来源revision、sections、checksum与split。

**文本块（Chunk）**：用于索引与拼上下文的一段原文。一个长论文可变成十几个块；块越小检索越精细，但关键前提更容易散落。`ragscope/ingestion/chunkers.py` 生成块ID、前后邻接、原文坐标与证据投影。

**词元（Token）**：模型读写文本的计量单位，不等于词或汉字。一个生僻词可能拆成多个token；“512字符”与“512token”不同。项目切块注入固定的Qwen tokenizer，打包另用 `cl100k_base` 参考预算，再由生成模型模板核对实际输入。

**分词器（Tokenizer）**：把字符串与token ID互相转换的规则和词表。BPE根据常见片段编码，并不保证把每个单词视为一个token。`TokenizerAdapter` 不带真实后端时明确退为Unicode码点参考，不可把这种离线测试冒充真实模型token计数。`[代码]` `chunkers.py:17`、`local_embedding.py:15`。

**块大小（Chunk Size）**：每块token上限，不是固定物理长度。R512中的512是Qwen切块预算，实际较短段落可更少。结构块可能因句子或段落边界提早结束；不是每块都填满512。

**重叠（Overlap）**：相邻块共享一小段，以避免一句关键信息落在边界。例子为fixed512/64，理想步长约448token，但实现按真实文本边界重计，不能简单靠token位置倒推字符。项目记录实际重叠字符/token，打包时再减去重复原文区间。

**向量表示（Embedding）**：模型把一段文字编码成一串浮点数，使语义相关文本在向量空间较近。例如“用了什么数据集”可以接近描述评估数据的段落，而不必逐字相同。当前Qwen embedding输出1024维并L2归一化；它不负责写自然语言答案。`ReusableEncoder.embed`。

**向量（Vector）与余弦相似度（Cosine Similarity）**：向量是数值坐标；余弦 `q·d/(||q||||d||)` 比较方向。归一化后点积就等于余弦。高相似只表示表征接近，不能证明某个数字、否定或比较方向正确。当前块检索用NumPy点积；答案语义评价也用余弦，但输入与用途不同。

### 3.2 检索与排序

**稠密检索（Dense Retrieval）**：查询和文本分别编码，再搜索相似向量。好处是能跨表述匹配；弱点是可能把主题相似但不能作答的段落排前面。当前 `FullCorpusRetriever.search` 对指定行做精确矩阵乘，不是Chroma近似最近邻。

**稀疏检索（Sparse Retrieval）**：主要用词项出现情况匹配，向量维度对应词表而多数位置为零。例如精确模型名常能被关键词检索找到。当前块BM25预计算为SciPy稀疏矩阵。`ragscope/v3/compressed_index.py:12`。

**词频逆文档频率（TF-IDF）**：词在本文常见、在全库少见时权重更高。它解释了关键词检索的基本思想；项目BM25使用其相关思想，但默认不是另有一个独立TF-IDF检索器。不能把路由的文本特征与正文BM25混为同一模块。

**最佳匹配25（BM25）**：将词频贡献做饱和，并按文档长度归一。词重复100次不应比10次重要十倍；长文也不应因词多而天然获胜。常见形式 `Σ IDF(t)·f(t,d)(k1+1)/(f(t,d)+k1(1-b+b·|d|/avgdl))`。项目压缩实现 `k1=1.5,b=.75`，负IDF按均值修正，查询重复词也会加权；不是现代神经模型。

**混合检索（Hybrid Retrieval）**：把稀疏与Dense两路互补信号结合。名字像型号的查询可能靠BM25，换一种说法的问题可能靠Dense；融合不保证每题都更好。当前默认方法hybrid、普通范围问答alpha=.5，复杂指定任务alpha=.75。

**倒数排名融合（Reciprocal Rank Fusion，RRF）**：不用比较两个检索器量纲不同的分数，只看名次。项目使用 `score(c)=Σ w_i/(60+rank_i(c))`，rank从1开始；alpha是Dense路权重。某块两路都前列会累加得分。60是当前实现常数，不是经过本轮单独寻优的最佳值。`full_retrieval.py` 与 `hybrid_retriever.py` 存在不同实现时期。

**交叉编码器（Cross-Encoder）**：把问题和候选文本一起输入模型，允许词之间交互后给相关性分。它不能像双塔一样预先给所有文档算完最终分，通常只处理少量候选。项目有BGE重排研究，但当前默认问答不调用它。`ragscope/retrieval/reranker.py`、`verified_reranker.py`。

**重排器（Reranker）**：对已召回候选重新排序。它无法找回候选池里根本不存在的证据；打分提升不等于整组证据覆盖提升。项目重排曾损失互补证据，默认关闭；原因的实测与推断在第8、12章分开。

**向量数据库（Vector Database）**：保存向量、元数据并提供近邻查询的系统，如Chroma。V1使用持久化Chroma/cosine；V2/V3当前主要使用 `.npy` /mmap和BM25文件。NumPy矩阵不是伪装成数据库的分布式服务，也没有因此获得事务、权限和在线扩容能力。

**父子检索（Parent-Child Retrieval）**：用小子块定位，返回更大父块补上下文。例子为256token子块指向1024token父块。项目实现并做过长文研究，但当前R512默认没有进入父块替换路径。`ParentChildChunker`、`pack_context(expansion='parent')`。

**召回率（Recall@k）**：前k结果找回了多少标注相关对象。若每题只有一个相关文档，它等于是否命中的HitRate；若需要多个证据则不是。项目既有文档Recall，也有证据区间覆盖，必须说明对象粒度。

**命中率（HitRate@k）**：前k中是否至少找到一个相关对象，再对题目取平均。找到论文并不意味着找全论文中的证明段落；当前global `document_hit_at10` 与完整证据覆盖是两列不同指标。

**平均倒数排名（MRR）**：每题取首个相关结果名次倒数，例如第2名是.5，再平均。只奖励第一个命中，不评价后面是否含完整证据。项目 `ragscope/evaluation/retrieval_metrics.py`。

**归一化折损累计增益（nDCG）**：靠前的相关结果收益较高，用理想排序归一化。可以支持分级相关性，但旧检索实验使用二元相关，不能说已经训练了分级相关模型。代码同上。

### 3.3 模型、上下文与训练

**上下文窗口（Context Window）**：一次推理可处理的总token空间，包含聊天模板、问题、证据和输出预留。8K后端窗口不等于给了8K原文；当前事实问答仅2048参考证据token，总结/比较最多6144，输出另限512/768。

**提示词（Prompt）与聊天模板（Chat Template）**：Prompt是系统要求、问题、证据等内容；chat template把角色和内容包装成模型训练时使用的格式。内容相同但模板不同，token与行为都可能变。项目按固定tokenizer/template计数，保存模板哈希，不把Python字符串长度当模型输入长度。

**大语言模型（LLM）与小语言模型（SLM）**：都是生成或理解序列的模型，SLM没有跨行业统一参数阈值。此处2B表示约20亿参数的生成模型；embedding .6B虽也基于语言模型，部署职责是编码而非回答。默认模型名与量化身份见第9章。

**Transformer**：用注意力等层处理序列的网络架构。检索双塔、交叉重排器与生成器都可使用Transformer，但“同属Transformer”不表示权重、输入或训练目标相同。当前项目使用现成模型，没有从零训练一个基础Transformer。

**循环网络（RNN）、长短期记忆网络（LSTM）**：顺序更新隐藏状态，LSTM用门控制记忆流动。项目LSTM是路由研究中的小分类器，不是给论文生成答案的默认模型；不能说已经用LSTM做了高质量全文理解。

**量化（Quantization）**：用更低位数近似权重以节省存储和带宽，通常会有量化误差。Q4_K_M、Q8_0是GGUF量化方案，不是“所有运行数据都是4位/8位”；激活、缓存和临时工作空间另占内存。推理量化与训练精度也不能混说。

**微调（Fine-tuning）**：在已有模型上训练，使其更适配某种输出或任务。它能学格式与行为，不会自动获得检索不到的事实；目标数据过窄还可能损伤泛化。本项目微调是有边界的独立研究，不在用户每次提问时发生。

**低秩适配（LoRA）**：冻结基础权重W，只学低秩增量，典型形式 `W'=W+(alpha/r)BA`。若W为m×n，更新参数约r(m+n)，远少于mn。可训练小adapter，但仍需前向/反向激活内存。项目实际使用BF16底座+r8 adapter，详见第11章。

**量化低秩适配（QLoRA）**：把冻结底座低位量化，再训练LoRA等适配参数；原方案包括NF4、双重量化和分页优化器等机制。项目已有GGUF推理量化加BF16 LoRA实验，**不能因此称为已完成QLoRA训练**。通用原理见本章开头原论文。

**路由（Routing）**：根据问题或检索特征决定后续处理方式。历史研究选择BM25/Hybrid/Rerank；当前规则选择fact/explanation/comparison/summary和证据组织。一个是检索策略分类，一个是任务工作流，不是同一训练目标。

**神谕上下文（Oracle Context）**：评估时直接给模型标注证据，用来问“材料给对了，生成还有多大问题”。它会使用仅用于评估的答案证据，不能部署成真实检索，也不能把Oracle成绩当普通用户效果。项目有相同预算Oracle与补噪声研究。

### 3.4 质量、实验与运行

**自然语言推断（NLI）**：判断“证据前提”是否蕴含、矛盾或无法判断某个主张。它比文字重合更接近支持关系，但受模型、长证据截断、否定与领域适配影响。项目NLI支线大量UNKNOWN，最终实用评分没有把它当全知裁判。

**精确匹配（Exact Match，EM）**：规范化后答案完全相同才给1。适合简短标准答案；自由总结用不同措辞容易得0。项目对合法多参考取最佳匹配，错误输出留分母。

**词元F1（Token F1）**：按答案规范化后的词项多重集合计算precision与recall，再取调和平均。这里“词元”并非Qwen分词器token；它衡量答案和参考的词项重合，不能读成“.4065就是40.65%题目正确”。`generation_metrics.py`。

**最长公共子序列指标（ROUGE-L）**：利用预测和参考的最长公共子序列衡量内容及顺序重合；跳过部分词也能匹配。它对复述仍敏感，无法判断“高于/低于”写反是否造成严重事实错误。

**语义相似度（Semantic Similarity）**：用表征比较是否说了相近意思。最终项目用Qwen embedding对答案/参考做归一化余弦，不使用检索query前缀；这不是BERTScore，也不是模型判定的事实准确率。高相似的“没有提升”和“有提升”仍可能含相反含义。

**忠实性（Faithfulness）**：答案中的主张是否由提供的材料支持。它不同于答案是否符合外部真实世界，也不同于引用ID是否合法。项目自动规则只能给提示或UNKNOWN，尚无充分人工审核支持一个总体忠实率。

**幻觉（Hallucination）**：生成了材料不支持或错误的内容。引用真实段落也可能幻觉；“回答看起来流畅”“JSON合法”都不能排除。项目保留raw输出、引用与上下文，便于查证。

**证据覆盖（Evidence Coverage）**：本项目把输入上下文与标注原文区间求交，测覆盖比例或是否完整覆盖一组合法参考证据。它是“材料有没有给够”，不是“模型是否用对”。一题可能有多组合法证据，代码保留并采用合法集合聚合，不能把不同注释随意拼成唯一真值。

**自助法（Bootstrap）**：从观测样本中重复重采样，估计统计量的变化范围。论文内多题相关，所以V2按论文聚类抽样，而不是假装每题完全独立。2000次抽样不能弥补原始论文数量少或测试集被反复使用的问题。`legacy_statistics.py`、`analysis_report.py`。

**中位延迟P50 / 尾延迟P95**：P50是排序后的中间位置，P95描述较慢请求的分位。P95不是最大值，也不是未来95%请求的保证；几十条样本的尾部分位不稳定。项目分别记录热请求、冷加载与完整任务，未测TTFT就明确留空。

**缓存（Cache）**：复用已算结果。查询向量缓存省编码、答案缓存省整次生成、模型驻留省加载，是三个不同层次。当前UI答案缓存默认关闭；query向量可复用身份一致的ArrayCache；正式在线实验禁用两种结果缓存。哈希命中不是语义缓存。

**GPU / 显存（VRAM） / KV Cache**：GPU负责大量并行张量计算，显存同时容纳权重、KV缓存与工作空间；KV缓存保存自回归生成的键和值，窗口或批量增大通常会增加占用。项目是单卡8GB，单生成模型串行，不意味着每次必须卸载较小的编码器，也不意味着全部8GB都能留给权重。

**配置哈希（Configuration Hash）、断点（Checkpoint）、账本（Ledger）**：分别回答“是不是同一配置”“中断后从哪里继续”“每次尝试实际发生了什么”。哈希可以检测身份变化，但不能证明实验设计正确；断点文件存在也不等于恢复成功，需有实际恢复记录。项目同时保存索引完成位图、模型训练步数断点和逐题JSONL记录。

## 4. 系统架构总览

<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 850" role="img" aria-labelledby="ragscope-01-global-title ragscope-01-global-desc">
<title id="ragscope-01-global-title">RAGScope：当前系统与独立研究链路</title>
<desc id="ragscope-01-global-desc">实线为默认问答；虚线为独立或历史研究，不表示默认启用。</desc>
<defs><marker id="ragscope-01-global-arrow" markerWidth="9" markerHeight="9" refX="8" refY="4.5" orient="auto"><path d="M0,0 L9,4.5 L0,9 Z" fill="#77776f"/></marker></defs>
<rect width="100%" height="100%" rx="16" fill="#fdfcf9"/>
<text x="30" y="38" fill="#252522" font-size="24" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="700" text-anchor="start">RAGScope：当前系统与独立研究链路</text>
<rect x="30" y="80" width="340" height="130" rx="10" fill="#eff2fc" stroke="#534ab7" stroke-width="1.6"/>
<text x="45" y="109" fill="#534ab7" font-size="19" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="700" text-anchor="start">离线数据与切块</text>
<text x="45" y="135" fill="#252522" font-size="15" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="400" text-anchor="start">QASPER 长论文与历史数据</text>
<text x="45" y="158" fill="#252522" font-size="15" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="400" text-anchor="start">规范文本 / section / 原文坐标</text>
<text x="45" y="181" fill="#252522" font-size="15" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="400" text-anchor="start">R512 参考块与身份清单</text>
<rect x="430" y="80" width="340" height="130" rx="10" fill="#eff2fc" stroke="#534ab7" stroke-width="1.6"/>
<text x="445" y="109" fill="#534ab7" font-size="19" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="700" text-anchor="start">完整共享索引</text>
<text x="445" y="135" fill="#252522" font-size="15" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="400" text-anchor="start">1585 篇 / 19673 个参考块</text>
<text x="445" y="158" fill="#252522" font-size="15" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="400" text-anchor="start">BM25 + 1024 维 Dense</text>
<text x="445" y="181" fill="#252522" font-size="15" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="400" text-anchor="start">文档级索引 / SHA 与行顺序校验</text>
<rect x="830" y="80" width="340" height="130" rx="10" fill="#fff7e8" stroke="#a86a19" stroke-width="1.6" stroke-dasharray="7 5"/>
<text x="845" y="109" fill="#a86a19" font-size="19" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="700" text-anchor="start">实验与评测资产</text>
<text x="845" y="135" fill="#252522" font-size="15" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="400" text-anchor="start">冻结划分 / 配对基线 / 失败保留</text>
<text x="845" y="158" fill="#252522" font-size="15" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="400" text-anchor="start">检索、生成与真实在线记录</text>
<text x="845" y="181" fill="#252522" font-size="15" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="400" text-anchor="start">独立于手工工作台交互</text>
<polyline points="370,145 430,145" fill="none" stroke="#77776f" stroke-width="1.8" marker-end="url(#ragscope-01-global-arrow)"/>
<polyline points="770,145 830,145" fill="none" stroke="#77776f" stroke-width="1.8" marker-end="url(#ragscope-01-global-arrow)" stroke-dasharray="7 5"/>
<rect x="30" y="300" width="300" height="130" rx="10" fill="#edf7f2" stroke="#257455" stroke-width="1.6"/>
<text x="45" y="329" fill="#257455" font-size="19" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="700" text-anchor="start">本地 Streamlit 工作台</text>
<text x="45" y="355" fill="#252522" font-size="15" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="400" text-anchor="start">ui/full_workbench.py</text>
<text x="45" y="378" fill="#252522" font-size="15" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="400" text-anchor="start">问题 / 论文范围 / 任务</text>
<text x="45" y="401" fill="#252522" font-size="15" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="400" text-anchor="start">答案 / 引用原文 / 实验历史</text>
<rect x="420" y="300" width="350" height="130" rx="10" fill="#edf7f2" stroke="#257455" stroke-width="1.6"/>
<text x="435" y="329" fill="#257455" font-size="19" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="700" text-anchor="start">本地 API 与 FullWorkbench</text>
<text x="435" y="355" fill="#252522" font-size="15" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="400" text-anchor="start">127.0.0.1:11437</text>
<text x="435" y="378" fill="#252522" font-size="15" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="400" text-anchor="start">deployment_context + 检索</text>
<text x="435" y="401" fill="#252522" font-size="15" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="400" text-anchor="start">证据预算 / schema / 解析</text>
<rect x="860" y="300" width="310" height="130" rx="10" fill="#edf7f2" stroke="#257455" stroke-width="1.6"/>
<text x="875" y="329" fill="#257455" font-size="19" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="700" text-anchor="start">私有本地模型后端</text>
<text x="875" y="355" fill="#252522" font-size="15" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="400" text-anchor="start">127.0.0.1:11435</text>
<text x="875" y="378" fill="#252522" font-size="15" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="400" text-anchor="start">Qwen Embedding 0.6B</text>
<text x="875" y="401" fill="#252522" font-size="15" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="400" text-anchor="start">Qwen 2B 量化生成 / 8K 窗口</text>
<polyline points="330,352 420,352" fill="none" stroke="#77776f" stroke-width="1.8" marker-end="url(#ragscope-01-global-arrow)"/>
<text x="350" y="338" fill="#5e5e56" font-size="14" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="400" text-anchor="start">请求</text>
<polyline points="420,390 330,390" fill="none" stroke="#77776f" stroke-width="1.8" marker-end="url(#ragscope-01-global-arrow)"/>
<text x="338" y="418" fill="#5e5e56" font-size="14" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="400" text-anchor="start">答案与引用</text>
<polyline points="770,352 860,352" fill="none" stroke="#77776f" stroke-width="1.8" marker-end="url(#ragscope-01-global-arrow)"/>
<text x="775" y="338" fill="#5e5e56" font-size="14" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="400" text-anchor="start">编码 / 生成</text>
<polyline points="860,390 770,390" fill="none" stroke="#77776f" stroke-width="1.8" marker-end="url(#ragscope-01-global-arrow)"/>
<polyline points="595,210 595,300" fill="none" stroke="#77776f" stroke-width="1.8" marker-end="url(#ragscope-01-global-arrow)"/>
<text x="610" y="259" fill="#5e5e56" font-size="14" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="400" text-anchor="start">加载</text>
<rect x="30" y="535" width="350" height="135" rx="10" fill="#fff7e8" stroke="#a86a19" stroke-width="1.6" stroke-dasharray="7 5"/>
<text x="45" y="564" fill="#a86a19" font-size="19" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="700" text-anchor="start">BF16 LoRA 研究</text>
<text x="45" y="590" fill="#252522" font-size="15" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="400" text-anchor="start">独立训练、checkpoint、adapter</text>
<text x="45" y="613" fill="#252522" font-size="15" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="400" text-anchor="start">真实实验与负结果均保留</text>
<text x="45" y="636" fill="#252522" font-size="15" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="400" text-anchor="start">不在当前默认问答路径</text>
<rect x="425" y="535" width="350" height="135" rx="10" fill="#fff7e8" stroke="#a86a19" stroke-width="1.6" stroke-dasharray="7 5"/>
<text x="440" y="564" fill="#a86a19" font-size="19" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="700" text-anchor="start">路由研究</text>
<text x="440" y="590" fill="#252522" font-size="15" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="400" text-anchor="start">规则 / 线性 / MLP / LSTM / Laya</text>
<text x="440" y="613" fill="#252522" font-size="15" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="400" text-anchor="start">学习选择检索动作的独立研究</text>
<text x="440" y="636" fill="#252522" font-size="15" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="400" text-anchor="start">当前默认不加载学习型路由</text>
<rect x="820" y="535" width="350" height="135" rx="10" fill="#f0f0ee" stroke="#74746b" stroke-width="1.6" stroke-dasharray="7 5"/>
<text x="835" y="564" fill="#74746b" font-size="19" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="700" text-anchor="start">历史 V1 / V2 实现</text>
<text x="835" y="590" fill="#252522" font-size="15" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="400" text-anchor="start">BGE / Chroma / Cross-Encoder</text>
<text x="835" y="613" fill="#252522" font-size="15" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="400" text-anchor="start">旧入口、报告、索引仍保留</text>
<text x="835" y="636" fill="#252522" font-size="15" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="400" text-anchor="start">不能当作当前默认配置</text>
<text x="30" y="728" fill="#252522" font-size="17" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="400" text-anchor="start">默认生成：2B；默认检索：Hybrid。Cross-Encoder、LoRA、学习型路由均未在默认链路启用。</text>
<rect x="30" y="770" width="250" height="40" rx="10" fill="#edf7f2" stroke="#257455" stroke-width="1.6"/>
<text x="45" y="799" fill="#257455" font-size="15" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="700" text-anchor="start">当前默认链路</text>
<rect x="300" y="770" width="250" height="40" rx="10" fill="#eff2fc" stroke="#534ab7" stroke-width="1.6"/>
<text x="315" y="799" fill="#534ab7" font-size="15" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="700" text-anchor="start">共享数据 / 索引</text>
<rect x="600" y="770" width="250" height="40" rx="10" fill="#fff7e8" stroke="#a86a19" stroke-width="1.6" stroke-dasharray="7 5"/>
<text x="615" y="799" fill="#a86a19" font-size="15" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="700" text-anchor="start">独立实验 / 非默认</text>
<rect x="900" y="770" width="250" height="40" rx="10" fill="#f0f0ee" stroke="#74746b" stroke-width="1.6" stroke-dasharray="7 5"/>
<text x="915" y="799" fill="#74746b" font-size="15" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="700" text-anchor="start">历史实现 / 非默认</text>
</svg>

### 4.1 四条链路与共享底座

**离线构建链**：结构化数据 → canonical Document/EvidenceUnit → Chunk → BM25与向量索引。它不需要问题的参考答案来建立检索文本。

**当前问答链**：Streamlit → loopback API → 范围与任务判定 → query编码 → 检索融合 →证据打包 → 本地生成 → JSON/引用检查 → 页面与应用账本。它读取既有索引，不在点击时重建语料或训练模型。

**实验评测链**：冻结问题、上下文、模型与配置 → 逐题真实响应 → lexical/semantic/evidence/规则评价 → 配对汇总及性能记录。应用新问题与正式实验分账保存，不能让用户交互覆盖冻结成绩。

**研究支线**：LoRA训练/恢复/对照，以及规则、线性、MLP、LSTM、Laya路由；另有reranker、NLI与claim研究。它们复用语料或冻结上下文，但不是当前默认问答必经步骤。

### 4.2 包与依赖的真实职责

| 位置 | 职责 | 主要输入/输出 | 当前关系 |
|---|---|---|---|
| `ragscope/ingestion/` | 数据摄取、标准化、坐标和切块 | raw JSON→Document/Query/Chunk | 离线共享 |
| `ragscope/indexing/` | 模型身份、ArrayCache、旧Chroma与新数组索引 | 文本→向量/索引制品 | 多时期实现共存 |
| `ragscope/retrieval/` | V1/V2检索器、重排封装 | 问题→候选块 | 旧入口与研究复用 |
| `ragscope/generation/` | Prompt、context、客户端、schema、解析与资源采样 | 证据→JSON回答/错误 | 当前复用核心原语 |
| `ragscope/v3/` | 完整语料检索、任务工作流、私有后端、工作台与质量研究 | 完整索引→在线结果 | 最新部署集中于此 |
| `ragscope/evaluation/` | 检索、文本重合、证据评价 | 结果+参考→评分 | 离线调用 |
| `ragscope/experiments/` | 任务、冻结清单、预算、账本、统计和报告 | 配置→可核验实验产物 | 不等于在线服务框架 |
| `ragscope/training/` / `routing/` | LoRA与小分类器研究 | 训练/开发数据→adapter/路由策略 | 默认停用研究支线 |
| `scripts/` / `scripts/v3/` | 准备、执行、验收入口 | 参数/清单→各阶段文件 | 不能把所有脚本都串成默认流程 |
| `ui/` | 三代Streamlit页面 | 手工问题→显示 | 当前为full_workbench |

NumPy/SciPy用于矩阵与稀疏检索，rank-bm25用于部分旧索引，tokenizers/tiktoken用于不同预算，Streamlit用于页面，stdlib HTTPServer用于当前API，Ollama为独立模型进程，训练另依赖PyTorch/Transformers/PEFT。**当前API不是FastAPI**，即使旧requirements列过相关包。`[代码]` `scripts/v3/serve_full_workbench.py`。

### 4.3 持久化对象与接口边界

Document保存canonical全文；EvidenceUnit是在切块之前固定的段落证据；Chunk保存原文切片与单位投影；ContextBundle保存实际提供给生成器的片段、去重、预算、跳过原因和整体哈希；Query保存题目、范围、answerability与多参考；RecordStore把一条请求与配置身份写进追加账本。`[代码]` `ragscope/schema.py`、`generation/context.py`、`experiments/ledger.py`。

它们共享的是ID和原文坐标，不是“同一份拼好字符串到处复用”。检索前缀、输出引用和训练样本各自有用途，只有canonical原文区间可以作为证据对照。

## 5. 数据来源与数据处理

### 5.1 数据资产分层

| 数据资产 | 规模/格式 | 位置 | 用途与边界 |
|---|---|---|---|
| V1 SQuAD段落+arXiv cs.LG摘要 | 历史22029文档/22054fixed块；JSONL/Parquet | 旧 `data/` 与V2 legacy_manifest指向的快照 | 短文本检索历史，不冒充长论文生成测试 |
| QASPER v0.3全部论文 | 1585篇；`all_documents.jsonl`，106940110字节 | `RUNTIME/corpora/20261014_v2/` | 当前完整检索库的源 |
| V2选定QASPER论文 | 509篇/34615证据单元；`documents.jsonl` | 同上 | V2冻结主实验目标 |
| V2问题 | train1000/dev100/test200；另global100 | `queries_train/dev/test.jsonl`、`queries_global_test.jsonl` | 官方按论文划分基础上选样；global是复制测试题并去scope，不是新题 |
| QASPER审计与隔离 | `all_queries_audit.jsonl`、`quarantine.jsonl` | 同上 | 保留冲突、图表证据、匹配失败等，不悄悄删掉原始数据 |
| CPython技术文本与自编题 | 3页技术文档及病例JSONL | `technical_documents.jsonl`、`technical_cases.jsonl` | PSF来源技术案例；题目为agent编写、非人审基准 |
| V3开发与验证划分 | D_search120/D_confirm100/H_QA200/H_tasks20等 | V3EARLY的冻结输入与V3FINAL结果 | 与V2问题身份分开，详情第12章 |

`[实测]` 数量通过DATA_VALIDATION、原始JSONL和实验审计交叉核对，不是按README推测。QASPER全库不等于额外下载了1585个PDF；本项目使用其已有结构化全文。

旧数据的实际保留快照由 `RUNTIME/corpora/20261014_v2/legacy_manifest.json` 指向：`legacy_snapshot/data/raw/{squad_corpus.jsonl,arxiv_corpus.jsonl,train-v2.0.json}` 与 `legacy_snapshot/data/processed/{corpus.jsonl,eval_queries.jsonl}`。arXiv摘要没有这份历史评测的独立标注问答，不能把“SQuAD+arXiv语料”误说成“两类数据都做过有标签QA测试”。

### 5.2 规范化、章节与坐标

`qasper.make_document`按title→abstract→各章节标题及段落的顺序构造全文，用两个换行连接；先NFC与换行规范化，再累计每个单位的canonical字符起止。每个段落保留原section index与paragraph index。坐标是Python Unicode字符串中的半开区间 `[start,end)`，不是原JSON文件字节偏移，也不是PDF页坐标。`[代码]` `ragscope/ingestion/qasper.py:38`。

例子：规范化全文的 `[120,150)` 对应30个码点；切块后若只覆盖 `[140,170)`，共同证据是 `[140,150)`。通过区间交集可计算覆盖，而不依赖“新切块ID是否等于旧段落ID”。中文、换行、代码都必须以同一canonical字符串为坐标底座。

当前 `clean_text` 仅在显式HTML类型时解析HTML，保留普通文本/代码的结构；V1的NFKC、去标签、压空白和50—5000字符过滤留在 `legacy=True` 路径。**不能把旧清洗规则描述成当前长论文默认规则。** `ragscope/ingestion/cleaner.py`。

### 5.3 标注答案怎样映射回原文

`convert_question`按规范化后的段落文本匹配证据；唯一匹配才得到稳定unit_id。多处相同段落、浮动图表证据、找不到文本、可答/不可答注释冲突均带原因隔离。抽取答案、自由答案、yes/no与不可答分别保存，多参考答案和多套合法证据不被强行压成一个。

V2隔离1286题；原因计数可重叠，不能把各原因数简单相加当题目数。float_evidence737、missing_text_evidence564、answerability_conflict246等属于数据适用范围边界，而不是本系统“答错了这么多题”。`[实测]` `DATA_VALIDATION.json`；`[代码]` `qasper.py` 的 `convert_question`。

### 5.4 去重、划分与泄漏

精确正文哈希、规范化标题，以及bottom-128词三元组候选筛选后再算Jaccard≥.9，用于保守近重复审计。这不是穷尽语义改写重复的证明。跨split重复组隔离，同split保留规范代表但不跨结构复制证据；原始记录保留。`qasper.duplicate_groups`、`freeze_splits`。

V2选中论文按train398/dev36/test75分开，问题为1000/100/200；可答869/92/182，不可答131/8/18。V3 H_QA的170/30是另一批身份，不能误套V2的182/18。训练/开发/测试按论文隔离比随机拆问题更能防止模型见过同一篇文章的近似题。

仍存在的风险包括：基础模型预训练可能见过公开论文、题集在开发过程中被操作者暴露、AI生成任务参考带来的风格偏差、global问题去掉指向论文后变得歧义、近重复筛查不完备。`[未知]` 没有足够证据排除模型预训练污染。严格隔离项目训练数据，不能反推基础模型没有记忆。

### 5.5 当前索引文本与生成原文分离

离线 `prepare_retrieval_sources.py` 提供R512、F512_O0、P256_1024三种来源。metadata检索文本是 `Title: ... / Section: ... / canonical_body`；文档级文本是标题、摘要和章节名。添加这些前缀是为了定位论文主题，**生成时仍使用原canonical body**，不把前缀伪造为原文坐标。当前默认读取R512_body向量与body BM25，另以metadata BM25提供融合信号；并非默认把标题前缀输入每个Dense文档向量。`[代码]` `scripts/v3/prepare_retrieval_sources.py`、`full_retrieval.py`；`[实测]` `FULL_DENSE_INDEX.json.sparse_manifest`。


## 6. Chunking 模块

### 6.1 共用约束比切块器名字更重要

所有当前切块器从同一份canonical文本切片，而不是先解码token再猜回原文。`BaseChunker._assemble`生成包含文档内容身份、切块配置和 `[start,end)` 的确定性chunk_id，并保存 `doc_id`、`chunk_index`、`prev_chunk_id`、`next_chunk_id`、`parent_id`、`unit_spans` 与tokenizer身份。原文变化、tokenizer变化、参数变化都会改变相应身份。

`[代码]` `ragscope/ingestion/chunkers.py` 的 `BaseChunker._assemble`。这是证据评分、邻块扩展、断点复用与引用可追溯共同依赖的基础；不能只存一段文本和一个自增ID就宣称具备相同可复核性。

### 6.2 各种方法怎样切

| 方法 | 实际算法与参数 | 优点/代价 | 在项目中的状态 |
|---|---|---|---|
| Fixed Size | 用真实tokenizer计数，寻找不超预算的字符终点；优先空白边界；可配置重叠 | 稳定易控；可能割裂语义 | `FixedSizeChunker:175`；V1主路径、V2/V3实验 |
| Sentence Aware | 正则句界分段后合并至max_tokens；超长句退固定切分；0/1句重叠，放不下时减少重叠 | 尽量不断句；句界不是完美语言学解析 | `SentenceAwareChunker:208`；有测试与长文对照 |
| Structure Recursive | 先按空行、标题、列表、代码fence等结构分段；超长结构退句子/固定切分；再合并，overlap0 | 保留结构；不是模型理解语义后切界 | `StructureRecursiveChunker:273`；当前R512来源 |
| Semantic兼容名 | `SemanticChunker = StructureRecursiveChunker` | 兼容旧调用名 | `chunkers.py:295`；**没有相邻embedding相似度阈值切界** |
| Parent-Child | 先建结构父块，在每个父块内部切小子块，给子块写parent_id | 小块定位、大块补语境；父块可能占掉预算 | `ParentChildChunker`；独立实验，当前默认不用 |
| Neighbor扩展 | 命中后加入同文档的前后块，再去重/预算筛选 | 补前后条件；可能挤掉后续高排名证据 | `pack_context(expansion='neighbor')`；普通指定论文问答仍用 |

当前R512使用Qwen embedding tokenizer、max_tokens512、结构overlap0。V3还准备F512_O0与P256_1024（child256、overlap32、parent1024），是候选研究来源，不是三种同时运行在每次用户请求上。

### 6.3 Overlap与token预算的细节

`_fit_end`以字符边界搜索并重新计token；BPE token数并非严格可加，因此代码不宣称找到全局最大填充。`_overlap_start`保证尾部重叠不超预算；若重叠阻止向前推进则减少重叠并记事件。Sentence方式的重叠单位是一句，不应写成“64token”。

检索块大小使用embedding tokenizer，ContextBundle默认参考计数器是 `cl100k_base`，生成器还会用自身聊天模板验证总输入。三个计数服务三个边界：**能编码、便于公平打包、能完整推理**。512块与2048上下文不能简单按“正好放四块”计算，标题、引用标签和分隔符也占token。

### 6.4 原始短文本为何几乎不需要切块

`[文档]` 旧架构核验记录22004/22029篇只有一块，约99.89%；原文token中位149、p99约372，大多低于400/450/512阈值。`[实测]` 原始切块资产的可复查路径及重建口径见第12章。这个观察只能支持“旧语料难以区分长文切块策略”，不能支持“所有任务的切块都无意义”。

长论文把问题变成多个证据分散在不同位置。即使文档命中，关键对照实验或限制条件也可能位于另一块。因此V2/V3新增证据单位坐标、邻块、父子和结构对照，关注真正送入模型的材料，而不只统计有无命中doc_id。

### 6.5 边界情况与可学习点

代码fence被当成结构边界保留，但超长代码仍可能被切开；正则句界可能误判缩写、公式和列表。结构标题只是线索，不是完整的论文逻辑树。保留所有原文坐标、记录fallback与超预算跳过，能暴露这些问题；它不保证块划分最优。`[推断]` 结构方式在本项目的优势主要是可解释、确定性与避免破坏原文，而不是已证明具有最强语义边界质量。

## 7. Embedding 与索引模块

<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 850" role="img" aria-labelledby="ragscope-02-offline-title ragscope-02-offline-desc">
<title id="ragscope-02-offline-title">离线：从论文原文到可复核索引</title>
<desc id="ragscope-02-offline-desc">同一参考块的文本、坐标、ID、向量行号与配置哈希共同构成索引身份。</desc>
<defs><marker id="ragscope-02-offline-arrow" markerWidth="9" markerHeight="9" refX="8" refY="4.5" orient="auto"><path d="M0,0 L9,4.5 L0,9 Z" fill="#77776f"/></marker></defs>
<rect width="100%" height="100%" rx="16" fill="#fdfcf9"/>
<text x="30" y="38" fill="#252522" font-size="24" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="700" text-anchor="start">离线：从论文原文到可复核索引</text>
<rect x="30" y="85" width="330" height="130" rx="10" fill="#eff2fc" stroke="#534ab7" stroke-width="1.6"/>
<text x="45" y="114" fill="#534ab7" font-size="19" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="700" text-anchor="start">原始语料与题库</text>
<text x="45" y="140" fill="#252522" font-size="15" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="400" text-anchor="start">QASPER full_text / 章节段落</text>
<text x="45" y="163" fill="#252522" font-size="15" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="400" text-anchor="start">历史 SQuAD / arXiv 另行保留</text>
<text x="45" y="186" fill="#252522" font-size="15" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="400" text-anchor="start">源数据、划分、版本清单</text>
<rect x="430" y="85" width="330" height="130" rx="10" fill="#eff2fc" stroke="#534ab7" stroke-width="1.6"/>
<text x="445" y="114" fill="#534ab7" font-size="19" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="700" text-anchor="start">规范文本与证据映射</text>
<text x="445" y="140" fill="#252522" font-size="15" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="400" text-anchor="start">doc_id / section_id</text>
<text x="445" y="163" fill="#252522" font-size="15" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="400" text-anchor="start">canonical_start / canonical_end</text>
<text x="445" y="186" fill="#252522" font-size="15" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="400" text-anchor="start">证据片段可回到原文坐标</text>
<rect x="830" y="85" width="330" height="130" rx="10" fill="#eff2fc" stroke="#534ab7" stroke-width="1.6"/>
<text x="845" y="114" fill="#534ab7" font-size="19" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="700" text-anchor="start">结构与参考切块</text>
<text x="845" y="140" fill="#252522" font-size="15" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="400" text-anchor="start">R512 参考块 / chunk_id</text>
<text x="845" y="163" fill="#252522" font-size="15" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="400" text-anchor="start">邻块与 parent/section 关系</text>
<text x="845" y="186" fill="#252522" font-size="15" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="400" text-anchor="start">不冒称相似度驱动语义切块</text>
<polyline points="360,150 430,150" fill="none" stroke="#77776f" stroke-width="1.8" marker-end="url(#ragscope-02-offline-arrow)"/>
<polyline points="760,150 830,150" fill="none" stroke="#77776f" stroke-width="1.8" marker-end="url(#ragscope-02-offline-arrow)"/>
<rect x="30" y="315" width="330" height="135" rx="10" fill="#eff2fc" stroke="#534ab7" stroke-width="1.6"/>
<text x="45" y="344" fill="#534ab7" font-size="19" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="700" text-anchor="start">BM25 稀疏索引</text>
<text x="45" y="370" fill="#252522" font-size="15" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="400" text-anchor="start">正文 / 元数据 chunk 索引</text>
<text x="45" y="393" fill="#252522" font-size="15" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="400" text-anchor="start">文档级 BM25</text>
<text x="45" y="416" fill="#252522" font-size="15" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="400" text-anchor="start">压缩矩阵、词表与清单</text>
<rect x="430" y="315" width="330" height="135" rx="10" fill="#edf7f2" stroke="#257455" stroke-width="1.6"/>
<text x="445" y="344" fill="#257455" font-size="19" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="700" text-anchor="start">真实 Embedding 编码</text>
<text x="445" y="370" fill="#252522" font-size="15" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="400" text-anchor="start">Qwen Embedding 0.6B / 1024 维</text>
<text x="445" y="393" fill="#252522" font-size="15" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="400" text-anchor="start">批量编码 / 受控卸载 / 断点</text>
<text x="445" y="416" fill="#252522" font-size="15" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="400" text-anchor="start">复用 13045；新增 6628 向量</text>
<rect x="830" y="315" width="330" height="135" rx="10" fill="#eff2fc" stroke="#534ab7" stroke-width="1.6"/>
<text x="845" y="344" fill="#534ab7" font-size="19" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="700" text-anchor="start">完整 Dense 矩阵</text>
<text x="845" y="370" fill="#252522" font-size="15" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="400" text-anchor="start">vectors.npy：19673 × 1024</text>
<text x="845" y="393" fill="#252522" font-size="15" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="400" text-anchor="start">float32 / L2 归一化 / mmap</text>
<text x="845" y="416" fill="#252522" font-size="15" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="400" text-anchor="start">文档级 Dense 另有资产</text>
<polyline points="990,215 990,260 190,260 190,315" fill="none" stroke="#77776f" stroke-width="1.8" marker-end="url(#ragscope-02-offline-arrow)"/>
<polyline points="990,260 595,260 595,315" fill="none" stroke="#77776f" stroke-width="1.8" marker-end="url(#ragscope-02-offline-arrow)"/>
<polyline points="760,380 830,380" fill="none" stroke="#77776f" stroke-width="1.8" marker-end="url(#ragscope-02-offline-arrow)"/>
<rect x="230" y="550" width="740" height="145" rx="10" fill="#edf7f2" stroke="#257455" stroke-width="1.6"/>
<text x="245" y="579" fill="#257455" font-size="19" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="700" text-anchor="start">索引对齐、覆盖与身份校验</text>
<text x="245" y="605" fill="#252522" font-size="15" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="400" text-anchor="start">文档集合 / chunk 行 ID SHA / 文本与模型身份</text>
<text x="245" y="628" fill="#252522" font-size="15" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="400" text-anchor="start">向量维度、有限值、单位范数 / sparse 与 dense 行顺序</text>
<text x="245" y="651" fill="#252522" font-size="15" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="400" text-anchor="start">FULL_DENSE_INDEX.json → FullCorpusRetriever 启动时验证</text>
<polyline points="195,450 195,620 230,620" fill="none" stroke="#77776f" stroke-width="1.8" marker-end="url(#ragscope-02-offline-arrow)"/>
<polyline points="995,450 995,620 970,620" fill="none" stroke="#77776f" stroke-width="1.8" marker-end="url(#ragscope-02-offline-arrow)"/>
<text x="30" y="736" fill="#252522" font-size="17" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="400" text-anchor="start">所有大资产位于 D:/RAGScope_Runtime；本说明生成不重新编码、不重建索引、不启动 GPU。</text>
<rect x="30" y="770" width="250" height="40" rx="10" fill="#edf7f2" stroke="#257455" stroke-width="1.6"/>
<text x="45" y="799" fill="#257455" font-size="15" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="700" text-anchor="start">当前默认链路</text>
<rect x="300" y="770" width="250" height="40" rx="10" fill="#eff2fc" stroke="#534ab7" stroke-width="1.6"/>
<text x="315" y="799" fill="#534ab7" font-size="15" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="700" text-anchor="start">共享数据 / 索引</text>
<rect x="600" y="770" width="250" height="40" rx="10" fill="#fff7e8" stroke="#a86a19" stroke-width="1.6" stroke-dasharray="7 5"/>
<text x="615" y="799" fill="#a86a19" font-size="15" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="700" text-anchor="start">独立实验 / 非默认</text>
<rect x="900" y="770" width="250" height="40" rx="10" fill="#f0f0ee" stroke="#74746b" stroke-width="1.6" stroke-dasharray="7 5"/>
<text x="915" y="799" fill="#74746b" font-size="15" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="700" text-anchor="start">历史实现 / 非默认</text>
</svg>

### 7.1 旧BGE与当前Qwen是不同向量空间

V1使用 `BAAI/bge-small-en-v1.5` 的SentenceTransformer方式与Chroma cosine索引；该家族small向量维度384。旧权重具体revision没有完整保存，因此V2的E0重建固定官方版本并独立命名，不能冒称逐权重复现原始Dense部署。当前 `EmbeddingService` 会要求显式device、固定revision、真实tokenizer和输入长度，使用ArrayCache保存身份校验的npy。`[代码]` `ragscope/indexing/embedding.py`。

V2/V3主要使用 `Qwen/Qwen3-Embedding-0.6B`，当前全库空间是官方GGUF **Q8_0 /1024维/单位L2/float32输出/Ollama0.34.4/GPU**。词表和tokenizer revision固定为 `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`；权重SHA为 `06507c7b42688469c4e7298b0a1e16deff06caf291cf0a5b278c308249c3e439`。`[实测]` V3FINAL `FULL_DENSE_INDEX.json.embedding`。

模型元数据原生窗口32768，而本地embedding请求主动设4096，且 `truncate=False`。二者不是矛盾：**模型理论可接收长度与项目为资源/一致性选择的运行长度不同**。官方定义可查[Qwen embedding模型卡](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)；这里的实际维度、窗口和权重以本机清单为准。

### 7.2 文档编码与查询编码

文档端编码既有R512原文块；query端在问题前加：

```text
Instruct: Given a question about a scientific paper, retrieve relevant passages that answer the query
Query:<用户问题>
```

编码输出先检查形状 `(batch,1024)`、有限值与非零范数，再做L2归一化。检索只允许查询identity与索引embedding完全一致；当前实现按身份一致性协议拒绝CPU旧服务或另一种量化模型的向量，即使维度也为1024；这不是断言CPU/GPU在数学上必然属于不同空间。`[代码]` `ragscope/indexing/local_embedding.py:12`、`ReusableEncoder`、`FullCorpusRetriever.search`。

语义评分的答案/参考编码不加这个检索问题前缀；因此“同一个embedding模型”并不等于“相同输入协议”。当前GGUF元数据包含pooling_type3，但代码只把pooling身份记录为后端模型定义；不能声称Python手写复现了HF模型全部pooling细节。

### 7.3 当前具体有哪些索引

| 层次 | Sparse | Dense | 当前用途 |
|---|---|---|---|
| 文档层1585篇 | 标题+摘要+章节名的BM25 | 对应文档表示矩阵 | 全库论文候选与文档先验 |
| R512正文19673块 | CompressedBM25 body | V3FINAL `vectors.npy` | 当前主块检索 |
| R512 metadata | 标题/章节前缀BM25 | 候选实现与构建状态另存 | 当前全库融合使用metadata BM25信号，不默认替代body Dense |
| 其他切块/早期scoped | 单独manifest与缓存 | 早期逐论文NPZ、候选矩阵 | 历史实验、共享/复用来源 |
| V1 | pickle BM25 | Chroma持久集合 | 旧入口/E0核验，不是最新默认数据库 |

`load_hierarchy`校验既有文档层、body sparse与早期scoped制品；`FullCorpusRetriever`再接上完整mmap矩阵，并验证向量/块行序、metadata sparse行序和模型身份。代码仍加载历史层的部分制品作为共享数据，不代表运行范围仍被早期141篇限制。`[代码]` `ragscope/v3/hierarchical_runtime.py`、`full_retrieval.py:11`。

### 7.4 压缩BM25与精确Dense搜索

`CompressedBM25.build`先用CountVectorizer生成词频，计算长度、IDF与BM25权重，再存成CSC稀疏矩阵。查询通过词表定位列，只把命中词的权重累加到相关行。正文表面看是BM25，工程上已不必每次重建一个巨大Python词频对象。`[代码]` `compressed_index.py:12`。

Dense矩阵形状 `(19673,1024)`，float32主体约80.58MB（含npy头80580736字节），并不是几十GB。`mmap_mode='r'`按文件映射加载；计算仍要访问参与运算的行，并不是零内存开销。全库矩阵点积是精确穷举，当前没有FAISS/HNSW近似检索的默认路径，因此不要在简历写“当前使用ANN向量数据库加速百万级检索”。

### 7.5 缓存身份、批量编码与断点

`ArrayCache`目录按完整model_identity哈希分隔，文件key又包含文本+模型身份；每个npy旁边有SHA/路径/identity清单。读取校验实际字节而不只相信文件名。模型名相同但revision、tokenizer、量化、后端等身份不同，不应复用。`[代码]` `ragscope/indexing/array_index.py:14`。

补全脚本 `scripts/v3/complete_dense.py` 采用16条一批、预分配mmap与完成位图；先复用通过校验的缓存，再编码缺失项。写入顺序是向量flush→缓存→完成位图原子替换→批次回执。恢复时已完成行要通过有限值与单位范数检查，行ID哈希必须一致。

V3收尾时复用13045、新计算6628，合计19673；1585篇全部覆盖，原509目标缺失0。`[实测]` FULL_DENSE_INDEX、FULL_CORPUS_VALIDATION、TARGET_509_COVERAGE及批次记录。这个数字是索引完成度，不是检索正确率。

### 7.6 内存工程的实际处理

编码器驻留后，再为每批要求几GiB额外内存会重复计入已驻留开销。补全入口缩小启动请求量，保留系统reserve，并在可用内存低或每512个新向量时受控卸载**自有编码器**，让后端回收主机内存；可用内存低于512MiB时停止，保存断点。没有靠修改WSL内存上限、重启系统或结束未知用户任务完成。

当前 `ReusableEncoder.embed(cached=False)` 只跳过读缓存，仍会写新向量；不能把这个参数理解成“绝不写缓存”。最终在线实验改用专门 `UncachedOnlineEmbedding`，避免覆盖旧query缓存；受影响的24条缓存已按冻结数据逐位恢复，原SHA核验保留。`[代码]` `finish_runtime.py`、`online_runtime.py`；`[实测]` `QUERY_CACHE_RESTORATION.json`。这是需要读实现才能发现的缓存语义差异。

## 8. Retrieval 检索模块

<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 1010" role="img" aria-labelledby="ragscope-04-retrieval-title ragscope-04-retrieval-desc">
<title id="ragscope-04-retrieval-title">多级论文检索、任务证据与生成预算</title>
<desc id="ragscope-04-retrieval-desc">全库与指定论文使用不同范围策略；文档得分在全库中作为块级融合先验，不是强制只搜前十篇。</desc>
<defs><marker id="ragscope-04-retrieval-arrow" markerWidth="9" markerHeight="9" refX="8" refY="4.5" orient="auto"><path d="M0,0 L9,4.5 L0,9 Z" fill="#77776f"/></marker></defs>
<rect width="100%" height="100%" rx="16" fill="#fdfcf9"/>
<text x="30" y="38" fill="#252522" font-size="24" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="700" text-anchor="start">多级论文检索、任务证据与生成预算</text>
<rect x="415" y="75" width="370" height="105" rx="10" fill="#edf7f2" stroke="#257455" stroke-width="1.6"/>
<text x="430" y="104" fill="#257455" font-size="19" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="700" text-anchor="start">问题 + 显式论文范围</text>
<text x="430" y="130" fill="#252522" font-size="15" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="400" text-anchor="start">doc_ids 为空：全库；否则：所选论文</text>
<text x="430" y="153" fill="#252522" font-size="15" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="400" text-anchor="start">最多 10 篇；任务可自动或显式选择</text>
<rect x="45" y="245" width="495" height="145" rx="10" fill="#edf7f2" stroke="#257455" stroke-width="1.6"/>
<text x="60" y="274" fill="#257455" font-size="19" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="700" text-anchor="start">全库检索</text>
<text x="60" y="300" fill="#252522" font-size="15" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="400" text-anchor="start">文档 BM25 / Dense → 文档先验</text>
<text x="60" y="323" fill="#252522" font-size="15" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="400" text-anchor="start">全部 chunk 候选：正文 + 元数据 + Dense</text>
<text x="60" y="346" fill="#252522" font-size="15" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="400" text-anchor="start">按权重 RRF 融合，并加入文档先验分数</text>
<rect x="660" y="245" width="495" height="145" rx="10" fill="#edf7f2" stroke="#257455" stroke-width="1.6"/>
<text x="675" y="274" fill="#257455" font-size="19" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="700" text-anchor="start">单篇 / 多篇指定范围</text>
<text x="675" y="300" fill="#252522" font-size="15" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="400" text-anchor="start">只取指定 doc_id 对应的 chunk 行</text>
<text x="675" y="323" fill="#252522" font-size="15" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="400" text-anchor="start">跳过全库文档定位；BM25 / Dense / Hybrid</text>
<text x="675" y="346" fill="#252522" font-size="15" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="400" text-anchor="start">多篇可按论文交替组织候选</text>
<polyline points="490,180 490,210 295,210 295,245" fill="none" stroke="#77776f" stroke-width="1.8" marker-end="url(#ragscope-04-retrieval-arrow)"/>
<polyline points="710,180 710,210 905,210 905,245" fill="none" stroke="#77776f" stroke-width="1.8" marker-end="url(#ragscope-04-retrieval-arrow)"/>
<rect x="45" y="475" width="495" height="145" rx="10" fill="#edf7f2" stroke="#257455" stroke-width="1.6"/>
<text x="60" y="504" fill="#257455" font-size="19" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="700" text-anchor="start">普通事实 / 解释</text>
<text x="60" y="530" fill="#252522" font-size="15" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="400" text-anchor="start">所选论文：保留 B0 Hybrid + 邻块</text>
<text x="60" y="553" fill="#252522" font-size="15" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="400" text-anchor="start">全库：开发集选定的全库检索参数</text>
<text x="60" y="576" fill="#252522" font-size="15" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="400" text-anchor="start">最多 2048 参考 token；输出 512</text>
<rect x="660" y="475" width="495" height="145" rx="10" fill="#edf7f2" stroke="#257455" stroke-width="1.6"/>
<text x="675" y="504" fill="#257455" font-size="19" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="700" text-anchor="start">总结 / 比较 / 多篇</text>
<text x="675" y="530" fill="#252522" font-size="15" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="400" text-anchor="start">总结：章节线索与原文顺序补充证据</text>
<text x="675" y="553" fill="#252522" font-size="15" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="400" text-anchor="start">比较 / 多篇：分组交替，避免单篇占满</text>
<text x="675" y="576" fill="#252522" font-size="15" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="400" text-anchor="start">最多 6144 参考 token；输出 768</text>
<polyline points="295,390 295,432 295,475" fill="none" stroke="#77776f" stroke-width="1.8" marker-end="url(#ragscope-04-retrieval-arrow)"/>
<polyline points="905,390 905,432 905,475" fill="none" stroke="#77776f" stroke-width="1.8" marker-end="url(#ragscope-04-retrieval-arrow)"/>
<polyline points="540,315 600,315 600,433 905,433 905,475" fill="none" stroke="#77776f" stroke-width="1.8" marker-end="url(#ragscope-04-retrieval-arrow)"/>
<polyline points="660,315 625,315 625,450 295,450 295,475" fill="none" stroke="#77776f" stroke-width="1.8" marker-end="url(#ragscope-04-retrieval-arrow)"/>
<rect x="245" y="715" width="710" height="125" rx="10" fill="#edf7f2" stroke="#257455" stroke-width="1.6"/>
<text x="260" y="744" fill="#257455" font-size="19" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="700" text-anchor="start">去重、原文坐标与实际模型预算检查</text>
<text x="260" y="770" fill="#252522" font-size="15" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="400" text-anchor="start">pack_context → 带 chunk_id / doc_id / canonical 坐标的 ContextBundle</text>
<text x="260" y="793" fill="#252522" font-size="15" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="400" text-anchor="start">统一后端窗口 8192；完整 prompt + 输出预留必须适配</text>
<text x="260" y="816" fill="#252522" font-size="15" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="400" text-anchor="start">可见引用 ID 验证 ≠ 引用支持性证明；证据覆盖 ≠ 答案正确率</text>
<polyline points="295,620 295,665 430,665 430,715" fill="none" stroke="#77776f" stroke-width="1.8" marker-end="url(#ragscope-04-retrieval-arrow)"/>
<polyline points="905,620 905,665 770,665 770,715" fill="none" stroke="#77776f" stroke-width="1.8" marker-end="url(#ragscope-04-retrieval-arrow)"/>
<rect x="45" y="890" width="1110" height="72" rx="10" fill="#f0f0ee" stroke="#74746b" stroke-width="1.6" stroke-dasharray="7 5"/>
<text x="60" y="919" fill="#74746b" font-size="18" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="700" text-anchor="start">历史 Cross-Encoder / 学习型检索路由：当前未进入默认链路</text>
<text x="60" y="945" fill="#252522" font-size="15" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="400" text-anchor="start">历史候选筛选与重排可改变证据覆盖，不能把实验存在写成在线已启用。</text>
</svg>

### 8.1 最新融合公式与查询范围

`FullCorpusRetriever.search(question, scope, query_vector, embedding_identity, method, alpha, metadata_weight, document_weight, k, candidate_depth, balanced)` 是当前检索入口。`method`为bm25/dense/hybrid。默认方法hybrid；普通指定问答候选深度50，默认search返回至多50块。它先对允许的chunk行分别排序，再用加权RRF合并：

```text
chunk_score(c) = (1-alpha)/(60+rank_body_bm25)
               + alpha/(60+rank_body_dense)
               + metadata_weight/(60+rank_metadata_bm25)
               + document_weight * document_score(doc(c))
```

不在某路候选中的项没有该项贡献。纯BM25或纯Dense在这个类中也将相应排名映射为倒数rank得分后返回，因此返回score不应直接解释为原始BM25分或原始余弦值。`document_score`由文档BM25与Dense排名融合而来；纯Dense不加metadata sparse。当前全库普通问答选metadata_weight=.25、document_weight=1.0、alpha默认.5；指定复杂任务选alpha=.75。

**关键区别**：当前全库并不是“先取10篇，再只搜索这10篇”。它在全库块候选上融合，再加文档先验；Top文档仅是额外信号。早期 `HierarchicalRetriever` 的多级流程是另一份实现，不能统一描述成同一个硬级联架构。

### 8.2 全库、单篇、多篇

- **全库**：scope为空，参与行是19673块；计算文档候选，给候选块加入文档先验，返回论文候选供展示。
- **单篇**：scope中有明确doc_id，参与Dense乘法仅为该论文的行；跳过全库文档定位，不拿其他论文补答案。
- **多篇**：只在用户选定论文集合内检索；复杂工作流开启balanced，将各论文候选轮询交织，避免单篇占满前列。轮询并不保证每篇最终都有足够预算。

`FullWorkbench.answer`会拒绝不存在的论文；未选论文而问题含英文“this paper/the paper/this study”时返回CLARIFY_DOCUMENT。这个检查是有限的英文规则，不是完整指代消解；中文“这篇论文”或其他表达可能不被同一guard识别。`[代码]` `full_workbench.py:50`。

### 8.3 从“找到了”到“真的送给模型”

检索返回候选不等于候选全部进入上下文。`pack_context`按顺序处理候选：scope检查→邻块/父块扩展→同ID/同文档同文本去重→减去已覆盖canonical区间→加入标签标题后重计token→参考预算及模型预算检查→保留片段或记录跳过原因。

超预算候选通常是**整段跳过并继续看后续候选**，不是静默把末尾截掉。重叠移除可能生成带区间后缀的新引用ID并保留source_chunk_id，所有坐标同步投影。ContextBundle保存去重、truncations、unit_spans和context_hash，因此能查“证据检索到了，却为何没有实际输入”。`[代码]` `generation/context.py:97`。

普通指定问答保留旧neighbor打包，先处理一个核心及邻居再看下个核心，可能让邻居挤掉其他证据；新 `pack_core_first`先保核心，再给邻居使用不超过总预算25%的剩余空间，并验证核心未变。它属于部分新工作流，不可写成所有当前问答均使用core-first。

### 8.4 为什么总结不能只搜“summarize this paper”

泛泛的总结问题本身没有具体术语，直接按相关性可能只重复命中摘要和引言。当前指定论文summary先从原文块顺序中找起始块、方法相关标题、实验结果相关标题、结论/局限末端，再补剩余块；多论文交织，最多6144参考token，并实际检查8192总窗口可容纳768输出。

这是结构关键词与顺序启发式，不是训练了“知道全篇重点”的摘要规划器。UI显示source_character_coverage，覆盖低时不能声称读完全文；即使高覆盖也不证明重点理解正确。`[代码]` `deployment.py:25`。

### 8.5 重排负结果应该怎样解释

CrossEncoder通常在候选内估计问题-文本相关性，而本项目长论文题可能需要多段互补证据。重排改变顺序后，固定预算下进入模型的块也改变；邻块扩展可能进一步放大排序差异。**覆盖下降是可观察结果；“模型不擅长跨段证据”只是可能原因之一。** 还需控制候选池、token长度、章节、多段需求才能作因果判断。

历史V1 Hybrid+Rerank与Hybrid的Recall@10可同为.95，但MRR下降；V2/V3实际证据覆盖也出现退步，详见第12章。当前FullCorpusRetriever没有CrossEncoder调用，默认关闭重排。不能把“代码保留reranker”写成“在线有重排优化”。

### 8.6 用一道真实题模拟路径

最终H_QA中有题 `Which news corpus is used?`，范围论文为 `qasper:2002.11402`。查询编码加科学论文检索指令；scope限定该篇；hybrid=.5召回候选；G0/neighbor打包最多2048参考token；生成返回JSON，答案是私有英文新闻语料并提及Wikipedia标题。原始parsed与实际payload均在 `DEPLOYMENT_FINAL_ANSWERS_RESIDENT8K.jsonl`。

这个例子证明整条链路实际运行过，**不作为完美引用的示范**：引用ID合法不自动证明每个陈述都由对应段落支持。核验引用应把answer、citation pieces与参考原文同时阅读。正文没有为了展示而编造一段“理想模型输出”。

## 9. Generation 生成模块

### 9.1 模型与后端的分工

真实生成比较使用Qwen3.5-2B/4B/9B及V2的LFM2.5-1.2B；当前默认tag为 `ragscope-qwen3.5:2b-q4_K_M`，量化Q4_K_M，不加载LoRA。Qwen embedding .6B独立用于编码。模型权重、tokenizer、模板和后端版本清单见第13章及附录D。

当前服务由 `scripts/v3/serve_full_workbench.py` 创建私有Ollama0.34.4进程，后端11435、API11437、UI8502，均loopback。原系统11434服务不是当前工作台的默认调用目标。普通指定论文的explanation标签仍走B0/G0；虽然源码保存了新解释提示文本，它并不因此在这条默认分支启用。生成请求使用 `stream=False,think=False,temperature=0,seed=42`；这些设置降低随机性，不等于所有硬件/后端结果逐位确定。

### 9.2 Prompt与Chat Template怎样连接

`answer_messages`决定系统指令和用户证据；普通指定问答采用G0，要求仅用所给上下文、把资料视为数据、证据不足时拒答、输出JSON。新summary/comparison指令要求区分方法、条件、结论和局限；未提供的局限不应编造。`finish_runtime.py`。

消息列表送入 `DeploymentClient`，它继承ProtocolClient/CapturedClient/LocalLLMClient，保存actual_sent_payload与raw backend response。`original_client`从已固定的V2模型包构建token计数器；chat template由本地模型/包模板渲染，模型身份记录chat_template_hash。**Python中的system/user文本不是完整模型输入；角色包装也要计数。** `scripts/v3/finish_experiments.py`、`ragscope/generation/llm_client.py:113`。

### 9.3 JSON schema与引用检查

```json
{"answer":"根据证据形成的回答", "citations":["本次上下文里可见的ID"], "abstain":false}
```

这是输出结构示意，不是某次实测答案。schema只允许answer/citations/abstain三个字段，引用元素枚举本次实际piece IDs；无piece时maxItems=0。多篇输入额外限制最多4条引用、answer最多1600字符，并要求按论文清晰作答，以降低768输出token内JSON截断风险。

模型输出经 `RAGPipeline.parse_output:165` 检查类型、JSON与abstain布尔值，FullWorkbench再检查引用在实际上下文内，以及token_count_match必须为True。当前并没有一个强模型逐条判断这些引用蕴含了答案。合法率高具有schema构造因素，不能宣传成“引用100%正确”。

### 9.4 长度预算与任务差异

| 默认任务 | 证据策略 | 参考证据token | 输出token上限 | 后端窗口 |
|---|---|---:|---:|---:|
| 单篇事实/解释 | B0普通hybrid+neighbor、G0 | 2048 | 512 | 8192 |
| 全库普通问题 | 文档先验/metadata融合，core-first打包 | 2048 | 512 | 8192 |
| 总结/比较或多篇 | 结构/范围平衡，原文片段 | 6144 | 768 | 8192 |

`LocalLLMClient.chat`先检查完整prompt+输出预留≤窗口，并比较后端prompt_eval_count与本地计数。6144是参考token上限，不是实际必定塞满；超长片段被跳过。长证据增加prefill与KV内存、输出更长增加decode耗时，仍可能发生注意力分散或比较方向混乱。`[代码]` `deployment.py:25`、`llm_client.py:113`；`[实测]` 最终payload与token_count_match记录。

### 9.5 错误、拒答与降级的真实行为

“不可答”应是有意义的 `abstain=true`，不是请求超时或解析失败。当前应用把异常写进独立APPLICATION账本并返回EXECUTION_ERROR；无效输入、未知论文和有限英文指代分别返回INPUT_ERROR/INVALID_DOCUMENT/CLARIFY_DOCUMENT。客户端无隐藏重试，错误不会悄悄改用Mock。

服务离线时UI可读保存的catalog与实验历史，但生成按钮禁用；这是界面离线降级，不是模型回答降级。索引或源哈希不符时启动拒绝，不自动重建旧文件。早期Workbench曾有BM25零分NO_EVIDENCE预判，**新FullWorkbench没有相同的零分前置分支**，不要跨版本合并行为。

最终220条答案0生成失败，不代表应用永不报错；真实应用验收保留过1次多篇JSON截断，随后限制输出/引用并复测场景通过。30不可答仅3正确拒答，说明prompt要求没有自动转化成可靠行为。Mock仅为功能替身，`real_model=False`，正式 `RAGPipeline`会拒绝用它产出模型成绩。


## 10. Routing 与任务工作流


### 路由究竟在选什么

[代码] V2 路由器的输出是三个**检索路径**之一：`bm25`、`hybrid`、`hybrid_rerank`，而不是生成答案，也不是判定“事实/比较/总结”。`ragscope/routing/core.py:12` 的 `ROUTES` 固定了这三个动作。生成器始终冻结为同一 Qwen3.5-2B Q4_K_M、本地 Ollama 0.34.4、4096 后端窗口、greedy、512 输出 token、seed42；因此实验试图隔离“选哪条检索路径”的作用。完整模型身份在 `ROUTE/ROUTING_MANIFEST.json` 的 `comparison.model_identity` 中。

通俗地说，Embedding 模型负责“把问题和段落变成可比较的数值”；检索器负责“把材料找出来”；路由分类器只负责“这道题应该派哪个检索器”；生成模型最后才根据材料写答案。这四种角色不能统称为同一个“小模型”。

### 标签来自真实答案的代理效用，并非人工理想路线

[代码] `ragscope/routing/core.py:118` 的 `build_proxy_labels` 要求同一道题三个路径都有真实生成记录、同一配置哈希、明确失败状态。可答题用文字 token F1，不可答题用是否正确拒答；失败保留零分。在最高代理得分相差不超过0.01的路径中，先选记录耗时较低的，再比较证据覆盖，最后按固定路径顺序打破平局。于是一个“BM25标签”表示这次具体模型、prompt、上下文和成本规则下的代理选择，绝不表示问题本质上只能用关键词检索。

[实测] `ROUTE/LABEL_PROVENANCE.json` 记录 train200/dev60；训练标签分布为 BM25 128、Hybrid42、重排30，开发标签为31/23/6。全部三条路线的真实生成来源保存在 `V2` 对应 generation 目录的 `ROUTE_TRAIN_qwen_2b.jsonl`、`ROUTE_DEV_qwen_2b.jsonl`。训练器不读 test 标签来拟合或选模型。`ragscope/routing/core.py:180` 同时拒绝 train/dev 问题重叠及论文重叠。

### 每种实际实现

|方法|实际输入与算法|已有资产与状态|
|---|---|---|
|规则|含 compare/contrast/difference/why 等词选重排；引号、代码标识或数字字母标识选BM25；其他选Hybrid|[代码] `ragscope/routing/core.py:68` `RuleRouter`；[实测] 已重放并做真实在线对照|
|线性|问题加合法元数据 → unigram/bigram TF-IDF，最多2048特征 → LogisticRegression，C=1.0，max_iter20|[代码] `ragscope/routing/learned.py:32` `TfidfRouter`；[实测] 实际1755特征，求解17次迭代；`ROUTE/linear/{manifest.json,weights.npz,training_report.json}`|
|MLP|同TF-IDF → train-only TruncatedSVD64维 → 隐层32的ReLU MLP → 三类softmax；Adam学习率0.001，batch最多16；dev log loss早停，patience3|max20轮，实际历史14轮；[实测] `ROUTE/mlp/`；模型取最佳dev轮，不是最后一轮|
|LSTM|自有单词/中文字分词，词表只从train建立，最多8000；截到128词；64维词向量 → 单层LSTM hidden64 → 三类线性层；Adam0.001、batch16、CPU4线程、dev早停|[代码] `ragscope/routing/lstm.py:24`、`:67`；[实测] 实际69251参数、训练历史9轮、train/dev无截断；`ROUTE/lstm/`|
|Laya|固定本地英语 checkpoint，ModernBERT 编码器和SDK的choice接口；显式传入三个路线说明，不训练项目专属新Laya；max_len512，CPU eager、float32、4线程、置信阈值0.5，低置信回退Hybrid|[代码] `ragscope/routing/laya_adapter.py:39`；[实测] 360题×3候选顺序=1080次决策，另3次预热，错误0、回退646次；`LAYA/RAW_ACCEPTANCE.json` 和 `LAYA/records.jsonl`|
|Jev|没有本地可验证实现或本轮请求|[代码] `configs/v2/routing.json` 为 `NOT_APPLICABLE`，理由是可选付费外部API被排除；不能把它算作完成的实验|

[代码] 路由特征仅含 question 与白名单元数据：文档长度分组、document_type、language。论文ID可用于合法范围控制，但不会作为训练快捷特征，见 `RoutingRequest.features_text`，`ragscope/routing/core.py:28`。TF-IDF、SVD词表与权重保存为 JSON+NPZ，加载时核验文件哈希且 `allow_pickle=False`；这与“随意加载一个pickle分类器”不同。

### V2 路由结果及结论

[实测] 下表均为同一 Qwen2B Q4_K_M、同一冻结问题与各路径真实生成结果的**重放**。每策略 dev60（可答56）和test100（可答92）；三类学习模型在dev/test均预测BM25。8策略×160=1280次选择重放，并非1280次新生成调用。原始来源 `ROUTE/REPLAY_<policy>.json` 的 `groups.<split>.summary`。

|策略|dev文字F1|test文字F1|说明|
|---|---:|---:|---|
|固定BM25|0.380598|0.459276|学习模型在测试集上的实际输出等同此路径|
|固定Hybrid|0.429777|0.481006|开发选出的固定基线|
|固定Hybrid+重排|0.370988|0.466471|dev证据覆盖0.663690，低于Hybrid0.767857|
|规则|0.426660|0.477269|质量接近，但不构成明确提升|
|线性/MLP/LSTM|0.380598|0.459276|当前样本与标签下退化为BM25选择|
|Laya（预固定顺序0）|0.429777|0.477564|dev回退95%、test回退96%；分类本身仍有成本|

[实测] `ROUTE/train_diagnostics/ACCEPTANCE.json` 是保存权重的额外train200重放，不再fit、无新LLM调用。linear train/dev分类accuracy为0.675/0.516667；MLP/LSTM均0.64/0.516667。训练多数类比例64%、开发多数类51.667%，所以不能只看到train/dev差距就判定经典“模型太复杂导致过拟合”。至少已直接观察到多数类主导、标签代理有偏、学习模型未学到可部署的分流规则；根因因果分解仍[未知]。

[实测] 最终仅对固定Hybrid和规则进行同30个dev题的实际在线运行，各30条正式请求、另各3次预热，均无正式调用失败。固定/规则在线P50为1.277657/1.213204秒，降低5.0446%；P95为2.026595/2.201954秒。开发集F1容差通过，但未达到预先规定的至少10% P50减少，因此 `ROUTE/online/SUMMARY.json` 的 `deployment` 为 `deploy:false, default:"fixed"`。重放分量相加的耗时不能与这次 actual_online 混在一起，也不能只挑P50忽略P95变差。

[实测] Laya的候选顺序敏感：1080次原始决策中142/360题的最终路线随顺序改变；dev有59/60题、test99/100题的原始首选发生改变（后者是dev/test分组的原始choice指标，不同于全360题的阈值回退后route指标）。`ROUTE/LAYA_PERMUTATION_CALIBRATION.json` 保留三个顺序的概率、10固定bin的ECE与Brier。非默认顺序有个别更高测试F1也不能事后改成正式选型；原顺序在读取质量标签前已经固定。

### V3 的任务类型与旧路线分类器不同

[实测] V3前期固定60题、每题W_local/W_compare/W_summary三次生成，共180次，加9次独立预热。可答52；问题类别28事实、22解释、5比较、5其他，**没有总结标签题**。W_local/W_compare/W_summary的可答文字F1分别0.363394/0.340127/0.360794。事先规定的类别→工作流映射，相对最佳固定W_local的混合代理效用低0.012080；因此没有训练新linear/MLP/LSTM/Laya分类器。来源 `V3D/S4_WORKFLOW_UTILITY_ANALYSIS.json`。用短问答参考答案评价W_summary，不能证明整篇总结能力差；该实验本来就缺少总结类别监督。

[代码] 最新V3F默认并未接入V2这些学习分类器，而是范围控制加可审计的任务规则：普通事实/解释保留原有可靠短证据协议；比较、总结、多篇任务使用更广的证据组织。`V3F/FINAL_CONFIGURATION.json` 冻结 `SAFE_QA_TASKS` 和选型依据；`configs/v3/default.json` 指向当前已验收发布。用户可以手动指定任务。真实语义问题在于“这道题需要哪种证据范围”，而不是任何复杂问题都去重排。当前规则不是经过训练并证明泛化更好的语义分类器。



## 11. 微调与模型训练


### 通俗原理与本项目的实际选择

LoRA可以理解成给原模型的一些线性变换加一组很小的可训练“修正层”。原权重W冻结，只训练低秩矩阵A、B，使输出近似使用 `W + (alpha/r)BA`。这样需要优化和保存的参数少很多，但底座仍须常驻并执行前向/反向，并不等于训练几乎不占显存。

[代码] [实测] 本项目实际是**BF16底座LoRA，非QLoRA**。底座是 `Qwen/Qwen3.5-2B` 固定revision `15852e8c16360a2fea060d615a32b45270f8a8fc`，本地完整safetensors约4.55GB；它与部署的GGUF Q4模型属于不同表示与后端。配置有0.8B后备路径，不等于实际下载或训练了0.8B；最终选中 `selected_path_index:0`、真实122步来自2B。模型只在取得GPU lease、通过资源/资产校验后加载。

### 数据怎样制成训练样本

[代码] `ragscope/training/prepare.py:15` 先核验源 `SPLIT_MANIFEST.json`、queries与documents的哈希。`ragscope/training/data.py:39`：

1. 按query_id排序，从train最多取1000、dev最多取60，test不可传入准备流程。
2. 可答题取同一个合法annotation配对的参考答案和完整参考证据，校验原文坐标及全文覆盖；不会把一条annotation的答案与另一条annotation的材料随意拼接。
3. 不可答题使用真实标注负例和论文的真实首段，目标为标准拒答JSON，不虚构证据段落。
4. 形成system/user/assistant消息；assistant目标为 `{"answer":...,"citations":[...],"abstain":...}`。
5. 用实际Qwen tokenizer/chat template编码；prompt部分label=-100，仅assistant与结束标记参与交叉熵。完整样本超过1024 token就拒绝，**不截掉证据或目标答案硬塞**。

[实测] 本次直接读取 `LORA/prepared_2b_1024_v2/` 中的token数组再聚合，核对输入/labels/attention_mask长度和assistant-only mask，全部通过；同时读取train/dev文档ID确认交集为0。对应源manifest SHA为 `901bb1485cc2490fac9f149e02b7f3b48f48e39e0b581852022390473634f5e8`，当前源文件再次计算一致。

|数据|拟选上限|实际接受|文档数|可答/不可答|完整样本最大token|超预算被拒|
|---|---:|---:|---:|---|---:|---:|
|train|1000|972|394|841/131|999|28|
|dev loss|60|59|31|53/6|825|1|

数据格式JSONL保留query_id、doc_id、split、annotation_id、messages、input_ids、attention_mask、labels、prompt/supervised/total token数、context_hash及evidence_policy。`train_manifest.json`、`dev_manifest.json`、`MASK_AND_SPLIT_AUDIT.json`都存在。源语料是QASPER官方按论文划分后冻结的V2语料，不是用当前生成模型自造972道QA。

### 精确训练配置与执行

|项目|实际设置/观察|证据|
|---|---|---|
|底座|BF16；eager attention；本地加载，`trust_remote_code=False`|`ragscope/training/run.py:117`|
|LoRA|r8、alpha16、dropout0.05、bias none、CAUSAL_LM|`configs/v2/lora.json`；最终 `adapter_config.json`|
|适配目标|从模型真实模块类及祖先发现语言attention/GatedDeltaNet中的Linear；排除vision/MLP/lm_head/embed|`discover_attention_targets`，`ragscope/training/run.py:63`|
|实际规模|114个投影模块、228个A/B张量、3691008个可训练参数|`TRAIN/model_observed.json`|
|输入与batch|1024总token；microbatch1，梯度累积8；最后不足8条按实际条数归一|`TRAIN/steps.config.json`；`train_loop`|
|优化|AdamW学习率1e-4；线性warmup+decay，warmup_ratio0.03；梯度裁剪1.0|`ragscope/training/run.py:215`附近|
|内存|非reentrant gradient checkpointing；训练use_cache=False；实际运行启用每microbatch和dev后清未使用allocator cache|`run.py:137`附近；steps runtime_policy|
|上限与实际|配置max_steps200、一轮；`min(200,ceil(972/8))`产生122实际optimizer steps|`run.py:215`；本次重算steps连续1..122|
|随机性|seed42在底座与adapter初始化之前设置；固定样本顺序|`model_observed.initialization_seed`，初始adapter哈希|
|dev评估|step25/50/75/100/122，平均监督loss；以最低dev loss保留best指针|`TRAIN/TRAINING_AUDIT.json`|

[实测] dev loss为0.158186 →0.142838 →0.139419 →0.137208 →0.135663；最佳为step122。首个训练step loss0.383955、末step0.525464并不要求逐步单调，因为每步样本不同。不能把较低dev teacher-forcing loss直接翻译成“真实RAG答案更正确”。Torch allocator的峰值allocated7,608,193,024字节、reserved8,283,750,400字节，不等同物理显卡占用；Windows共享/分页可能参与，原训练审计已明确这一测量边界。step计时和不包含checkpoint/dev/model load的累计1215秒，也不是总训练墙钟时间。

### 完整断点恢复到底恢复了什么

[代码] `ragscope/training/checkpoint.py:26` 先写拥有明确run归属的临时目录，保存全部文件并fsync，生成每文件大小/SHA的manifest，通过原子rename提交，再更新latest/best。状态含adapter、optimizer、scheduler、Python/NumPy/Torch/CUDA RNG、data_order、data_position、step、epoch；`training_state.pt`以 `weights_only=True` 读取。目录归属或数据/配置哈希不符就拒绝采用，不会接管用户其他断点。

[实测] step5与step10恢复收据都为 `RESTORED_VERIFIED`：adapter、optimizer、scheduler、RNG、样本顺序逐项精确相等，数据位置与optimizer步数一致。当前保留的是step120、121、122及best122指针；step5/10的恢复验证收据仍在，不应误说所有历史step断点都永久保留。保留上限3仅作用于本run中已验证归属的训练断点。

`[文档]` `reports/v2/TRAINING_REPORT.md:7` 还明确披露：旧BF16配对原执行句柄丢失，原OS退出码UNKNOWN；后续独立审计核对完整206预算票及产物，审计自身实际exit0。本手册以原始响应与独立审计证明产物完整，不补写未知的原进程退出码。

### 同精度前后比较

[代码] `ragscope/training/evaluate.py:89` 使用同一已加载BF16底座、tokenizer、prompt及冻结上下文，通过 `disable_adapter()` 对照基础模型与adapter；greedy、max_new_tokens512、总窗口4096，两条件不使用Ollama的citation enum schema约束。`validate_comparison_inputs` 拒绝训练论文进入test，并要求上下文可验证。质量评估单独于训练loop完成。

[实测] `TRAIN/paired_test100/SUMMARY.json` 和独立完成审计证实100+100正式题，另3+3预热；其中92题可答、8题不可答，解析失败按原分母保留。

|指标|同BF16基础模型|同BF16+adapter|
|---|---:|---:|
|可答文字F1|0.464889|0.426683|
|可答EM|0.184783|0.217391|
|可答ROUGE-L|0.446955|0.421246|
|全部正式题解析/请求失败|1/100|2/100|
|整条记录P50秒|6.194457|22.400814|
|可答题拒答|0/92|23/92|
|不可答正确拒答|1/8|3/8|

这里可以得出“此adapter在这100题协议中token F1下降、拒答增多、耗时变长”，不能得出所有LoRA无用；EM反而上升也提醒不同指标可以指向不同侧面。`V3D/LORA_OLD_RECOUNT.json` 将92题F1差−0.038205分解：28题改善贡献+0.123935、新增23次可答拒答贡献−0.097927、继续作答但退步贡献−0.051438、失败贡献−0.012775。23次拒答不自动等于23次幻觉抑制失败：问题在论文中可答，不代表当前检索包足够回答。旧充分性代理对它们判断YES8/NO12/UNKNOWN3，未经过人工审核。

### V3 已经做过哪些机制验证

|待解释现象|已有检查与观察|能支持/不能支持的结论|
|---|---|---|
|是否监督mask坏了|972条数值mask、EOS、可见引用检查通过；20条目标真实解码完整；固定8夹具兼容smoke完成2个optimizer steps、228张量变化、平均监督loss下降0.061489|没有发现整体assistant mask失效；小夹具能学习不等于泛化质量提升|
|是否训练与部署材料不同|train prompt中位350.5 token，旧test2147.5；841干净参考证据、131真实首段负例，部署则用检索包|存在明确分布差异；尚未通过匹配分布消融证明它是退步原因|
|是否EOS错误无限续写|D_confirm前24题，同BF16/eager/greedy/512，base和unmerged各24正式+1预热；两分支全24次自然EOS结束，首im_end后仅多2 token|本样本反驳“大量EOS续写造成变慢”；不能证明所有输入都无停止问题|
|是否输出变长|上述24题输出token中位67→171，生成P50 5.7494→19.8797秒|长度增加与时延相关，但不能单独识别每token开销|
|是否adapter逐token计算更慢|同3输入强制64token，每分支3正式+1预热；首token P50 0.4912→0.5489秒，后63token速度中位12.5634→8.5900 token/s|有未合并adapter每token额外开销的直接微基准证据；只3样本，不是在线SLA，不评价答案质量|
|合并能否修复|3输入末位置logits的事先容差max_abs≤0.08、relativeL2≤0.01；实际max_abs0.50–0.625、relativeL2 0.0332–0.0501；argmax相同但gate失败|未保存合并权重快照，未继续合并24题/微基准；不能说合并已成功或已改善速度|

[实测] 新24题可答22，包含失败零分的F1是base0.301848、unmerged0.393765，方向与旧100题相反；base出现8条不可见引用失败、adapter0条。它体现**协议、引用验证和失败构成会改变结论**，不是adapter已证明事实能力更强。数字来自 `V3D/S5_RUNTIME_INDEPENDENT_ANALYSIS.json`；合并探针来自 `S5_MERGE_LOGIT_GATE.json`；监督分布来自 `S5_SUPERVISION_DISTRIBUTION_DIAGNOSIS.json`；固定64微基准本次对 `S5_FIXED64_{base,unmerged}.jsonl` 排除warmup后独立重算，与报告一致。

[未知] [推断] BF16合并舍入、残差计算路径可能产生logits差，但未隔离原因，不应称PEFT bug。可进一步验证的方向是同分布的干净证据与检索噪声分层、长度匹配、拒答阈值/标注充分性审核、同协议语义与事实指标、不同checkpoint的冻结dev对照。这些是后续研究问题，不是本轮已完成的新训练。

[代码] [实测] 当前没有把旧adapter并入Q4默认模型；V3正式新LoRA未触发，只做了监督链路smoke与机制诊断。原因是旧结果没有稳定质量收益，且预设“材料充分时仍有可训练缺陷/足够部署分布样本/可信主事实评价”等前置条件未同时满足。`configs/v3/default.json`、V3F选型记录与实际workbench均保留基础2B路线。历史 `LORA_DIAGNOSIS.md` 末尾“正式holdout受阻、默认v2”属于较早V3D阶段，V3F后续已实际完成新holdout与发布；训练资产未变化，不能把那一段当作当前全项目状态。



## 12. 评价系统与实验设计

<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 1000" role="img" aria-labelledby="ragscope-06-evaluation-title ragscope-06-evaluation-desc">
<title id="ragscope-06-evaluation-title">实验与评价：从问题、对照到可以支持的结论</title>
<desc id="ragscope-06-evaluation-desc">只使用真实模型日志作为真实实验；手工应用日志与冻结评测独立，失败样本不剔除。</desc>
<defs><marker id="ragscope-06-evaluation-arrow" markerWidth="9" markerHeight="9" refX="8" refY="4.5" orient="auto"><path d="M0,0 L9,4.5 L0,9 Z" fill="#77776f"/></marker></defs>
<rect width="100%" height="100%" rx="16" fill="#fdfcf9"/>
<text x="30" y="38" fill="#252522" font-size="24" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="700" text-anchor="start">实验与评价：从问题、对照到可以支持的结论</text>
<rect x="30" y="85" width="350" height="125" rx="10" fill="#eff2fc" stroke="#534ab7" stroke-width="1.6"/>
<text x="45" y="114" fill="#534ab7" font-size="19" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="700" text-anchor="start">数据与划分身份</text>
<text x="45" y="140" fill="#252522" font-size="15" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="400" text-anchor="start">训练 / 开发 / 冻结评估分离</text>
<text x="45" y="163" fill="#252522" font-size="15" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="400" text-anchor="start">问题、参考答案与证据坐标</text>
<text x="45" y="186" fill="#252522" font-size="15" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="400" text-anchor="start">推理输入不混入参考答案</text>
<rect x="425" y="85" width="350" height="125" rx="10" fill="#fff7e8" stroke="#a86a19" stroke-width="1.6" stroke-dasharray="7 5"/>
<text x="440" y="114" fill="#a86a19" font-size="19" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="700" text-anchor="start">开发选型</text>
<text x="440" y="140" fill="#252522" font-size="15" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="400" text-anchor="start">模型 / Prompt / 检索参数对照</text>
<text x="440" y="163" fill="#252522" font-size="15" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="400" text-anchor="start">有限预算与不利结果保留</text>
<text x="440" y="186" fill="#252522" font-size="15" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="400" text-anchor="start">选型后固定正式配置</text>
<rect x="820" y="85" width="350" height="125" rx="10" fill="#fff7e8" stroke="#a86a19" stroke-width="1.6" stroke-dasharray="7 5"/>
<text x="835" y="114" fill="#a86a19" font-size="19" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="700" text-anchor="start">固定实验条件</text>
<text x="835" y="140" fill="#252522" font-size="15" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="400" text-anchor="start">同题配对 B0 与最终配置</text>
<text x="835" y="163" fill="#252522" font-size="15" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="400" text-anchor="start">模型、索引、配置 SHA 与请求</text>
<text x="835" y="186" fill="#252522" font-size="15" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="400" text-anchor="start">真实后端响应 / 错误 / 耗时</text>
<polyline points="380,145 425,145" fill="none" stroke="#77776f" stroke-width="1.8" marker-end="url(#ragscope-06-evaluation-arrow)"/>
<polyline points="775,145 820,145" fill="none" stroke="#77776f" stroke-width="1.8" marker-end="url(#ragscope-06-evaluation-arrow)"/>
<rect x="30" y="315" width="350" height="155" rx="10" fill="#fff7e8" stroke="#a86a19" stroke-width="1.6" stroke-dasharray="7 5"/>
<text x="45" y="344" fill="#a86a19" font-size="19" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="700" text-anchor="start">检索评价</text>
<text x="45" y="370" fill="#252522" font-size="15" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="400" text-anchor="start">全库：200 题 × 3 方法 × 2 配置</text>
<text x="45" y="393" fill="#252522" font-size="15" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="400" text-anchor="start">共 1200 条；scope ID 移除</text>
<text x="45" y="416" fill="#252522" font-size="15" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="400" text-anchor="start">文档命中 / 完整证据覆盖</text>
<text x="45" y="439" fill="#252522" font-size="15" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="400" text-anchor="start">固定预算：检索好 ≠ 生成一定好</text>
<rect x="425" y="315" width="350" height="155" rx="10" fill="#fff7e8" stroke="#a86a19" stroke-width="1.6" stroke-dasharray="7 5"/>
<text x="440" y="344" fill="#a86a19" font-size="19" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="700" text-anchor="start">生成评价</text>
<text x="440" y="370" fill="#252522" font-size="15" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="400" text-anchor="start">H_QA 200 + H_tasks 20</text>
<text x="440" y="393" fill="#252522" font-size="15" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="400" text-anchor="start">B0 / 最终，共 440 条评分</text>
<text x="440" y="416" fill="#252522" font-size="15" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="400" text-anchor="start">EM / F1 / ROUGE-L / 语义余弦</text>
<text x="440" y="439" fill="#252522" font-size="15" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="400" text-anchor="start">拒答、数字与极性规则诊断</text>
<rect x="820" y="315" width="350" height="155" rx="10" fill="#fff7e8" stroke="#a86a19" stroke-width="1.6" stroke-dasharray="7 5"/>
<text x="835" y="344" fill="#a86a19" font-size="19" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="700" text-anchor="start">真实在线评价</text>
<text x="835" y="370" fill="#252522" font-size="15" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="400" text-anchor="start">每组 30 条，无答案/查询缓存</text>
<text x="835" y="393" fill="#252522" font-size="15" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="400" text-anchor="start">编码 + 检索 + 打包 + 真实生成</text>
<text x="835" y="416" fill="#252522" font-size="15" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="400" text-anchor="start">热请求 P50 / P95；保留预热</text>
<text x="835" y="439" fill="#252522" font-size="15" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="400" text-anchor="start">不将缓存命中当作模型速度</text>
<polyline points="995,210 995,260 195,260 195,315" fill="none" stroke="#77776f" stroke-width="1.8" marker-end="url(#ragscope-06-evaluation-arrow)" stroke-dasharray="7 5"/>
<polyline points="995,260 600,260 600,315" fill="none" stroke="#77776f" stroke-width="1.8" marker-end="url(#ragscope-06-evaluation-arrow)" stroke-dasharray="7 5"/>
<polyline points="995,260 995,315" fill="none" stroke="#77776f" stroke-width="1.8" marker-end="url(#ragscope-06-evaluation-arrow)" stroke-dasharray="7 5"/>
<rect x="230" y="575" width="740" height="145" rx="10" fill="#fff7e8" stroke="#a86a19" stroke-width="1.6" stroke-dasharray="7 5"/>
<text x="245" y="604" fill="#a86a19" font-size="19" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="700" text-anchor="start">汇总、配对变化与证据边界</text>
<text x="245" y="630" fill="#252522" font-size="15" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="400" text-anchor="start">正式逐题记录 → 分组统计 → 配对改善/退步 → 中文报告</text>
<text x="245" y="653" fill="#252522" font-size="15" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="400" text-anchor="start">保留旧版、失败与准确分母；语义余弦不是事实正确率</text>
<text x="245" y="676" fill="#252522" font-size="15" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="400" text-anchor="start">引用 ID 可见不是引用支持；自动标记不是人工确认的事实错误</text>
<polyline points="195,470 195,650 230,650" fill="none" stroke="#77776f" stroke-width="1.8" marker-end="url(#ragscope-06-evaluation-arrow)" stroke-dasharray="7 5"/>
<polyline points="600,470 600,575" fill="none" stroke="#77776f" stroke-width="1.8" marker-end="url(#ragscope-06-evaluation-arrow)" stroke-dasharray="7 5"/>
<polyline points="995,470 995,650 970,650" fill="none" stroke="#77776f" stroke-width="1.8" marker-end="url(#ragscope-06-evaluation-arrow)" stroke-dasharray="7 5"/>
<rect x="30" y="810" width="535" height="110" rx="10" fill="#edf7f2" stroke="#257455" stroke-width="1.6"/>
<text x="45" y="839" fill="#257455" font-size="19" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="700" text-anchor="start">工程功能验证</text>
<text x="45" y="865" fill="#252522" font-size="15" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="400" text-anchor="start">82 项测试；真实 API 与 Streamlit 组件验收</text>
<text x="45" y="888" fill="#252522" font-size="15" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="400" text-anchor="start">不作为模型质量、人工事实审核的替代</text>
<rect x="635" y="810" width="535" height="110" rx="10" fill="#edf7f2" stroke="#257455" stroke-width="1.6"/>
<text x="650" y="839" fill="#257455" font-size="19" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="700" text-anchor="start">独立应用交互</text>
<text x="650" y="865" fill="#252522" font-size="15" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="400" text-anchor="start">APPLICATION_*.jsonl；可选内存答案缓存</text>
<text x="650" y="888" fill="#252522" font-size="15" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="400" text-anchor="start">不回写正式成绩；浏览器视觉验收未完成</text>
</svg>

### 12.1 先明确“分数”在回答什么

RAG 至少有五道关：找对论文、找全证据、把证据放进模型窗口、依据证据正确作答、把引用关联到对应陈述。只看一个 F1 不能定位哪道关出问题。

|指标|计算对象和通俗含义|能说明什么|不能说明什么|
|---|---|---|---|
|Recall@k / Hit@k|标注相关对象有没有出现在前 k 项；单一相关对象时两者可能相同|检索有没有找到目标|模型最终答案是否正确|
|MRR|首个正确结果越靠前，分数越高|首个正确结果的位置|所有证据是否齐全|
|nDCG@k|对前 k 项相关性按位置折扣|排序整体质量|生成质量|
|完整参考证据覆盖|实际打包上下文是否完整覆盖至少一套有效标注证据|模型是否收到了足够的标注材料|模型是否使用了材料；未标注材料是否也能支持答案|
|连续证据覆盖|原文证据坐标被上下文覆盖的比例|丢失了多少标注证据字符|答案事实正确率|
|EM|规范化后的答案是否完全一致|严格的短答案一致性|换一种正确说法是否可接受|
|Token F1|答案与参考的词语重合，兼顾 precision / recall|遗漏参考用词、冗长和文字差异|“答对题比例”；数字和否定关系的事实正确率|
|ROUGE-L|最长公共子序列的重合|保序文字重合|语义等价与因果关系|
|BERTScore|基于上下文 token 向量的匹配|软化同义表达差异|科学事实真实性；也不等于最终版答案向量余弦|
|答案向量余弦|整段答案与参考答案编码后的夹角相似性|主题和语义的接近程度|数字、否定、对象等细节完全正确|
|引用可见率|引用 ID 是否存在于给模型的证据中|引用是否合法、能否定位|引用是否真的支持该陈述|
|NLI/支持性 proxy|裁判判断陈述是否被材料蕴含|带模型假设的支持性信号|未经校准的人类真值；UNKNOWN 不能删掉|
|P50 / P95|排序后中位数和95%分位耗时|典型及较慢请求的体验|小样本中可靠的线上 SLA；非流式总耗时也不是首 token 时间|

例如参考是“比基线低 3%”，回答“比基线高 3%”可能有很高文字和语义相似度，却把结论颠倒。因此最终版额外保留数字与 yes/no 极性提示，但这些规则只报告疑点，不自动认定事实错误。[实测] 最终 440 条评分全部仍标 `factual_correctness=UNKNOWN`、`human_review=NOT_REVIEWED`；这是一项明确的尚未完成能力，而非把未知计作正确。

聚合口径：[实测] V2/V3 可答题的答案 F1 中，适用但失败的请求保留在分母中并记零；不可答题答案 F1 是 N/A，另算正确拒答。V3 H_QA 200 题的 F1 分母是170，不是200；正确拒答分母30。对多个标注的答案或证据集合按预先定义的兼容方式处理，不能随实验方法临时挑更有利的参考。源码公式及正式 metric version 应与正文评价模块章节互相参照。

#### 文字指标的精确定义与三个反例

`generation_metrics.py` 会小写化、删除ASCII标点、删除英文冠词a/an/the并压空白；F1对规范化后的空格词项多重集合计算 `2·共同词数/(预测词数+参考词数)`，ROUGE-L以同一token序列的LCS计算F1，多参考取最佳值。当前V2算法无stemming，不能与旧版stemmed rouge-score结果静默混算。

本次只读调用现有纯函数，得到下面的数学例子（不是新模型实验）：

| 参考 | 预测 | token F1 | EM | 说明 |
|---|---|---:|---:|---|
| BERT | This paper uses BERT as the baseline. | 0.285714 | 0 | 若问题只问模型名，正确而冗长的表达也会被文字指标惩罚 |
| No significant improvement | Significant improvement | 0.8 | 0 | 结论反向却仍有很高词项重合 |
| 3 | -3 | 1.0 | 1.0 | ASCII负号被规范化删除，形成明确盲点 |

数字规则也只是正则提取字面数字，未完整解析正负号、单位、范围或科学记数关系；yes/no只检查特定短参考和开头极性，全大写实体规则会漏掉普通专名。`[代码]` `ragscope/v3/practical_quality.py`。因此同时报告文字、语义、证据与疑点，比只报F1好，但仍不能得到可靠事实正确率；这些评价缺陷在本次只记录，不修改算法或旧成绩。

### 12.2 V1：高召回首先证明“能找到相关段落”

**问题与设计。** 早期任务研究 BM25、Dense、Hybrid 和重排在 SQuAD 短段落检索上的差异，尚不等于完整论文生成问答。`rerun/per_query.csv` 含四条件各200题、共800行；每条件200个不同 qid。[实测] 从逐题 `r5/r10/mrr/ndcg10` 重算得到：

|V1 原始逐题文件的条件|N|Recall@5|Recall@10|MRR|nDCG@10|
|---|---:|---:|---:|---:|---:|
|Dense|200|0.865|0.900|0.756649|0.791785|
|BM25|200|0.845|0.905|0.726054|0.769604|
|Hybrid|200|0.915|0.950|0.788012|0.827977|
|Hybrid + Rerank|200|0.885|0.950|0.678490|0.745153|

重要勘误：`reports/retrieval_results.csv` 也把 Dense 的 Recall@10 写作0.900、BM25写作0.905。“向量检索0.905、混合0.950”对应下面固定版本 **E0 重建** 的 Dense/Hybrid，而不是这份 V1 原始逐题 CSV。两份记录应并列保留，不能为统一叙述覆盖历史。

**固定权重重建。** 原旧 Chroma 库只有384维与cosine等元数据，原权重 revision/SHA 未保存。后续 E0 没有拿新查询向量冒险查询未知向量空间，而是对相同22054个 legacy chunks 和200个问题，使用官方 `BAAI/bge-small-en-v1.5` 的固定 revision `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a`，CLS pooling、L2 归一化、CPU FP32、精确 NumPy cosine 重建；Hybrid 每路取20、alpha0.5、RRF常数60、最终10。[实测] `L0/dense.jsonl` 与 `hybrid.jsonl` 各200条均 `real_model=true`、零失败：

|E0 固定版本重建|N|Recall@5|Recall@10|MRR|nDCG@10|
|---|---:|---:|---:|---:|---:|
|Dense|200|0.870|0.905|0.761560|0.796696|
|Hybrid|200|0.915|0.950|0.797012|0.834733|

**支持的结论：** Hybrid 在这批短段落检索任务上有较高 Recall@10；重排不必然改善早期排序。**不支持的结论：** 不能把0.950宣传为95%生成回答正确；也不能把E0与原索引的微小差异归因于单独一个优化，因为权重版本和搜索引擎已经变化，旧权重身份仍 `[未知]`。

### 12.3 V2 E1：长论文让“如何切块和拼证据”真正成为变量

**要回答什么。** 旧短段落常常一个段落就是一个 chunk，调整切块策略不一定实际改变输入；长论文才会产生章节边界、跨块证据、overlap和父子扩展的真实取舍。E1 在相同100个dev问题、36篇论文上比较11种切块配置与BM25/Dense/Hybrid，再补8个扩展诊断；非重排正式记录为41条件×100=4100条，而非4100个独立问题。[实测] 原始路径为 `R2/records/E1_*.jsonl`；各条件92题可答、8题证据指标N/A。嵌入统一Qwen3-Embedding-0.6B Q8_0 CPU，2048参考token上下文，固定query指令与tokenizer。

关键同题对照复算如下：

|切块及Hybrid条件|完整参考覆盖 none → neighbor|连续覆盖 none → neighbor|
|---|---:|---:|
|F512_O0|0.663043 → 0.684783|0.777828 → 0.707951|
|R512|0.739130 → 0.750000|0.779438 → 0.755435|

**为什么两个指标反向。** 邻块能把原本缺的一段带进来，让某套多段证据终于完整；但窗口总预算固定，也可能挤掉其他原本覆盖的证据。这是输入组合的取舍，不是实现必然有错。[推断] 需要看逐题替换才知道是哪种机制。原配对审计中R512完整覆盖8题改善、7题下降、77题不变，净增1/92≈1.09个百分点；预声明工程容差0.01恰低于这个离散步长，故不能把“通过选型”当统计显著。

最终简单候选F512_O0与结构候选R512均选择neighbor，依据的是预声明Hybrid完整证据指标；父子P256_1024的Dense覆盖0.804348更高，也不能事后改规则只挑最高单元。**结论边界：** E1证明长论文切块能改变证据分布；不直接证明生成回答改善，不是全509篇索引的规模统计。下一个E6才把切块带入真实生成对照。

### 12.4 V2 E2—E7：模型规模、证据与生成协议必须一起解释

**共同条件。** 正式QASPER冻结train/dev/test问答1000/100/200，test有182可答、18不可答；独立SQuAD E4为100个可答题。Qwen3.5 2B/4B/9B与LFM2.5-1.2B使用已记录本地模型tag、量化、模板和权重身份。主生成为本地Ollama0.34.4、Q4_K_M、greedy、关闭thinking、最多512输出token；主上下文按2048参考token并进一步检查真实tokenizer窗口。不同模型模板与tokenizer不完全相同；LFM是异构对照，不是假设只有参数量不同。

**规模。** `[实测]` 本次从各 `records/*.jsonl` 聚合得主生成 **5520条、39组、32失败**。其中E2为3模型×4检索×200=2400；E3为3×2×200=1200；E4为3×100=300；E5为1×2×100=200；E6为2×2×100=400；E7为2×3×40=240；路由标签路径为200train×3与60dev×3=780。总和5520；预热、LoRA配对、在线专项不混入这个分母。

#### E2：三种Qwen规模 × 四种检索

这里R0=BM25、R1=Dense、R2=Hybrid、R3=Hybrid+rerank，均使用最终R512及固定证据组织。每个模型每路径200题，F1对182可答题适用；各E2请求零失败。下表均由 `G2E2/records/E2_qwen_*.jsonl` 重算。

|路径|完整证据覆盖|2B F1|4B F1|9B F1|2B/4B/9B生成P50秒|
|---|---:|---:|---:|---:|---|
|R0 BM25|0.6703|0.4244|0.3542|0.3057|1.090 / 2.259 / 3.499|
|R1 Dense|0.8077|0.4947|0.3765|0.3248|1.098 / 2.442 / 3.802|
|R2 Hybrid|0.7967|0.4822|0.3790|0.3325|1.079 / 2.464 / 3.652|
|R3 Hybrid+rerank|0.6593|0.4332|0.3501|0.3009|1.093 / 2.436 / 3.634|

**能得出的结论。** 同题同context条件下，本轮2B的文字F1比4B/9B高；R3丢失证据，并在三模型上均低于R2。**不能得出的结论。** 不等于“小模型普遍理解更强”；F1受回答长度、参考样式、拒答、模板、量化与任务约束影响。表中耗时是冻结上下文后的生成，R3重排成本在计时前已发生，不能据此说重排只要1秒。默认2B/固定Hybrid是dev先选，不是看到test中Dense最高才更改部署。

#### E3：把检索与生成短板拆开

同200题分别给完整标注Oracle证据与无上下文。Oracle覆盖1、No Context覆盖0；各条件零失败。

|模型|No Context可答F1|Oracle可答F1|
|---|---:|---:|
|2B|0.0415|0.5786|
|4B|0.0044|0.5455|
|9B|0.0000|0.4705|

Oracle高于R2，说明提供更好材料仍有提升空间；Oracle仍远非1，说明即便证据已经送到，生成/表达/评价仍有损失。空材料任务要求strict-grounded回答，拒答不能等价为模型完全没有内部知识。Oracle也比实际检索更短、更干净，不能把Oracle−检索全部解释为纯召回损失。来源：`G2E3/records/E3_qwen_*.jsonl`。

#### E4：回到短SQuAD，但这次评生成

各模型使用同100题Hybrid上下文，证据覆盖0.9700。2B/4B/9B生成F1分别 **0.7474 / 0.7606 / 0.5473**，失败分别0/2/0。这里4B略高于2B，再次说明模型排序依赖题库和协议；这组是答案F1，仍不能与V1 Recall@10的0.950直接比。来源：`G2/records/E4_qwen_*.jsonl`。

#### E5：LFM异构模型试验

LFM2.5-1.2B在100题BM25/Hybrid上F1为 **0.1971 / 0.1982**，分别13/16失败；证据覆盖约0.6957/0.8043。保留失败分母意味着这些数字描述整个已实现输出协议的可靠性，不是只评价“成功解析时的模型能力”。不同模板、架构、tokenizer也使其不属于纯参数量对照。来源：`G2/records/E5_lfm.jsonl`。

#### E6：切块是否真的改善生成

相同100题、92题可答，比较简单与结构化切块；两模型分别固定检索和生成条件，证据覆盖由0.7065到0.8043。

|模型|simple F1|structured F1|失败 simple / structured|
|---|---:|---:|---:|
|2B|0.4426|0.4837|1 / 0|
|4B|0.4114|0.4329|0 / 0|

该受控子集支持“这次结构切块和证据打包有益”的局部观察；不证明所有结构切块、所有数据都更好。来源：`G2/records/E6_qwen_*.jsonl`。

#### E7与4K/8K：位置诊断与资源诊断不同

E7给同40题完整证据、调整证据位置，2B start/middle/end F1约 **0.5527 / 0.5684 / 0.5876**，4B约 **0.5138 / 0.5209 / 0.5475**，均无失败。这里只支持这批固定提示中的描述性位置差异，不足以宣布一般lost-in-the-middle规律。来源：`G2/records/E7_qwen_*.jsonl`。

独立4K/8K压力试验各8正式+3预热，2B Q4实际输入3584/7680 tokens；正式生成均值约1.366/1.920秒、P95约1.648/2.319秒。它只测试这8个来源可行题的资源和窗口，不参与主质量排名，其他92题排除原因保留；同期间有CPU重排/编码活动，不是独占性能基准。来源：`D:/RAGScope_Runtime/runs/20261014_v2_context_stress/records/window_4096.jsonl`、`window_8192.jsonl`。

#### E8：单篇问答高覆盖，不代表全库检索已经解决

100题全库检索、无额外生成，完整证据覆盖BM25/Dense/Hybrid/Hybrid+rerank GPU约 **0.16 / 0.18 / 0.19 / 0.14**。这比给定论文范围时明显更难，说明“先找论文”是独立瓶颈。GPU重排与历史CPU重排条件不同，不能用其时间声称部署提速。原始路径：`R2/records/E8_R512_{bm25,dense,hybrid}_neighbor.jsonl`及`E8_R512_hybrid_rerank_neighbor_gpu.jsonl`。

### 12.5 V2路由：实现、训练、重放与在线收益是四件事

最早路由研究选择BM25、Hybrid或Hybrid+rerank三条检索路径，目标是节省成本同时控制答案损失；它不是现在“事实/比较/总结”任务意图分类的同一个问题。训练路径用了200train与60dev问题的三路真实生成构造代理标签。线性、MLP、LSTM确实训练过；Laya也不是Mock：`D:/RAGScope_Runtime/runs/20261014_v2_laya/records.jsonl`有 **1080条真实CPU决策、360问题×3候选顺序、646次低置信fallback**。这是决策运行证据，不是质量提升证据。

八策略为三固定路径、rules、linear、MLP、LSTM、Laya；在同dev60与test100的冻结真实结果上做1280次**重放选择**。重放复用既有答案，并把记录中的组件耗时相加；不是1280次新模型推理，更不是实际用户总耗时。三学习模型在dev/test都预测BM25；多数类标签分布train64%、dev51.667%，故训练/开发准确率差距不能单独解释成“过拟合”。

真正在线专项只选择dev上确定的fixed_hybrid和rules，在相同30个dev问题上各实跑一次，另6条预热；正式60条零失败。CPU实时查询编码/检索/打包与GPU2B生成处于已驻留条件。[实测] 从 `D:/RAGScope_Runtime/runs/20261014_v2_routing/online/records.jsonl` 重算：

|策略|正式N|可答N|答案F1|在线P50秒|在线P95秒|
|---|---:|---:|---:|---:|---:|
|固定Hybrid|30|26|0.408830|1.277657|2.026595|
|rules|30|26|0.411395|1.213204|2.201954|

P50降低约5.04%，P95反而较慢，且未达预设10%工程门槛，所以默认仍固定Hybrid。这里“未部署”是实验结论，不是“路由没有实现”；同样不能由负结果推出所有任务路由都无意义。

### 12.6 V3早期诊断：拆解负结果，而非宣称全面提升

#### 生成协议和规模倒挂

固定24题，按不可答/yes-no/抽取/自由回答各6题，分别用干净Oracle与冻结检索。G0沿用schema+greedy；G1加强最小充分回答；G2取消schema；G3在G1基础上采用固定官方采样配置与三个预先种子。共864条评分，含复用G0的144条真实响应与720条新主矩阵；G3每组72条是24题×3种子，不能称72个独立问题。另有direct/pipeline72次冒烟和补充42次。

在冻结检索下，2B G0/G1/G2/G3可答F1约 **0.5229 / 0.4819 / 0.3247 / 0.3126**；4B约 **0.2937 / 0.4461 / 0.3191 / 0.4157**；9B约 **0.2118 / 0.2766 / 0 / 0.2845**。G2检索组失败2B6/24、4B8/24、9B19/24；零分包含失败，不是声称9B在19题里完全不懂。更简洁提示对4B/9B有帮助，对2B没有统一帮助；去schema损害输出可靠性。

36对direct/pipeline冒烟输入输出一致、实际prompt token mismatch0；未发现这批固定夹具有thinking通道/token错配。不能据此排除所有包装问题，也不能把模型规模倒挂已确定为某个模板bug。来源：`V3D/S2_PROTOCOL_RESULTS.jsonl`、`S2_G0_REUSED_RESULTS.jsonl`、`S2_PROTOCOL_SCORED864.jsonl`及`S2_PROTOCOL_ANALYSIS.json`。

补充Oracle padded36条（12可答×3模型）保留原证据、加入同论文非标注段，padded−clean F1在2B/4B/9B为−0.0352/−0.1505/−0.0042；材料噪声与长度是混杂因素，但12题和未人工复核的“非标注段”不足以概括通用鲁棒性。额外6次H2O/一周天数知识探针回答正确，只说明两个简单问题。

#### 重排漏斗：排序靠前仍可能在窗口打包时丢掉证据

V3把候选集合、排名top10、真正pack进prompt三层分开。开发搜索40题上BGE/Qwen有改善线索，但独立D_confirm100题（92可答）里，Hybrid/ BGE max / BGE top2的完整打包覆盖约 **0.793478 / 0.652174 / 0.663043**；三者候选覆盖均1.0，排名top10仍约0.9783/0.9565/0.9565。也就是候选池内有证据，重排改变顺序后，在固定2048预算和邻块扩展中被较早片段挤掉。

这是“证据在哪一层丢失”的具体观察；短窗口相关性模型与长证据、按相关性排序与有限预算不是相同目标。[推断] 它解释为何高级模块可能负收益，但没有证明所有问题的唯一根因都是模型偏好短文本。BGE FP32/512窗口与Qwen BF16/≤8192输入条件不同，跨模型差不能只归因架构。原始：`V3D/S3_NATURAL_RERANK_bge_D_confirm.jsonl`；逐题漏斗核算：`S3_NATURAL_RERANK_D_confirm_ANALYSIS.json`。当前默认不启用Cross-Encoder重排。

core-first在D_confirm100的2B真实生成，B0/C1可答F1 **0.395329 / 0.407682**，增量约0.01235；论文聚类95%区间跨零，因此早期门槛没有据此发布“显著提升”。证据完整覆盖0.793478/0.826087。来源：`V3D/S3_SCOPED_GENERATION_D_confirm.jsonl`及独立ANALYSIS。

#### 任务工作流和分类不等于换一个生成模型

相同60题×W_local/W_compare/W_summary=180次真实生成，另9次独立预热；统一2B Q4/G0/2048证据/512输出。问题标签在看结果前冻结：fact28、explain22、compare5、other5，**没有summary标签**；52可答、8不可答。可答F1依次 **0.363394 / 0.340127 / 0.360794**。全部已经给定论文，因此W_clarify委托W_local，新增请求0，不得重复计作240次生成。

按题目类型的预标映射混合代理效用0.319528，低于最佳固定W_local的0.331608，差−0.012080；每题事后挑最好路径的hindsight上界0.419707不可部署，也不是分类准确率。因此停止新router训练。这个结果不能回答“若有充分分类标签，LSTM是否优于规则”：这轮没有总结题，比较仅5题；分类模型本身并未在新目标上公平训练对比。原始生成与效用向量：`V3D/S4_WORKFLOW_GENERATION.jsonl`、`S4_WORKFLOW_UTILITY_VECTORS.jsonl`。

另一次独立来源开发构造只接受3个可逐字引证任务（2总结/1比较），共6正式生成，保留构造失败。两总结F1约0.4695→0.4745和0.3481→0.6691，比较0.3360→0.3077；后者关键点输入覆盖0.6→0，说明“比较工作流支持”不等于“比较一定更好”。132条NLI关系实际完成，但仍有UNKNOWN。3题不足以支持通用结论。来源：`V3D/S4_DEVELOPMENT_TASK_SCORES_resident_v2.jsonl`、`S4_DEVELOPMENT_TASK_NLI_RELATIONS_resident_v2.jsonl`。

#### 为什么增加语义/事实评价仍未得到“正确率”

V3对旧5520主矩阵+200 LoRA共5720答案重算原文字算法，未发现算法不一致；以DeBERTa-v3-base第9层、CPU FP32、batch4、无IDF的BERTScore资源替代模型，实际完成6342个不同答案/参考对编码，无截断，最终按题聚合。这个裁判是按资源可行性替代，不能把结果理解成挑选“最能提高分数”的模型；也不能与常见RoBERTa-large BERTScore分数直接横比。来源：`V3D/BERTSCORE_PAIRS_nli_base_cpu_smallbatch.jsonl`、`BERTSCORE_ALL_OLD_nli_base_cpu_smallbatch.json`。

更复杂的claim/support/NLI链路有真实校准和部分实际评分，但全量旧答案并没有可靠完成。[实测] `CLAIM_ALL_OLD_COVERAGE_AND_UNKNOWNS.jsonl`共5720条，其中4715为资源未运行、335为部分实际CPU NLI、670为失败/拒答按合同质量记零；5042条claim F1仍未知，并保留UNKNOWN区间。自动裁判既受资源限制，也受自然问答适用性和错误传播影响。不能删除未知，只在成功子集上宣称faithfulness提高。最终收尾另用Qwen答案向量余弦与字面数字/极性提示，降低运行成本；仍诚实保留事实真值未知。

### 12.7 V3收尾：完整索引与最终工作台的正式比较

这里必须与早期诊断分开。早期 `docs/execution/v3/EXECUTIVE_SUMMARY_ZH.md` 写Dense不全、H_QA未跑、execution_complete=false，在其当时是事实；后续 `V3F` 已补全1585论文/19673块向量并完成正式比较，不能把前一报告的未完成状态当现在默认。

#### 先开发选型，再查看正式留出结果

收尾检索开发使用D_search120与D_confirm100，6recipes各220题；D_confirm完整覆盖B0 **0.793478**、CORE **0.826087**、DENSE75 **0.858696**、SPARSE75 **0.760870**、META **0.804348**、DENSE_META **0.858696**。这只是更好的标注材料输入，不自动意味着更好的回答。原始：`V3F/DEV_RETRIEVAL.jsonl`。

另用同20个已暴露dev问题检验生成，实际80条：2B旧G0 F1 **0.491916**，2B新协议0.367962、4B新协议0.409922、9B新协议0.429152；均0失败。额外2B CONTROL/ADAPT各20真实请求，F1 **0.433624 / 0.383764**，仍不如原G0。最终因此保留2B及普通QA旧策略，将新证据组织用于全库、总结和比较。9B还受8GB线上驻留预算约束，但不能声称它完全没实际运行。来源：`DEV_GENERATION.jsonl`、`QA_PROMPT_CONTROL.jsonl`、最终`GENERATION_SELECTION.json`；中间`GENERATION_SELECTION_DEV80.json`曾写4B，属于候选快照，不能当当前默认。

#### H_QA200与H_tasks20：最终结果及分母

两条件B0与FINAL均在**同一扩展语料和GPU embedding空间**运行；B0是移植的V2策略，不是原CPU部署的计时基线。H_QA200中170可答、30不可答；H_tasks20为14总结/6比较，AI构造且原文引句核验，**不是官方或人工审核gold**。最终评分共440行：旧220条B0 + 当前resident8K220条FINAL，历史较早FINAL与混合窗口响应保留作诊断，未删除。

后续220条最终部署记录中196条新真实调用、24条与既有真实请求payload完全一致而复用；故“220评分”不等于“这一步又新发220请求”。首次矩阵440、混合部署220及resident部署220各自有原文件和来源身份。B0全部220条采用4096窗口、2048参考证据tokens和512输出上限；FINAL中196道普通QA为8192窗口、2048参考证据tokens和512输出上限，4道复杂QA及20道总结/比较题为8192窗口、6144参考证据tokens和768输出上限。因此总结/比较是**整体部署策略对照**，增益不能单独归因于证据选择算法；FINAL均另查实际模型8192窗口能否容纳完整请求。

|最终正式指标|H_QA B0|H_QA FINAL|H_tasks B0|H_tasks FINAL|
|---|---:|---:|---:|---:|
|正式记录数|200|200|20|20|
|F1/EM/ROUGE/语义适用数|170|170|20|20|
|失败数|0|0|0|0|
|Token F1|0.390401|0.406514|0.427401|0.458162|
|EM|0.176471|0.182353|0|0|
|ROUGE-L|0.364498|0.375359|0.267961|0.275915|
|答案/参考向量余弦|0.746144|0.752492|0.812471|0.859379|
|完整证据覆盖|0.711765|0.717647|N/A|N/A|
|关键点进入上下文比例|N/A|N/A|0.574468|1.000000|
|不可答正确拒答|3/30|3/30|N/A|N/A|
|可答题错误拒答|0/170|0/170|0/20|0/20|

H_QA同题F1：25改善、17退步、128不变；H_tasks：14改善、6退步。任务94个标注关键点最终全部进入上下文，但“100%输入覆盖”不等于“94点被模型全部正确说出”。最终任务原文章符覆盖均值约97.35%，也不是全文理解准确率。语义余弦为同一Qwen编码器的归一化答案/参考向量相似度，没有query prefix；**不是BERTScore，也不是事实正确率**。

H_QA两组yes/no参考极性疑点均9条，数字疑点4→5；任务数字疑点1→3。这些规则有漏检和误报，不能因为F1增加就忽略它们。所有440条事实真值仍UNKNOWN、人工NOT_REVIEWED。引用可见率均1.0，但H_QA只有196/192条有适用引用，其余不能从该比例隐去；schema本来约束引用ID，所以不能宣传“事实支持率100%”。

**结论。** 普通QA提升很小，总结/比较的语义接近与证据覆盖改善更清晰；没有做独立人评和更多留出，不宣布普遍能力突破。正确拒答10%是当前显著短板。结果来源：`V3F/FINAL_SCORES_RESIDENT8K.jsonl`、`DEPLOYMENT_FINAL_ANSWERS_RESIDENT8K.jsonl`、`FINAL_GENERATION.jsonl`，本稿独立聚合与`FINAL_SUMMARY_RESIDENT8K.json`一致。

**留出解释边界。** 配置冻结文件声明生成前未读H_QA参考，首次正式矩阵有固定输入；但之后又根据输出/运行行为修正长任务上下文及窗口并重评，因此最终版不应被包装成“一次性、完全盲测且从未反馈”的独立发布成绩。H_tasks参考由AI构造，人工状态未变化。可以准确称“固定任务上的最终部署对照”，不要称外部竞赛排行榜成绩。

#### 全库1200条检索：证据改善与文档排名退步同时存在

同H_QA200×3检索×2配置，先移除论文scope，在完整1585篇/19673块上搜索，无答案生成；170可答题的证据指标适用。固定2048上下文。原始：`V3F/FINAL_GLOBAL_RETRIEVAL.jsonl`，下表逐条重算。

|方法|B0文档Hit@10|FINAL文档Hit@10|B0完整证据|FINAL完整证据|
|---|---:|---:|---:|---:|
|BM25|0.435|0.480|0.100000|0.170588|
|Dense|0.325|0.270|0.100000|0.105882|
|Hybrid|0.405|0.380|0.111765|0.176471|

Hybrid完整覆盖约11.18%→17.65%，同时文档Hit@10由40.5%→38%。前者评价真正送进模型的材料，后者评价前10结果是否包含目标论文，不能只报上升项。且QASPER原问题多为给定论文场景，去scope后“this paper”可能含糊，所以该实验是移除scope的检索压力测试，不是未经限定的全库真实用户正确率。

#### 在线速度：修复反复加载，并未全面超过基线

真实在线包含未缓存查询编码、检索、打包、生成与解析，答案缓存和查询向量缓存均关闭。每条件30题，另2条预热；resident8K原始60行全部real_model=true、0失败。B0仍用4096模型窗口、FINAL全8192，这不是两组相同窗口的纯算法成本对照。

|正式线上条件|N|P50秒|P95秒|
|---|---:|---:|---:|
|同轮B0|30|1.166037|1.478669|
|最终FINAL，统一8192驻留|30|1.187435|1.988290|

较早FINAL混用4096/8192时P95 **10.131892秒**，backend load_duration发现切换窗口反复加载，统一8192后P95 **1.988290秒**。这是修复自身重载卡顿的收益，不是声称比B0的1.478669秒更快。冷加载不混入热服务性能；接口缓存命中约毫秒级属于另一条件，不能拿来代替真实未缓存性能。来源：`V3F/FINAL_ONLINE.jsonl`、`FINAL_ONLINE_RESIDENT8K.jsonl`。

最终总结/比较输出更长，冻结生成P50约2.625→4.370秒，均值输出token275.75→501；这是更多材料/更长答案的实际成本。主F1表中有精确请求复用，所以最终部署速度以在线专项为准，不能用评分阶段的缓存和来源行耗时拼一个更好数字。

### 12.8 统计、失败与验收边界

V2主要分析按原论文/原文章进行2000次cluster bootstrap，seed42，95% percentile区间，避免把同论文多题当完全独立。多重探索比较没有校正，未校正区间不应包装成确定因果或显著优胜。V3早期core-first论文聚类区间跨零；最终440对照主要是描述性固定任务比较，没有新的人工正确率置信区间。

失败至少分成：加载前资源守卫、模型后端错误、超时/OOM、结构JSON解析错误、不可见引用、答案内容疑点、证据不足。加载前守卫不是CUDA OOM；格式通过不是事实正确。训练loss、单元测试、模型推理、线上请求、浏览器视觉检查的证据等级也不能互换。

V3收尾82项回归、9个真实API验收、Streamlit组件+真实API集成、离线220条历史阅读验证有各自产物；它们证明软件和实际请求可运行，不是额外质量benchmark。浏览器视觉验收曾因访问权限被拒，未完成像素级检查；不能写成“完整浏览器验收通过”。对应原始产物：`V3F/FINAL_PYTEST_DELIVERY.xml`、`API_REAL_ACCEPTANCE.json`、`UI_REAL_COMPONENT_INTEGRATION.json`、`UI_OFFLINE_FINAL.json`。

### 12.9 对项目成果的准确表达

**工程成果：** 从短段落检索扩展为完整论文、章节/坐标映射、可恢复编码、完整1585论文索引、范围明确的单篇/多篇/全库问答、真实本地模型、引用展示、逐题实验账本、隔离环境和资源释放。这些是“做出了什么系统”。

**质量成果：** 在固定任务中结构切块有局部正收益；最终普通QA F1小幅上升、总结比较文字/语义和证据覆盖改善；全库证据覆盖改善但准确性仍低。不能说每一个新模块都改善质量。

**有效负结果：** 参数更大未在固定协议中更好；重排可能挤掉证据；三路动态路由不足以抵消开销；旧LoRA降低F1并增加等待；NLI裁判并不能自动变成事实真值。记录这些结果，能把“为什么没有默认开启”解释为有证据的决策，而不是项目毫无成果或所有高级方法无效。

**仍不能宣称：** 95%问答准确率、100%事实支持、全库高准确率、LoRA提升、智能路由提速达标、9B普遍不如2B、人工审核完成、最终440评分等于440次全新推理。简历适合写“构建并验证可复现的本地论文RAG实验平台，完成多维受控评估，识别检索打包与生成协议瓶颈”，并单独列有适用范围的真实结果。



## 13. 工程与运行环境


### 模型资产不是同时加载的模型

[代码] 下表记录现有本地资产及角色。GGUF与safetensors可能是同家族的不同表示；历史下载的视觉projector仍保留，但文本专用manifest不加载它。不能把“有模型文件”升级为“在默认问答中启用”。

|模型|资产与体积|角色/实际状态|
|---|---|---|
|Qwen3.5-2B Q4_K_M|`models/ollama/blobs/sha256-20cb277...`，语言权重1,274,396,992字节；文本manifest `models/generation/text_only/qwen3.5_2b-q4_K_M.json`|当前默认生成；模型tag `ragscope-qwen3.5:2b-q4_K_M`，私有Ollama0.34.4；非LoRA合并模型|
|Qwen3.5-4B Q4_K_M|语言权重2,648,593,216字节；同目录4b manifest|V2模型尺度实验、V3开发对照；当前非默认|
|Qwen3.5-9B Q4_K_M|语言权重5,629,109,120字节；同目录9b manifest|V2模型尺度实验、V3开发对照；单卡串行，当前非默认|
|LiquidAI LFM2.5-1.2B-Instruct Q4_K_M|语言权重730,895,168字节；`models/generation/lfm2.5_q4_k_m.json`|V2异构小模型对照；不是当前默认；许可清单写LFM Open License v1.0/local noncommercial research，不是所有模型统一Apache|
|Qwen3-Embedding-0.6B Q8_0|`models/qwen3-embedding-0.6b/Qwen3-Embedding-0.6B-Q8_0.gguf`，639,150,592字节|当前1024维/L2检索编码，pooling由GGUF后端定义；V2有CPU llama.cpp b11429实现，V3 GPU编码桥；具体当前调用由V3运行层决定|
|BGE-small-en-v1.5|`models/bge-small-en-v1.5/5c38ec7c405ec4b44b94cc5a9bb96e735b38267a`；config为BertModel/hidden384/max_position512|历史Embedding/CPU比较；当前主索引不是该384维向量|
|BGE-reranker-base|`models/bge-reranker-base/2cfc18c9415c912f9d8155881c133215df768a70`；权重1,112,206,140字节，download_manifest完整|V2实际重排/CPU成本对照；当前默认关闭重排|
|Laya|`models/laya/7b928d828b7b0e022f929d9bd2e44165aa270148`；权重payload约842.6MB，421,293,830个元素（含buffer）；ModernBERT encoder hidden1024/layers28|实际CPU路线选择实验；加载时FP32，不能把F16文件体积当驻留内存；当前未启用|
|Qwen3.5-2B BF16|`models/Qwen3.5-2B/15852e8c16360a2fea060d615a32b45270f8a8fc`；官方单分片4,548,221,488字节，12文件总4,571,272,453字节|真实LoRA训练/同精度比较；与Q4部署文件独立，不能共用权重哈希|
|旧LoRA adapter|`TRAIN/checkpoints/step-000122/adapter/adapter_model.safetensors`，14,798,456字节，SHA `a532dc22942d84e4dceabacb64b936c191b37ebdfa9fda092f3f87464d0bbae4`|已训练/恢复/同精度运行，但不默认使用；未成功发布合并副本|
|RoBERTa-large、DeBERTa-v3-base MNLI/FEVER/ANLI|`models/v3_metrics/FacebookAI__roberta-large/722cf37b1afa9454edce342e7895e588b6ff1d59`；`models/v3_metrics/MoritzLaurer__DeBERTa-v3-base-mnli-fever-anli/6f5cf0a2b59cabb106aca4c287eed12e357e90eb`|高级评价资产。仅文件存在不代表NLI/ BERTScore所有正式行已完成；最新实用语义分数也不能冒称BERTScore|
|Qwen3-Reranker-0.6B|`models/v3_reranker/Qwen__Qwen3-Reranker-0.6B/e61197ed45024b0ed8a2d74b80b4d909f1255473`，权重1,191,588,280字节，config为BF16 Qwen3ForCausalLM|V3候选重排资产/研究路线；不在最终默认问答路径|

### Python 为什么分环境

[代码] 三个主要uv环境core/models/train的 `pyvenv.cfg` 都为CPython3.12.13、uv0.11.19、`include-system-site-packages=false`，base位于 `D:/RAGScope_Runtime/tools/python/cpython-3.12-windows-x86_64-none`。不是WSL Python，也不是系统全局Python。环境分开实际解决的是依赖版本冲突及轻量UI与模型库的隔离：CPU BGE/Laya保留Transformers4.57.6，Qwen3.5训练需要5.19.0/CUDA版torch。

|环境|本次直接metadata可见的主要依赖|用途与位置|
|---|---|---|
|core|74直接包；Streamlit1.65.0、tokenizers0.23.3、NumPy2.5.4、psutil7.2.2；无torch/transformers|默认API/UI/运行协调，`envs/core/Scripts/python.exe`|
|core_acceptance_20261014|同类74包，无torch/transformers|独立验收环境，非默认启动要求|
|models|72直接包；torch2.9.1+cpu、Transformers4.57.6、sentence-transformers5.2.0、Laya0.4.6、tokenizers0.22.2、Streamlit1.65.0|CPU历史模型/路由；`envs/models/requirements.lock.txt`，SHA `4248ab17be28f682c46c8425870a5e5361ca55ad19a37ef5342dc246356addee`|
|train|47直接包；torch2.9.1+cu128、Transformers5.19.0、PEFT0.21.2、Accelerate1.15.0、tokenizers0.23.3|真实BF16 LoRA/部分GPU重排；当前47包锁为 `envs/train/requirements.r3_gpu.lock.txt`，SHA `1a955cc8f6db69ff61ba4ebaa2b918ec06a6b7ab1f4054142e13e997d54bc742`|
|reports|11直接包，不含torch|绘图/报告依赖|
|v3_metrics|6直接分发包；bert-score0.3.13；通过.pth读取train+reports依赖，因此可见CUDA torch/Transformers5|stdlib venv创建，`envs/v3_metrics/Lib/site-packages/ragscope_v2_readonly_dependencies.pth`|
|v3_metrics_cpu|1直接分发包bert-score0.3.13；通过.pth读取models+reports依赖，因此可见CPU torch/Transformers4|CPU评价兼容环境，同名.pth保存借用路径|

[实测] 本次使用stdlib `importlib.metadata.distributions(path=...)`读取安装元数据，没有import torch或初始化CUDA。`configs/v3/dependencies.lock.json`还保存了各环境实际解释器观察和合成后的可见版本，主环境版本与本次读取一致。core和reports未各自放完整pip freeze文本；不能伪造所有环境都拥有requirements.lock.txt。项目 `requirements-v2.lock` 与V3的JSON清单、环境内历史freeze分别覆盖不同范围。

[文档] 冲突 `docs/execution/TRAIN_ENV_AUDIT.md` 末尾的“当前46包”描述准备阶段；现有train已补入rank-bm25，直接分发包47，旧 `requirements.lock.txt` 的SHA为 `e2cd66b2...`，而新 `requirements.r3_gpu.lock.txt`才对应后续GPU重排需要。旧锁没有删除，但不能拿旧46包记录宣称当前完整环境一致。两个metrics环境也不应称为再下载两套完整Torch，.pth明确复用已有依赖；其独立性是新增包与入口隔离，而非全部底层包物理隔离。

### 运行资产为何集中D盘

[代码] `scripts/runtime_env.ps1:1` 设置当前PowerShell进程及子进程环境：HF/PyTorch/SentenceTransformers/uv/pip/tokenizer缓存、TEMP/TMP、CUDA/torch扩展/inductor/Matplotlib目录都指向 `D:/RAGScope_Runtime`。不修改系统全局环境。默认置 `CUDA_VISIBLE_DEVICES=''`、OMP/MKL/OpenBLAS/NumExpr四线程；显式GPU分支先通过lease验证再开放设备。`PYTHONDONTWRITEBYTECODE=1`避免额外pycache。

[代码] 最新工作台用core Python通过HTTP控制本地私有模型后端，core没有torch并不妨碍GPU推理，因为真正计算在独立模型服务进程中。当前后端窗口统一8192，不代表每道题都输入8K：普通QA参考证据最多2048、输出512；比较/总结参考证据最多6144、输出768。窗口、参考证据预算、实际token输入是不同数字。

[实测] 边界 文件体积、目录逻辑大小、硬链接去重后的大小、磁盘物理占用与运行时RAM/VRAM不是同一指标。模型身份manifest+blob布局会复用相同文件；原视觉projector仍可落盘但文本tag不加载。只读现有清单能说明资产位置，不能把整D盘用量变化全部归因于RAGScope，也不能由“模型文件1.27GB”推出“推理只需1.27GB显存”。



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

<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 1100" role="img" aria-labelledby="ragscope-03-online-title ragscope-03-online-desc">
<title id="ragscope-03-online-title">在线：一次真实问题的函数调用链</title>
<desc id="ragscope-03-online-desc">当前完整工作台经本地 API 使用 FullWorkbench；并非旧 Workbench 或 RAGPipeline.answer 全链路。</desc>
<defs><marker id="ragscope-03-online-arrow" markerWidth="9" markerHeight="9" refX="8" refY="4.5" orient="auto"><path d="M0,0 L9,4.5 L0,9 Z" fill="#77776f"/></marker></defs>
<rect width="100%" height="100%" rx="16" fill="#fdfcf9"/>
<text x="30" y="38" fill="#252522" font-size="24" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="700" text-anchor="start">在线：一次真实问题的函数调用链</text>
<rect x="55" y="90" width="485" height="120" rx="10" fill="#edf7f2" stroke="#257455" stroke-width="1.6"/>
<text x="70" y="119" fill="#257455" font-size="19" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="700" text-anchor="start">① 工作台输入</text>
<text x="70" y="145" fill="#252522" font-size="15" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="400" text-anchor="start">ui/full_workbench.py</text>
<text x="70" y="168" fill="#252522" font-size="15" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="400" text-anchor="start">question / doc_ids / task / method</text>
<rect x="665" y="90" width="485" height="120" rx="10" fill="#edf7f2" stroke="#257455" stroke-width="1.6"/>
<text x="680" y="119" fill="#257455" font-size="19" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="700" text-anchor="start">② API 接收</text>
<text x="680" y="145" fill="#252522" font-size="15" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="400" text-anchor="start">scripts/v3/serve_full_workbench.py</text>
<text x="680" y="168" fill="#252522" font-size="15" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="400" text-anchor="start">POST /api/answer；串行执行</text>
<polyline points="540,150 665,150" fill="none" stroke="#77776f" stroke-width="1.8" marker-end="url(#ragscope-03-online-arrow)"/>
<polyline points="1150,150 1180,150 1180,235 30,235 30,320 55,320" fill="none" stroke="#77776f" stroke-width="1.8" marker-end="url(#ragscope-03-online-arrow)"/>
<rect x="55" y="270" width="485" height="120" rx="10" fill="#edf7f2" stroke="#257455" stroke-width="1.6"/>
<text x="70" y="299" fill="#257455" font-size="19" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="700" text-anchor="start">③ 请求与范围核验</text>
<text x="70" y="325" fill="#252522" font-size="15" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="400" text-anchor="start">FullWorkbench.answer</text>
<text x="70" y="348" fill="#252522" font-size="15" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="400" text-anchor="start">空问题 / 无效论文 / 歧义对象处理</text>
<rect x="665" y="270" width="485" height="120" rx="10" fill="#edf7f2" stroke="#257455" stroke-width="1.6"/>
<text x="680" y="299" fill="#257455" font-size="19" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="700" text-anchor="start">④ 查询真实编码</text>
<text x="680" y="325" fill="#252522" font-size="15" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="400" text-anchor="start">ReusableEncoder.query(cached=True)</text>
<text x="680" y="348" fill="#252522" font-size="15" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="400" text-anchor="start">同身份查询缓存；Qwen 1024 维</text>
<polyline points="540,330 665,330" fill="none" stroke="#77776f" stroke-width="1.8" marker-end="url(#ragscope-03-online-arrow)"/>
<polyline points="1150,330 1180,330 1180,415 30,415 30,500 55,500" fill="none" stroke="#77776f" stroke-width="1.8" marker-end="url(#ragscope-03-online-arrow)"/>
<rect x="55" y="450" width="485" height="120" rx="10" fill="#edf7f2" stroke="#257455" stroke-width="1.6"/>
<text x="70" y="479" fill="#257455" font-size="19" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="700" text-anchor="start">⑤ 任务与证据选择</text>
<text x="70" y="505" fill="#252522" font-size="15" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="400" text-anchor="start">deployment_context</text>
<text x="70" y="528" fill="#252522" font-size="15" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="400" text-anchor="start">普通问答 / 总结 / 比较 / 多篇</text>
<rect x="665" y="450" width="485" height="120" rx="10" fill="#edf7f2" stroke="#257455" stroke-width="1.6"/>
<text x="680" y="479" fill="#257455" font-size="19" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="700" text-anchor="start">⑥ 完整索引检索</text>
<text x="680" y="505" fill="#252522" font-size="15" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="400" text-anchor="start">FullCorpusRetriever.search</text>
<text x="680" y="528" fill="#252522" font-size="15" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="400" text-anchor="start">范围过滤 / BM25 + Dense / RRF</text>
<polyline points="540,510 665,510" fill="none" stroke="#77776f" stroke-width="1.8" marker-end="url(#ragscope-03-online-arrow)"/>
<polyline points="1150,510 1180,510 1180,595 30,595 30,680 55,680" fill="none" stroke="#77776f" stroke-width="1.8" marker-end="url(#ragscope-03-online-arrow)"/>
<rect x="55" y="630" width="485" height="120" rx="10" fill="#edf7f2" stroke="#257455" stroke-width="1.6"/>
<text x="70" y="659" fill="#257455" font-size="19" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="700" text-anchor="start">⑦ 原文证据打包</text>
<text x="70" y="685" fill="#252522" font-size="15" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="400" text-anchor="start">pack_context / answer_messages</text>
<text x="70" y="708" fill="#252522" font-size="15" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="400" text-anchor="start">参考预算 2048 或最多 6144 token</text>
<rect x="665" y="630" width="485" height="120" rx="10" fill="#edf7f2" stroke="#257455" stroke-width="1.6"/>
<text x="680" y="659" fill="#257455" font-size="19" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="700" text-anchor="start">⑧ 本地真实生成</text>
<text x="680" y="685" fill="#252522" font-size="15" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="400" text-anchor="start">DeploymentClient.chat → 私有 Ollama</text>
<text x="680" y="708" fill="#252522" font-size="15" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="400" text-anchor="start">2B / 后端窗口 8192 / 输出 512 或 768</text>
<polyline points="540,690 665,690" fill="none" stroke="#77776f" stroke-width="1.8" marker-end="url(#ragscope-03-online-arrow)"/>
<polyline points="1150,690 1180,690 1180,775 30,775 30,860 55,860" fill="none" stroke="#77776f" stroke-width="1.8" marker-end="url(#ragscope-03-online-arrow)"/>
<rect x="55" y="810" width="485" height="120" rx="10" fill="#edf7f2" stroke="#257455" stroke-width="1.6"/>
<text x="70" y="839" fill="#257455" font-size="19" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="700" text-anchor="start">⑨ 解析与引用检查</text>
<text x="70" y="865" fill="#252522" font-size="15" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="400" text-anchor="start">RAGPipeline.parse_output</text>
<text x="70" y="888" fill="#252522" font-size="15" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="400" text-anchor="start">JSON 字段 / 可见 chunk ID / token 核验</text>
<rect x="665" y="810" width="485" height="120" rx="10" fill="#edf7f2" stroke="#257455" stroke-width="1.6"/>
<text x="680" y="839" fill="#257455" font-size="19" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="700" text-anchor="start">⑩ 返回与可追溯记录</text>
<text x="680" y="865" fill="#252522" font-size="15" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="400" text-anchor="start">answer / citations / abstain / 原文</text>
<text x="680" y="888" fill="#252522" font-size="15" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="400" text-anchor="start">独立 APPLICATION_*.jsonl 保留失败</text>
<polyline points="540,870 665,870" fill="none" stroke="#77776f" stroke-width="1.8" marker-end="url(#ragscope-03-online-arrow)"/>
<text x="55" y="1026" fill="#252522" font-size="17" font-family="Microsoft YaHei, PingFang SC, Segoe UI, sans-serif" font-weight="400" text-anchor="start">模型异常或 JSON/引用校验失败：记录 EXECUTION_ERROR；不会用 Mock 冒充真实答案。</text>
</svg>

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


## 17. 面试问题与个人学习清单

这些问题用于练习把原理、实际实现和证据边界连起来。回答时先说“我做了什么”，再说“哪些条件下有效”，最后承认尚未验证的部分。下面的源码路径绑定本次手册版本；没有定位到稳定函数行号的条目只列相关模块，不编造行号。实验目录采用第0章定义的 `V3FINAL`、`V3EARLY`、`V2RUN`；LoRA与路由原始目录见第11章和附录E。

### Q01 怎样在两分钟内介绍这个项目？

- **面试官问法：** “RAGScope解决什么问题？与普通聊天机器人有什么区别？”
- **通俗答案：** 它帮助我阅读本地论文：先找原文，再根据原文回答，并把引用展示出来。我同时建设了实验记录，能追踪每道题到底检索到什么、输入模型什么、回答是否失败。
- **技术答案要点：** 外部论文语料、结构切块、BM25与Dense融合、范围和任务控制、量化本地生成、JSON引用及逐条实验账本组成可复核链路。检索与生成解耦，真实模型运行和功能测试分开。
- **项目真实例子：** [实测] 当前覆盖1585篇/19673块，支持全库、单篇和多篇范围；原509篇目标缺失0。它尚不是已经验证可靠的事实裁判，也不默认支持任意PDF上传、OCR或图表理解。
- **源码位置：** `ragscope/v3/full_retrieval.py`、`ragscope/v3/full_workbench.py`、`ui/full_workbench.py`、`configs/v3/default.json`。
- **还需补充知识：** RAG与参数化知识的边界、实际用户需求评估、从本地实验系统走向生产服务所需的鉴权与并发能力。
- **现有证据是否足够：** 足够说明系统已实现并实际运行；不足以承诺任意论文、任意问题都能准确回答。

### Q02 一次用户提问怎样沿真实代码执行？

- **面试官问法：** “不要只画框图，按实际调用链讲一下。”
- **通俗答案：** 页面收集问题与论文范围，服务检查范围，检索相关材料并控制长度，模型写出答案，最后校验格式和引用，再展示原文。
- **技术答案要点：** 当前入口是FullWorkbench而非把历史RAGPipeline全流程直接套入；查询编码与索引身份必须一致，任务策略决定普通或复杂上下文，生成客户端保存实际请求，解析复用旧Pipeline的输出检查。
- **项目真实例子：** [实测] 最终真实API验收包含全库问题、超出旧141篇范围的论文、多篇比较、总结、无效输入及缓存场景。组件测试也真正请求过模型，浏览器视觉验收另有权限限制，不能混为一谈。
- **源码位置：** `ui/full_workbench.py` → `scripts/v3/serve_full_workbench.py` → `ragscope/v3/full_workbench.py` → `full_retrieval.py` / `deployment.py` / `finish_runtime.py` → `ragscope/generation/llm_client.py`；解析为 `RAGPipeline.parse_output`。
- **还需补充知识：** HTTP请求生命周期、故障传播、为什么同步本地服务不能直接宣称支持高并发。
- **现有证据是否足够：** 足够追踪当前默认链路；不支持生产级吞吐或多用户隔离的结论。

### Q03 为什么早期Recall接近0.95，现在答案F1却只有约0.4？

- **面试官问法：** “功能增加这么多，怎么分数反而低了？”
- **通俗答案：** 以前测的是能不能找回正确资料，现在测的是能不能读懂长论文并写出参考答案。找到书与答对考试不是同一个难度，也不是同一个分数。
- **技术答案要点：** V1短段落文档Recall@k与V3长论文生成token F1的对象、数据分布和评价阶段均不同。长文还需要多证据覆盖、预算打包和生成，误差会逐层累积。
- **项目真实例子：** [实测] V3最终同题对照中，普通可答QA F1为0.3904→0.4065，总结/比较0.4274→0.4582；这些才是相同条件下的有效比较，不拿它们与SQuAD文档Recall作升降百分比。
- **源码位置：** `ragscope/evaluation/retrieval_metrics.py`、`ragscope/evaluation/generation_metrics.py`、`ragscope/v3/practical_quality.py`；原始结果 `V3FINAL/FINAL_SUMMARY_RESIDENT8K.json`。
- **还需补充知识：** 检索指标与端到端指标、难度迁移、指标量纲及比较条件。
- **现有证据是否足够：** 足够证明两类数字不可直接比较，并描述本轮小幅增益；不足以证明项目整体已达到高可靠质量。

### Q04 为什么要切块，块越小是不是越精确？

- **面试官问法：** “直接把整篇论文向量化不好吗？chunk size怎么选？”
- **通俗答案：** 整篇像一本书的简介，容易丢掉具体细节；小块更容易定位，但太小又会把原因、条件和结果拆散。所以块大小是定位精度与上下文完整性的取舍。
- **技术答案要点：** 切块影响向量语义粒度、BM25长度分布、候选冗余和上下文预算。不能只按检索命中选chunk size，还要检查实际输入的完整参考证据覆盖与生成质量。
- **项目真实例子：** [代码] [实测] 当前R512由结构切分、Qwen tokenizer、max_tokens512、overlap0构建；另有固定、句子及父子候选。V1语料约99.89%一文一块，因此旧短文实验难以验证长论文切分策略。
- **源码位置：** `ragscope/ingestion/chunkers.py` 的 `FixedSizeChunker`、`SentenceAwareChunker`、`StructureRecursiveChunker`、`ParentChildChunker`。
- **还需补充知识：** token长度分布、检索粒度与答案所需证据粒度、长文分层表示。
- **现有证据是否足够：** 足够说明当前算法与研究取舍；不足以宣称512是所有任务的最优大小。

### Q05 项目里的“Semantic Chunker”真的用语义相似度切界吗？

- **面试官问法：** “你怎么计算相邻段落的语义边界？”
- **通俗答案：** 当前没有做那种计算。代码中的兼容名称叫Semantic，但实际按段落、标题、列表等结构切分，太长再按句子或固定长度切。
- **技术答案要点：** 结构启发式与embedding相似度阈值切界是不同算法。名称不能作为实现证据；需要指出真正的分段规则以及fallback。
- **项目真实例子：** [代码] `SemanticChunker = StructureRecursiveChunker`。没有相邻句向量距离、动态阈值或边界模型，不能在简历写“实现相似度驱动语义切块”。
- **源码位置：** `ragscope/ingestion/chunkers.py` 的 `StructureRecursiveChunker` 与 `SemanticChunker`兼容别名。
- **还需补充知识：** 真正语义分段的距离度量、阈值选择、主题转折评估及其额外编码成本。
- **现有证据是否足够：** 足够回答当前真实实现；不具备相似度驱动切块的实验依据。

### Q06 Overlap有什么作用，为什么还要保存原文坐标？

- **面试官问法：** “相邻块重复了，怎么避免重复证据占满窗口？”
- **通俗答案：** 重叠避免一句话被边界切坏，但搜索后不能把同一段反复塞给模型。原文坐标相当于给每段标页码和位置，方便去重、引用与检查覆盖。
- **技术答案要点：** 块保存canonical `[start,end)`、文档和内容身份；打包时减去已覆盖区间并重新投影证据span。token边界与字符边界不同，BPE也不是简单可加，必须按实际文本重新计数。
- **项目真实例子：** [代码] ContextBundle保留去重、跳过原因、unit_spans与context_hash；部分重叠移除后生成新区间引用ID并保留source_chunk_id，而不是复制文本后丢掉原始位置。
- **源码位置：** `ragscope/ingestion/chunkers.py` 的 `BaseChunker._assemble`、`_overlap_start`；`ragscope/generation/context.py` 的 `pack_context`。
- **还需补充知识：** 区间并集/差集、Unicode坐标、不同tokenizer下重计数与证据映射。
- **现有证据是否足够：** 足够说明可追溯和去重机制；不能保证引用区间中的文字一定蕴含答案。

### Q07 Embedding模型与生成模型有什么区别？

- **面试官问法：** “为什么需要两个Qwen模型？一个不能全部做完吗？”
- **通俗答案：** 一个负责把文字变成适合搜索的坐标，另一个负责读材料后写答案。就像图书馆的索引员与回答问题的人，工作目标不同。
- **技术答案要点：** embedding输出固定维连续表征，用于相似度；生成器逐token输出词表分布。训练目标、输入协议、缓存、模型身份及评估指标均不同。
- **项目真实例子：** [实测] 当前Qwen3-Embedding-0.6B Q8_0输出1024维归一化向量；Qwen3.5-2B Q4_K_M生成JSON。BGE-small历史384维不能直接放进当前1024维索引。
- **源码位置：** `ragscope/indexing/local_embedding.py`、`ragscope/v3/finish_runtime.py` 的 `ReusableEncoder`、`ragscope/generation/llm_client.py`。
- **还需补充知识：** 双塔表征学习、对比学习、生成式语言建模与embedding任务的区别。
- **现有证据是否足够：** 足够说明角色和实际身份；没有自行训练基础embedding或生成模型的证据。

### Q08 查询与文档为什么要用不同的编码输入？

- **面试官问法：** “query instruction有什么用？文档也加同一句吗？”
- **通俗答案：** 问题和答案段落写法不同。给编码器说明“我要找能回答科学论文问题的段落”，可以让它按检索任务理解问题；原文段落则保持相应文档端协议。
- **技术答案要点：** 指令式embedding常采用非对称query/document输入。前缀是编码协议的一部分，必须随模型身份和缓存记录，不能随意改变后还复用旧向量。
- **项目真实例子：** [代码] 查询前加 `Instruct: Given a question about a scientific paper, retrieve relevant passages that answer the query\nQuery:`；当前正文Dense为R512 body。答案语义相似度编码不加这个检索问题前缀。
- **源码位置：** `configs/v2/retrieval.json` 的 `query_instruction`；`ragscope/indexing/local_embedding.py`、`ragscope/v3/practical_quality.py`。
- **还需补充知识：** 非对称检索、任务指令敏感性、同模型不同输入协议的校准。
- **现有证据是否足够：** 足够回答实际输入；本轮没有单独前缀消融，不能声称该前缀贡献了具体提升。

### Q09 为什么向量归一化后可以直接做点积？

- **面试官问法：** “你用的是余弦距离还是内积？二者什么时候等价？”
- **通俗答案：** 如果把每个向量长度都缩成1，比较点积就只是在比较方向，相当于余弦相似度。
- **技术答案要点：** `cos(q,d)=q·d/(||q||||d||)`；单位L2归一化时分母为1。先检查维度、有限值和非零范数，否则NaN或零向量会破坏排序。
- **项目真实例子：** [代码] [实测] 最终矩阵19673×1024、float32，全部有限且单位L2；FullCorpusRetriever在允许行上做NumPy矩阵乘，当前没有默认FAISS/HNSW近似搜索。
- **源码位置：** `ragscope/v3/full_retrieval.py`、`ragscope/v3/finish_runtime.py`；`V3FINAL/FULL_DENSE_INDEX.json`、`FULL_CORPUS_VALIDATION.json`。
- **还需补充知识：** 高维空间、归一化与向量模长信息、精确搜索和近似最近邻的成本差异。
- **现有证据是否足够：** 足够说明当前实现；不支持百万级近似检索或召回损失的性能结论。

### Q10 为什么缓存不能只按“模型名+文本”存？

- **面试官问法：** “同一个Qwen模型，为什么还要记录那么多哈希？”
- **通俗答案：** 同名模型也可能换权重、量化、分词器或运行方式。旧坐标与新坐标混用后，搜索还能跑，但结果可能悄悄错掉。
- **技术答案要点：** 缓存身份包含revision、权重SHA、tokenizer、量化、后端、归一化等；索引还要固定chunk行序与语料身份。完整性校验应读实际文件字节，不只看文件名。
- **项目真实例子：** [实测] 补全索引复用13045个通过身份校验的向量、新算6628个；一次在线实验误写24条旧query缓存后已按冻结向量逐位恢复，并保存原SHA核验。`cached=False`在ReusableEncoder中并不等于绝不写缓存。
- **源码位置：** `ragscope/indexing/array_index.py` 的 `ArrayCache`；`scripts/v3/complete_dense.py`；`ragscope/v3/online_runtime.py` 的 `UncachedOnlineEmbedding`；`QUERY_CACHE_RESTORATION.json`。
- **还需补充知识：** 内容寻址、缓存失效、数据版本与计算身份、确定性和数值近似的边界。
- **现有证据是否足够：** 足够说明已发生的问题和修复；不能保证不同后端仅因维度相同就兼容。

### Q11 BM25为什么不只是统计关键词出现次数？

- **面试官问法：** “TF-IDF和BM25有什么区别？k1、b控制什么？”
- **通俗答案：** 生僻而关键的词应更重要；重复太多次不应无限加分；长文也不能只因为词多就占优势。BM25把这几件事放进同一个打分公式。
- **技术答案要点：** IDF体现词的区分度；词频饱和由k1调节；长度归一由b调节。TF-IDF是相关思想，但当前正文检索不是路由TF-IDF分类器。
- **项目真实例子：** [代码] 当前压缩BM25使用k1=1.5、b=.75；按词表列累加预计算权重，负IDF按平均值修正。标识符或精确数据集名可能适合稀疏信号，但不是所有题都胜过Dense。
- **源码位置：** `ragscope/v3/compressed_index.py` 的 `CompressedBM25`；`configs/v2/retrieval.json`；历史 `ragscope/retrieval/bm25_retriever.py`。
- **还需补充知识：** BM25推导、词频饱和、IDF变体、分词和领域词表的影响。
- **现有证据是否足够：** 足够说明参数与实现；没有本轮k1/b系统寻优的证据。

### Q12 为什么Hybrid用RRF，而不是直接把BM25和余弦分数相加？

- **面试官问法：** “融合公式是什么？alpha有什么含义？”
- **通俗答案：** 两个检索器的分数刻度不一样，直接相加容易被数值更大的那一路主导。RRF先看排名，再给靠前结果加分。
- **技术答案要点：** 当前 `Σ w_i/(60+rank_i)`，rank从1开始；alpha是Dense权重，另一路为1-alpha。未入某路候选的对象没有该路贡献；常数60不能说是本轮发现的最优值。
- **项目真实例子：** [代码] 普通指定论文QA alpha=.5；复杂指定任务用.75；全库还加入metadata BM25和文档先验。它们是不同任务配置，不是全项目只有一个融合权重。
- **源码位置：** `ragscope/v3/full_retrieval.py`；`ragscope/retrieval/hybrid_retriever.py`；`ragscope/v3/deployment.py`。
- **还需补充知识：** 排名融合、分数归一化、RRF常数与候选深度的联动。
- **现有证据是否足够：** 足够解释公式和当前配置；Hybrid的互补原理不等于每道题都提升。

### Q13 当前全库检索是不是“先找10篇，再在这10篇里找段落”？

- **面试官问法：** “你做的是哪种分层检索？第一阶段错了怎么办？”
- **通俗答案：** 当前默认不是这种硬筛选。块候选仍从全库产生，论文层结果只是给相关论文中的块增加先验信号。
- **技术答案要点：** 区分硬级联候选裁剪与软文档先验融合。前者第一阶段漏文档后无法恢复；后者仍有全库块候选，但先验可能改变排序，也会有误导风险。
- **项目真实例子：** [代码] [实测] FullCorpusRetriever在19673块上融合并加document_score；早期HierarchicalRetriever是另一实现。最终Hybrid完整证据覆盖11.18%→17.65%，但doc_hit@10为40.5%→38%，不能只报有利指标。
- **源码位置：** `ragscope/v3/full_retrieval.py`、`ragscope/v3/hierarchical_runtime.py`；`V3FINAL/FINAL_GLOBAL_SUMMARY.json`。
- **还需补充知识：** 级联误差、先验校准、候选多样性、文档命中与多段证据覆盖的关系。
- **现有证据是否足够：** 足够说明最新默认架构及结果；不足以称全库检索准确性已解决。

### Q14 Cross-Encoder与双塔Dense有什么不同？

- **面试官问法：** “既然重排更精细，为什么不用它直接搜索全库？”
- **通俗答案：** Dense可以提前给所有段落算坐标，提问时只算一次问题；Cross-Encoder要把问题和每个候选一起重新阅读，更细但更贵。
- **技术答案要点：** 双塔独立编码，支持预计算；交叉编码器允许query与document token交互，最终分依赖这对输入，不能预先为每个文档存一个固定相关性分数。一般只在候选集上重排。
- **项目真实例子：** [代码] [实测] BGE-reranker用于V2真实重排研究，另保留Qwen3-Reranker研究资产；当前FullCorpusRetriever不调用Cross-Encoder。
- **源码位置：** `ragscope/retrieval/reranker.py`、`ragscope/retrieval/verified_reranker.py`、`ragscope/v3/full_retrieval.py`。
- **还需补充知识：** cross-attention与序列分类目标、候选深度和token截断、reranker分数校准。
- **现有证据是否足够：** 足够解释角色与项目状态；资产存在不等于候选重排已用于默认服务。

### Q15 重排为什么反而降低了证据覆盖？

- **面试官问法：** “是你代码写错，还是模型不行？”
- **通俗答案：** 重排器可能把单段很相关的内容排前面，却挤掉回答所需的另一段。模型最终能读的材料有长度上限，排序变了，实际证据也会变。
- **技术答案要点：** 候选相关性、集合互补性、邻块扩展与固定预算打包不是同一个目标。需要分别控制候选池、顺序、长度、章节和多证据需求，才能定位因果。
- **项目真实例子：** [实测] V2路线重放dev60中，Hybrid证据覆盖0.767857、F1 .429777；重排覆盖.663690、F1 .370988。覆盖下降是实测，“模型不懂跨段关系”只是推断，不能直接归咎外部模型bug。
- **源码位置：** `ragscope/retrieval/verified_reranker.py`、`ragscope/generation/context.py`；`20261014_v2_routing/REPLAY_fixed_hybrid*.json`。
- **还需补充知识：** 集合选择、多样性、证据互补性评价、相关性与可回答性的差别。
- **现有证据是否足够：** 足够解释为何默认关闭；不足以确定唯一退步原因或证明所有重排无效。

### Q16 检索命中了证据，为什么模型还是答不出来？

- **面试官问法：** “你怎么区分retrieval failure和generation failure？”
- **通俗答案：** 搜到的段落不一定真正进入模型；进入了也不代表信息完整；材料完整也不代表模型读对。所以必须把每个环节分别记录。
- **技术答案要点：** 候选召回→实际ContextBundle→完整参考证据覆盖→生成输出→事实支持是不同判定。Oracle context用于去掉检索误差，但只隔离特定条件，不能变成部署成绩。
- **项目真实例子：** [代码] [实测] pack_context可因预算整段跳过，neighbor也可能挤掉后续核心；ContextBundle保存跳过原因和unit_spans。当前未用模型事实判断把所有答案自动判真。
- **源码位置：** `ragscope/generation/context.py` 的 `pack_context`；`ragscope/evaluation/evidence.py`；`ragscope/v3/practical_quality.py`。
- **还需补充知识：** 分阶段错误归因、必要与充分证据、oracle实验的控制变量。
- **现有证据是否足够：** 足够定位一些失败层次；自然语言事实支持仍有UNKNOWN，不能把全部错误完整自动归因。

### Q17 Parent-Child与邻块扩展解决同一个问题吗？

- **面试官问法：** “为什么做两种扩展？当前哪种在用？”
- **通俗答案：** 两者都补语境，但父子方式找小块后返回它所属的大段，邻块方式补前后内容；补得太多都会占掉别的证据位置。
- **技术答案要点：** parent_id描述预定义父块；prev/next描述同文档相邻块。扩展后的预算、去重和证据覆盖必须重算。更大上下文不是无成本的质量保证。
- **项目真实例子：** [代码] 项目保留P256_1024等父子候选，当前默认R512没有父块替换；普通指定论文QA仍用neighbor，部分新工作流采用core-first并限制邻居剩余预算。
- **源码位置：** `ragscope/ingestion/chunkers.py` 的 `ParentChildChunker`；`ragscope/generation/context.py`；`ragscope/v3/finish_runtime.py`。
- **还需补充知识：** 层级片段检索、窗口扩展与信息冗余、预算约束下的证据选择。
- **现有证据是否足够：** 足够区分实现与默认状态；不能把父子检索写成最新每道题都执行的核心步骤。

### Q18 总结与事实问题为什么不能共享完全相同的检索流程？

- **面试官问法：** “把top-k调大不就能总结了吗？”
- **通俗答案：** 事实题常只需要一个局部答案，总结要覆盖方法、实验和局限。泛泛的总结问题容易反复搜到摘要，top-k更大也未必覆盖整篇。
- **技术答案要点：** 任务所需证据范围不同。当前summary按原文顺序、方法/实验/结论等结构线索选块，多篇交织，预算6144参考token、768输出；这不是训练出来的摘要规划器。
- **项目真实例子：** [实测] 最终20道总结/比较题关键点进入上下文的比例57.45%→100%，文字F1 .4274→.4582；参考内容由AI构造并校验原文，引入小样本和非人工审核局限，100%也不是答案正确率。
- **源码位置：** `ragscope/v3/deployment.py` 的 `deployment_context`；`ragscope/v3/full_workbench.py`；`V3FINAL/FINAL_SUMMARY_RESIDENT8K.json`。
- **还需补充知识：** 摘要覆盖与显著性、跨文档平衡、claim/keypoint评估及人工标注。
- **现有证据是否足够：** 足够证明小规模任务的证据与代理分数改善；不足以推广为所有论文的高质量全文总结。

### Q19 JSON schema和引用枚举能消除幻觉吗？

- **面试官问法：** “引用合法率100%，是不是答案都有依据？”
- **通俗答案：** 它能要求模型按格式写，并只引用给过的编号；但模型仍可能引用对了编号、解释错了内容。
- **技术答案要点：** 结构约束、引用可见性、引用蕴含性是三层。schema控制语法和可选ID；实际事实支持需要把具体claim与对应证据比较，甚至人工审核。
- **项目真实例子：** [代码] [实测] 当前输出只有answer/citations/abstain，多篇限制最多4引用与1600字符；应用曾保留1次JSON截断失败，修正后复测通过。未宣称引用enum实现了自动事实裁判。
- **源码位置：** `ragscope/v3/full_workbench.py`、`ragscope/generation/pipeline.py` 的 `RAGPipeline.parse_output`、`ragscope/v3/practical_quality.py`。
- **还需补充知识：** 约束解码、schema表达能力、claim-to-evidence entailment、引用精确率与完整率。
- **现有证据是否足够：** 足够证明格式与可见性检查；不足以证明全部引用支持全部答案陈述。

### Q20 为什么写了“证据不足就拒答”，模型仍会勉强回答？

- **面试官问法：** “你们的拒答效果怎么样？怎么评？”
- **通俗答案：** 提示词只是要求，模型未必能准确判断自己有没有足够证据。要专门测不可答题，不能把格式失败当拒答，也不能把所有拒答都当正确。
- **技术答案要点：** 不可答的answer F1保留N/A，单独评correct abstention；可答但拒答还要结合当前材料充分性，不能只凭全论文有答案判定模型胡乱拒答。
- **项目真实例子：** [实测] 最新H_QA30道不可答仅3道正确拒答，10%没有改善。旧LoRA在92道可答题中新拒答23次，但其中一些检索材料不足，人工充分性审核未完成。
- **源码位置：** `ragscope/evaluation/generation_metrics.py`、`ragscope/routing/core.py` 的 `proxy_quality`、`ragscope/v3/practical_quality.py`。
- **还需补充知识：** 选择性预测、置信度校准、answerability与evidence sufficiency、拒答precision/recall的成本取舍。
- **现有证据是否足够：** 足够确认拒答仍是主要短板；没有可靠拒答阈值或事实安全保证。

### Q21 Q4、Q8和BF16分别表示什么，为什么不能直接比较它们的效果？

- **面试官问法：** “你证明了量化导致模型变差吗？”
- **通俗答案：** 它们是保存或计算数值的不同精细程度。更低位数通常更省空间，但不能只看到两次分数不同，就说一定是量化造成，因为后端、提示和格式约束也可能变了。
- **技术答案要点：** GGUF Q4_K_M/Q8_0是量化方案，不表示激活和KV cache全是4/8位；BF16 LoRA用Transformers eager，Q4部署用Ollama且有schema。要隔离量化需要同权重来源、协议、任务、上下文及后端尽可能匹配。
- **项目真实例子：** [代码] [实测] 默认生成Q4、embedding Q8，真实LoRA训练BF16；同BF16基础/adapter比较能隔离adapter开关，而跨Q4/BF16的差异不能全部归因于精度。
- **源码位置：** `ragscope/training/run.py` 的 `load_training_model`；`ragscope/training/evaluate.py` 的 `BF16ModelClient`；`RUNTIME/models/generation/text_only/`清单。
- **还需补充知识：** 量化误差、分组缩放、混合精度、KV cache精度与校准数据。
- **现有证据是否足够：** 足够说明实际表示和实验混杂；没有单因素量化质量损失结论。

### Q22 GPU显存为什么不等于模型文件大小？

- **面试官问法：** “1.27GB的2B模型为什么仍需要更多运行内存？”
- **通俗答案：** 文件只是模型参数。运行还要放中间结果、上下文记忆和计算工作区，训练又要加梯度、优化器和保存状态，所以不能按文件大小算显存。
- **技术答案要点：** 推理包含权重、KV cache、激活和临时workspace；训练包含反向激活、adapter梯度和optimizer state。Torch allocated/reserved、物理VRAM、进程RSS是不同口径，Windows还可能涉及共享内存。
- **项目真实例子：** [实测] BF16 LoRA峰值allocator allocated约7.61GB、reserved约8.28GB，原报告明确不是物理显存计量。最新服务只串行调用一个生成模型与独立encoder，不借此宣称模型体积等于驻留开销。
- **源码位置：** `ragscope/training/run.py` 的 `train_loop`资源记录；`ragscope/v3/local_runtime.py`、`ragscope/v3/finish_runtime.py`；LoRA `TRAINING_AUDIT.json`。
- **还需补充知识：** KV cache随层数/序列长度增长的估算、allocator缓存、prefill/decode资源差异。
- **现有证据是否足够：** 足够解释测量边界；没有把所有瞬时峰值和物理分页行为完整剖析的证据。

### Q23 为什么统一8K窗口反而解决了一次严重卡顿？

- **面试官问法：** “窗口变大不是更慢吗？你们究竟优化了什么？”
- **通俗答案：** 问题不只是读得多，而是普通题和总结题切换不同窗口时，服务反复重新加载模型。统一驻留设置后，避免了这部分重复等待。
- **技术答案要点：** 后端模型驻留配置变化可能触发reload；区分实际输入长度与num_ctx配置，区分冷启动与热请求，查看backend load_duration才能定位卡顿，而不只比较总时长。
- **项目真实例子：** [实测] 混用4K/8K版本FINAL在线P95约10.13秒；统一8192后约1.99秒。但同轮B0基线P95约1.48秒，所以结论是修复重载卡顿，不是最终配置全面更快。
- **源码位置：** `ragscope/v3/deployment.py`、`scripts/v3/finish_online.py`；`V3FINAL/FINAL_ONLINE_SUMMARY_RESIDENT8K.json`及旧在线日志。
- **还需补充知识：** 冷热分布、模型生命周期、后端缓存与配置键、P95在小样本中的不稳定性。
- **现有证据是否足够：** 足够定位这次重载问题和修复效果；不足以给出长期生产SLA或TTFT承诺。

### Q24 为什么分多个Python环境，core没有Torch怎么用GPU？

- **面试官问法：** “多环境是不是把依赖管理搞复杂了？”
- **通俗答案：** 页面、CPU历史模型和新训练库需要不同依赖。页面只发送本地请求，真正的GPU计算由另一个模型服务进程执行，所以页面环境不必安装Torch。
- **技术答案要点：** core负责协调/UI；models保留CPU Torch2.9.1与Transformers4.57.6；train使用cu128 Torch和Transformers5.19.0/PEFT。metrics环境通过只读.pth复用底层依赖，不是各复制一套完整Torch。
- **项目真实例子：** [代码] 七个环境均Python3.12.13。train当前47直接包，对应 `requirements.r3_gpu.lock.txt`；旧准备文档46包与旧lock属于早期状态，不能当当前完整锁。
- **源码位置：** `scripts/runtime_env.ps1`；`configs/v3/dependencies.lock.json`；`RUNTIME/envs/*/pyvenv.cfg`与环境锁；`scripts/v3/serve_full_workbench.py`。
- **还需补充知识：** 进程边界、ABI/版本兼容、虚拟环境和物理文件复用、可重复安装的锁定策略。
- **现有证据是否足够：** 足够描述当前依赖与分工；.pth借用意味着底层环境更新仍会影响metrics，不能称完全独立隔离。

### Q25 LoRA和QLoRA有什么区别，你真正做了哪一种？

- **面试官问法：** “用了低位模型就叫QLoRA吗？”
- **通俗答案：** LoRA是在冻结模型上学习一小组修正参数；QLoRA还把训练底座量化来节省内存。项目的训练底座是BF16，因此是LoRA；部署模型量化并不让这次训练自动变成QLoRA。
- **技术答案要点：** 低秩更新 `W+(alpha/r)BA`，参数规模约r(m+n)而非mn。QLoRA需要量化训练底座与相应数值计算流程；本项目没有以GGUF Q4文件反向训练。
- **项目真实例子：** [实测] Qwen3.5-2B BF16底座，r8/alpha16/dropout.05，114个目标模块、228个A/B张量，共3691008个可训练参数。adapter约14.8MB，不代表训练只占14.8MB内存。
- **源码位置：** `configs/v2/lora.json`；`ragscope/training/run.py` 的 `discover_attention_targets`、`load_training_model`；LoRA `model_observed.json`。
- **还需补充知识：** 矩阵低秩近似、rank与适配容量、量化训练、optimizer内存。
- **现有证据是否足够：** 足够说明真实训练方法；没有QLoRA实训、量化adapter发布或从零训练底座的证据。

### Q26 LoRA数据和训练配置如何保证不是“随便跑了一下”？

- **面试官问法：** “训练多少条、多少步、学了哪些参数？”
- **通俗答案：** 我冻结数据和配置，先保证每条训练样本的答案与证据配对完整，再做有上限的训练，并保存每步结果和恢复状态。
- **技术答案要点：** QASPER train最多1000、dev loss最多60；超1024完整样本丢弃而不截答案。实际972训练/59验证，microbatch1×累积8、AdamW1e-4、warmup.03、seed42、梯度裁剪1、gradient checkpointing。一轮实际ceil(972/8)=122步，未跑满配置上限200。
- **项目真实例子：** [实测] train394篇、dev31篇，文档交集0；841可答参考证据、131真实首段负例；最佳dev loss为step122的.135663。0.8B仅是配置后备路径，不能算已训练第二个模型。
- **源码位置：** `ragscope/training/data.py`、`prepare.py`、`run.py`；`20261014_v2_lora/prepared_2b_1024_v2/`与 `train_2b_1024_seed42/`。
- **还需补充知识：** 样本筛除偏差、有效batch、优化器步与microbatch区别、验证集选择偏差。
- **现有证据是否足够：** 足够证明一次具体可复核训练完成；规模有限，不足以代表充分训练或最优超参数。

### Q27 Assistant-only mask为什么重要，怎样验证？

- **面试官问法：** “你是不是把问题和参考材料也当目标训练了？”
- **通俗答案：** 训练应该主要教模型怎样回答，而不是要求它把给定材料照抄一遍。因此输入材料仍参与理解，但不会作为需要预测得分的目标部分。
- **技术答案要点：** prompt label置−100、assistant和结束标记保留真实token ID；使用实际chat template检查prompt是完整序列的精确前缀，避免凭字符串长度猜assistant起点。
- **项目真实例子：** [实测] 对972/59样本实际input_ids/labels/attention_mask重新核验，长度与mask通过；V3还解码20个监督目标并做固定8夹具两步学习smoke。证明监督链路有效，不证明真实问答泛化提升。
- **源码位置：** `ragscope/training/data.py:12` 的 `encode_assistant_only`；`MASK_AND_SPLIT_AUDIT.json`；`V3EARLY/S5_SUPERVISION_DISTRIBUTION_DIAGNOSIS.json`。
- **还需补充知识：** teacher forcing、交叉熵ignore_index、模板边界与EOS监督。
- **现有证据是否足够：** 足够反驳“整体mask完全失效”；不排除其他训练分布或标注质量问题。

### Q28 完整checkpoint恢复为什么不只是保存adapter？

- **面试官问法：** “断电后重跑能真的从同一步继续吗？”
- **通俗答案：** adapter只存学到的参数；要接着训练，还要记住优化器、学习率、随机状态以及已经读过哪些样本，否则只能算从相似权重重新开始。
- **技术答案要点：** adapter、optimizer、scheduler、Python/NumPy/Torch/CUDA RNG、data_order/data_position、step/epoch一起保存；临时目录写全并fsync、manifest校验后原子提交；恢复逐项精确比较。
- **项目真实例子：** [实测] step5与10均留下RESTORED_VERIFIED，adapter/optimizer/scheduler/RNG/顺序/位置一致。当前保留最新step120/121/122与best122；恢复收据存在不等于所有历史checkpoint都永久留存。
- **源码位置：** `ragscope/training/checkpoint.py` 的 `CheckpointStore`；`ragscope/training/run.py` 的 `train_loop`、`_restore_rng`、`_state_equal`。
- **还需补充知识：** 原子写入、故障一致性、确定性训练、梯度累积中的安全保存边界。
- **现有证据是否足够：** 足够证明两次实际完整恢复；不能承诺任意库版本/硬件变更后仍位级复现。

### Q29 验证loss下降，为什么LoRA最终问答反而变差？

- **面试官问法：** “是不是过拟合？你怎么证明？”
- **通俗答案：** loss衡量给定正确前文时预测目标词的能力，真实问答还要求检索、选择证据、判断拒答并完整生成。loss变好不代表整个任务都变好，也不能仅凭这两个数字断言过拟合。
- **技术答案要点：** 区分teacher-forced dev loss与free-generation质量；分布、输出长度、拒答校准、引用验证和参考表达都可能影响。因果结论需要匹配分布和协议的消融。
- **项目真实例子：** [实测] 同BF16100题、92可答的token F1 .464889→.426683；新增23次可答拒答是负向贡献之一，但EM .184783→.217391。训练prompt中位350.5token、旧评测2147.5，存在描述性分布差异，尚未证明唯一成因。
- **源码位置：** `ragscope/training/evaluate.py`；`V3EARLY/LORA_OLD_RECOUNT.json`、`S5_SUPERVISION_DISTRIBUTION_DIAGNOSIS.json`。
- **还需补充知识：** 自回归暴露偏差、分布迁移、拒答校准、多个质量指标与过拟合诊断。
- **现有证据是否足够：** 足够确认这次F1负结果；不足以证明“LoRA普遍有害”或锁定单一原因。

### Q30 LoRA为什么变慢，合并后有没有验证成功？

- **面试官问法：** “是不是EOS错了导致一直生成？merge不就解决了吗？”
- **通俗答案：** 我分别检查了输出是否更长、每个token是否更慢和停止是否正常。结果支持长度增加与未合并修正层开销；合并探针没有过预定数值容差，所以没有宣称合并成功。
- **技术答案要点：** 自然停止24题与固定64-token微基准解决不同问题；排除warmup、同步计时、区分首token与后续decode。merge需验证数值差异，不只看能否调用函数。
- **项目真实例子：** [实测] 新24题两分支全部正常EOS；输出token中位67→171。3题固定64输出的后续速度12.5634→8.5900 token/s。merge的max_abs .50–.625超过.08、relativeL2 .033–.050超过.01，未保存合并快照，也未跑合并正式对照。
- **源码位置：** `scripts/v3/run_lora_runtime_diagnostics.py`；`V3EARLY/S5_FIXED64_*.jsonl`、`S5_MERGE_LOGIT_GATE.json`。
- **还需补充知识：** prefill/decode、低秩分支计算、BF16舍入、logits容差与行为等价。
- **现有证据是否足够：** 足够支持特定微基准机制与合并失败事实；3题不能成为在线P95或所有输入结论。

### Q31 路由分类器训练的“正确标签”从哪里来？

- **面试官问法：** “你怎么知道一题该用BM25还是Hybrid？”
- **通俗答案：** 不是凭人工感觉给标签，而是让同一个生成模型在三条检索路线都答一遍，再按答案代理质量和成本规则选。但这仍只是该实验条件下的标签，不是绝对真理。
- **技术答案要点：** 三路线同配置真实生成；可答用F1、不可答用正确拒答；质量差≤.01时优先低耗时，再看证据覆盖。标签依赖生成器、上下文、prompt和代理指标，不能叫客观最优路线。
- **项目真实例子：** [实测] train200标签128 BM25/42 Hybrid/30重排，dev60为31/23/6；固定2B Q4生成条件。测试数据不用于fit或选路由器。
- **源码位置：** `ragscope/routing/core.py` 的 `build_proxy_labels`、`proxy_quality`、`validate_training_splits`；`configs/v2/routing.json`。
- **还需补充知识：** 代价敏感学习、弱监督标签、选择偏差与代理目标错配。
- **现有证据是否足够：** 足够复查标签来源；不足以把分类accuracy解释为最终答案正确率。

### Q32 线性、MLP、LSTM都试了，为什么没采用？

- **面试官问法：** “换成更强分类器会不会解决？”
- **通俗答案：** 这批数据里三者在开发和测试上都几乎没有分流，实际全部选BM25。换复杂模型不是自动补救，先要确认标签、样本和任务目标是否合适。
- **技术答案要点：** linear是TF-IDF logistic，MLP为SVD64+hidden32，LSTM为词向量64+hidden64；训练上限20轮、patience3。多数类train64%、dev51.667%，MLP/LSTM恰好等于多数类基线，不能只拿train/dev差就下过拟合结论。
- **项目真实例子：** [实测] 三者dev60/test100全部选BM25，F1等于固定BM25 .380598/.459276。规则在线P50仅较固定Hybrid快5.04%，未达10%门槛，默认仍固定。V3也没有训练“事实/比较/总结”的新语义分类器，旧负结果不能回答那个新目标是否有效。
- **源码位置：** `ragscope/routing/learned.py` 的 `TfidfRouter`；`ragscope/routing/lstm.py` 的 `LSTMRouter`；`ROUTE/train_diagnostics/ACCEPTANCE.json`、`online/SUMMARY.json`。
- **还需补充知识：** 多数类坍缩、类不平衡、分类校准、工作流效用上界与分类任务定义。
- **现有证据是否足够：** 足够解释当前停用；不支持“LSTM或SLM永远不能有效分类问题”的泛化判断。

### Q33 Laya做了什么，候选顺序为什么重要？

- **面试官问法：** “Laya是不是你训练的路由大模型？效果怎样？”
- **通俗答案：** 用的是固定现成英语模型，让它在三条路线说明中选择；没有训练一个新的项目专属Laya。模型会受候选展示顺序影响，低置信时又常回退Hybrid，所以没有证明值得默认增加它。
- **技术答案要点：** 固定权重与SDK revision、CPU eager、max_len512、阈值.5、三种顺序；保存原始概率与回退后路线。区分原始choice改变、阈值后route改变、校准误差和最终QA效用。
- **项目真实例子：** [实测] 360题×3=1080决策、错误0、回退646；默认顺序dev回退95%、test96%；142/360题最终路线受顺序影响。不能事后挑测试F1更高的顺序改正式选型。Jev仅为付费API被排除的配置项，无真实本地实验。
- **源码位置：** `ragscope/routing/laya_adapter.py` 的 `LayaRouter`；`laya_campaign.py`、`laya_validation.py`；`LAYA/RAW_ACCEPTANCE.json`、`ROUTE/LAYA_PERMUTATION_CALIBRATION.json`。
- **还需补充知识：** choice模型位置偏差、概率校准、ECE/Brier与任务代理标签的限制。
- **现有证据是否足够：** 足够证明真实跑过且未显示稳定收益；不能把1080决策算成1080次新QA生成。

### Q34 怎样防止数据泄漏？索引里有测试论文算泄漏吗？

- **面试官问法：** “训练和测试是否读过同一篇论文？测试答案会不会进入索引？”
- **通俗答案：** 检索系统需要能搜索测试论文原文，这本身是开卷任务的正常条件；危险的是把测试问题、答案或看过结果后的调参信息混入训练和选择过程。
- **技术答案要点：** 按论文划分train/dev/test；训练器核验doc/query隔离；索引正文与QA参考答案分开；配置在holdout gold读取前冻结；区分语料可见、标签可见与开发反馈泄漏。
- **项目真实例子：** [实测] LoRA394 train篇与31 dev篇交集0；路由验证拒绝train/dev论文交集；V3FINAL保存配置冻结和split manifest。总结/比较参考由AI辅助构造并校验原文，人工尚未审核，题目构造偏差仍存在。
- **源码位置：** `ragscope/training/prepare.py`、`ragscope/training/evaluate.py` 的 `validate_comparison_inputs`；`ragscope/routing/core.py`；`V3FINAL/FINAL_CONFIGURATION.json`。
- **还需补充知识：** 检索评测的可见性协议、文档级划分、同源近重复与测试集反复使用风险。
- **现有证据是否足够：** 足够说明已实现的隔离与冻结；不能证明预训练语料零污染或所有潜在近重复都不存在。

### Q35 文字F1、语义相似度和事实正确率分别告诉我们什么？

- **面试官问法：** “不逐字一样也可能正确，为什么还用F1？加语义评分就解决了吗？”
- **通俗答案：** F1看答案词语与参考重合，能发现一些明显偏离，但会误伤正确改写。语义相似度更容忍换说法，却可能把数字错、否定错的答案也判得很像，因此还要核对事实与证据。
- **技术答案要点：** EM严格匹配，token F1衡量precision/recall重合，ROUGE-L看最长公共子序列，答案embedding cosine衡量表征接近。NLI/claim-support又是另一层自动代理，存在误判；引用可见性是格式属性。
- **项目真实例子：** [实测] 最终总结/比较语义cosine .8125→.8594，不能称“事实正确率85.94%”，也不是BERTScore。最新规则还保留数字、极性等flags与UNKNOWN，flag不自动等于事实错误，人审未完成。
- **源码位置：** `ragscope/evaluation/generation_metrics.py`、`ragscope/v3/practical_quality.py`；`reports/v3/METRIC_SPEC.md`；`V3FINAL/FINAL_SCORES_RESIDENT8K.jsonl`。
- **还需补充知识：** 多参考答案、语义评价鲁棒性、NLI域迁移、事实支持的人审协议与一致性。
- **现有证据是否足够：** 足够说明指标及结果的含义；不足以输出可信的整体自动事实正确率。

### Q36 怎样证明实验结论可复核，又不把测试数量当成果？

- **面试官问法：** “这些分数能重算吗？P95、置信区间和失败怎么处理？”
- **通俗答案：** 不能只留一张好看的汇总表。我保存每道题的输入、答案、失败、配置和模型身份，让分数能从原记录重算；实验规模说明测得多，不等于答得好。
- **技术答案要点：** 固定同题配对、保留失败分母和N/A；正式请求、warmup、精确响应复用与重放分开计数；P50/P95注明冷热与计时范围。bootstrap是对既有样本重采样估计不确定性，存在同论文聚类时要考虑聚类抽样，不能制造新独立样本。
- **项目真实例子：** [实测] 最终评分为基线220+最终220，部分最终响应是payload精确一致的复用，不冒充新调用；在线30+30用无答案/查询缓存路线。普通QA170可答题25提高/17下降/128持平；82定向回归通过只证明所覆盖行为，不是82道真实质量题，更不是所有功能零缺陷。
- **源码位置：** `ragscope/experiments/ledger.py`、`scripts/v3/finalize_delivery.py`、`scripts/v3/finish_evaluation.py`、`scripts/v3/finish_online.py`、`ragscope/v3/online_runtime.py`；最终原始评分、Junit与验收文件。
- **还需补充知识：** 配对检验、bootstrap/聚类bootstrap、置信区间与工程门槛、统计显著性和实际效益、完整可重复环境。
- **现有证据是否足够：** 足够复算已保存的描述性结果和检查配置；没有本轮所有增益均统计显著、长期P95稳定或独立人工质量审核完成的证据。简历可写真实工程覆盖与具体同题增益，不应写“全链路准确率95%”或“LoRA显著提升”。


## 18. 附录与个人速查

### 附录 A：一页项目技术速查表

| 面试官问 | 可直接给出的准确回答 | 证据入口 |
|---|---|---|
| 系统是什么 | 本地论文RAG工作台+可复核实验平台 | 第1/4/14章 |
| 最新默认入口 | run_ui_v3.ps1 → full_workbench UI → 本地11437 API | LOCAL_RUN_V3.md |
| 用什么模型 | 2B Q4写答案、.6B Q8生成1024维检索向量 | default.json与模型清单 |
| 论文规模 | 1585篇/19673块；原509目标缺失0 | FULL_DENSE_INDEX/TARGET_509 |
| 默认检索后端 | NumPy精确mmap向量+压缩BM25/RRF，非当前Chroma | full_retrieval.py |
| 三种模型是否混淆 | embedding编码，router选流程，generator写答案 | 第7/9/10章 |
| 当前有智能学习路由吗 | 没有；研究做过，默认是范围+任务规则 | 第10章 |
| 当前有LoRA吗 | 训练/恢复/同精度对照做过，默认不加载 | 第11章 |
| 普通QA改善多少 | 同170可答题F1 .3904→.4065，描述性小幅提升 | 第12章最终表 |
| 总结比较如何 | 20题F1 .4274→.4582，语义余弦.8125→.8594 | 同上，非人审真值 |
| 最突出短板 | 不可答仅3/30正确拒答；全库准确性仍弱 | 原始评分与检索 |
| 速度如何 | 热在线30题FINAL P50约1.187s/P95约1.988s | FINAL_ONLINE_RESIDENT8K |
| 95%准确率是真的吗 | 不是；旧.95是Recall@10，不是答案正确率 | V1/E0原始逐题 |
| 工程规模与测试 | HEAD405个.py/44275行；487静态测试函数；v3定向82通过 | repository_snapshot/XML |
| 能否说全面胜出 | 不能；保留退步项、失败与UNKNOWN | 第16章 |

**两分钟陈述示例。** 我把RAGScope从短段落检索实验扩展成长论文问答与评测平台，建立规范化原文坐标和可恢复索引，接通完整1585篇论文的全库、单篇和多篇问答。系统用BM25与1024维Dense融合，再按问题范围组织证据，通过本地量化2B生成结构化回答和可查看的引用。我用同题对照把检索覆盖、文字重合、语义接近、事实支持代理和真实在线成本拆开，发现重排、学习路由、LoRA并不自动改善质量，因此没有默认开启。最终总结比较在20题上改善，普通QA仅小幅上升，正确拒答和全库定位仍是短板。这些边界和原始失败都保存在账本里。

### 附录 B：完整受控仓库地图

以下列出本次HEAD的全部572个受控文件，不把环境/模型/缓存列进源码树。未跟踪的个人任务包、旧根架构说明、`rerun/`与`svg/`保留，但不计入这份受控文件地图；它们的初始状态保存在repository_snapshot。当前新增 `docs/architecture.md/html` 与 `_architecture/`属于本次文档交付，单列于树后。

```text
RAGScope/
├── .claude/
│   └── settings.local.json
├── .env.example
├── .github/
│   └── workflows/
│       └── offline-tests.yml
├── .gitignore
├── .streamlit/
│   └── config.toml
├── AGENTS.md
├── LOCAL_RUN_V2.md
├── LOCAL_RUN_V3.md
├── README.md
├── configs/
│   ├── config.yaml
│   ├── v2/
│   │   ├── analysis.json
│   │   ├── assets_cpu.json
│   │   ├── chart_contracts.json
│   │   ├── chunking.json
│   │   ├── data.json
│   │   ├── lora.json
│   │   ├── retrieval.json
│   │   ├── retrieval_auxiliary.json
│   │   ├── retrieval_statistics.json
│   │   ├── routing.json
│   │   ├── squad_e4.json
│   │   ├── tasks.json
│   │   └── ui.json
│   └── v3/
│       ├── campaign.yaml
│       ├── default.json
│       └── dependencies.lock.json
├── docs/
│   ├── execution/
│   │   ├── ACCEPTANCE_GAP_REVIEW.md
│   │   ├── CHUNKING_AUDIT.md
│   │   ├── CODE_AUDIT.md
│   │   ├── CONTEXT_STRESS.md
│   │   ├── CONTEXT_STRESS_REPORT.md
│   │   ├── DATA_REPORT.md
│   │   ├── DATA_RETRIEVAL_CHECKPOINT.md
│   │   ├── DECISIONS.md
│   │   ├── E0_RECONSTRUCTION.md
│   │   ├── E2_CORE_SHARD.md
│   │   ├── E3_SHARD.md
│   │   ├── FINAL_DELIVERY.md
│   │   ├── GENERATION_AUDIT.md
│   │   ├── GENERATION_HANDOVER.md
│   │   ├── GENERATION_RESUME_20261014.md
│   │   ├── INDEXING_AUDIT.md
│   │   ├── LAYA_LOAD_MEMORY_AUDIT.md
│   │   ├── LAYA_RAW_RUN.md
│   │   ├── LAYA_REAL_RUN.md
│   │   ├── LOCAL_REPORTS.md
│   │   ├── MASTER_PLAN.md
│   │   ├── MODELS_ENV_ADDITIONS.md
│   │   ├── ONLINE_EMBEDDING_WINDOW.md
│   │   ├── ONLINE_RAM_GUARD.md
│   │   ├── ONLINE_ROUTING.md
│   │   ├── PILOT_REPORT.md
│   │   ├── PRODUCTION_ACCEPTANCE.md
│   │   ├── R3_MATERIALIZATION.md
│   │   ├── RECOVERY.md
│   │   ├── RETRIEVAL_REPORT.md
│   │   ├── ROOT_CHECKPOINT.md
│   │   ├── ROUTING_REAL_RESULTS.md
│   │   ├── ROUTING_REPORT.md
│   │   ├── ROUTING_TRAIN_DIAGNOSTIC.md
│   │   ├── RUNNER_AUDIT.md
│   │   ├── STATUS.md
│   │   ├── SUPPORT_JUDGE.md
│   │   ├── TRAINING_REPORT.md
│   │   ├── TRAIN_ENV_AUDIT.md
│   │   ├── UI_AUDIT.md
│   │   ├── UI_CANDIDATE.md
│   │   └── v3/
│   │       ├── ACCEPTANCE_V3.json
│   │       ├── BEFORE_AFTER_CASES.md
│   │       ├── COMMANDS.md
│   │       ├── DECISIONS.md
│   │       ├── DIAGNOSIS_INPUT.md
│   │       ├── EXECUTIVE_SUMMARY_ZH.md
│   │       ├── EXPERIMENT_BUDGET.yaml
│   │       ├── FINAL_RESOURCE_STATE.json
│   │       ├── FINISH_STATUS.md
│   │       ├── GENERATION_PROTOCOL_DIAGNOSIS.md
│   │       ├── HYPOTHESES.jsonl
│   │       ├── LOCAL_RUN_BEFORE_FINISH.md
│   │       ├── LORA_DIAGNOSIS.md
│   │       ├── MASTER_PLAN.md
│   │       ├── QUALITY_SCORECARD.json
│   │       ├── QUALITY_SCORECARD.md
│   │       ├── RECOVERY.md
│   │       ├── RETRIEVAL_FUNNEL.md
│   │       ├── ROOT_CAUSE_VERDICTS.md
│   │       ├── ROUTING_ABLATION.md
│   │       ├── S1_RESOURCE_AND_QA_AUDIT.md
│   │       ├── SOURCES_V3.md
│   │       ├── STATUS.json
│   │       └── TASK_KEYPOINT_DIAGNOSIS.md
│   └── learning/
│       ├── 00_project_map.md
│       ├── 01_cleaning_chunking.md
│       ├── 02_evidence_and_metrics.md
│       ├── 03_retrieval_generation.md
│       ├── 04_model_size_and_memory.md
│       ├── 05_routing_experiments.md
│       ├── 06_reading_the_results.md
│       ├── 06_v3_diagnosis.md
│       └── CHANGELOG_FOR_LEARNING.md
├── notebooks/
│   └── .gitkeep
├── pytest.ini
├── ragscope/
│   ├── __init__.py
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── benchmark.py
│   │   ├── evidence.py
│   │   ├── generation_metrics.py
│   │   ├── retrieval_metrics.py
│   │   └── v2.py
│   ├── experiments/
│   │   ├── __init__.py
│   │   ├── analysis_report.py
│   │   ├── assets.py
│   │   ├── aux_preencode.py
│   │   ├── context_stress.py
│   │   ├── error_cases.py
│   │   ├── generation_campaign.py
│   │   ├── ledger.py
│   │   ├── legacy_reconstruction.py
│   │   ├── legacy_statistics.py
│   │   ├── production_acceptance.py
│   │   ├── rerank_campaign.py
│   │   ├── rerank_gpu.py
│   │   ├── rerank_sources.py
│   │   ├── rerank_validation.py
│   │   ├── retrieval_campaign.py
│   │   ├── run.py
│   │   ├── runtime.py
│   │   ├── simple_scope.py
│   │   ├── status.py
│   │   ├── support_judge.py
│   │   ├── task_catalog.py
│   │   └── validate.py
│   ├── generation/
│   │   ├── __init__.py
│   │   ├── cache.py
│   │   ├── context.py
│   │   ├── llm_client.py
│   │   ├── pipeline.py
│   │   ├── prompts.py
│   │   ├── resources.py
│   │   └── token_budget.py
│   ├── indexing/
│   │   ├── __init__.py
│   │   ├── array_index.py
│   │   ├── dense_index.py
│   │   ├── embedding.py
│   │   ├── identity.py
│   │   ├── local_embedding.py
│   │   ├── scoped_index.py
│   │   ├── sparse_index.py
│   │   └── stopwords_en.txt
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── chunkers.py
│   │   ├── cleaner.py
│   │   ├── loaders.py
│   │   ├── qasper.py
│   │   ├── squad_v2.py
│   │   └── technical.py
│   ├── observability/
│   │   └── __init__.py
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── bm25_retriever.py
│   │   ├── dense_retriever.py
│   │   ├── hybrid_retriever.py
│   │   ├── reranker.py
│   │   └── verified_reranker.py
│   ├── routing/
│   │   ├── __init__.py
│   │   ├── campaign.py
│   │   ├── core.py
│   │   ├── evaluate.py
│   │   ├── laya_adapter.py
│   │   ├── laya_campaign.py
│   │   ├── laya_report.py
│   │   ├── laya_supervision.py
│   │   ├── laya_validation.py
│   │   ├── learned.py
│   │   ├── lstm.py
│   │   ├── online_campaign.py
│   │   ├── online_retrieval.py
│   │   └── train.py
│   ├── schema.py
│   ├── training/
│   │   ├── __init__.py
│   │   ├── checkpoint.py
│   │   ├── data.py
│   │   ├── evaluate.py
│   │   ├── prepare.py
│   │   └── run.py
│   └── v3/
│       ├── __init__.py
│       ├── atomization.py
│       ├── audit.py
│       ├── auxiliary_public.py
│       ├── budget.py
│       ├── claim_metrics.py
│       ├── compressed_index.py
│       ├── core_packing.py
│       ├── deployment.py
│       ├── embedding_service_guard.py
│       ├── finish_runtime.py
│       ├── full_retrieval.py
│       ├── full_workbench.py
│       ├── generation_protocols.py
│       ├── gpu.py
│       ├── hierarchical_retrieval.py
│       ├── hierarchical_runtime.py
│       ├── judge_protocol.py
│       ├── local_runtime.py
│       ├── nli.py
│       ├── online_runtime.py
│       ├── oracle_padding.py
│       ├── practical_quality.py
│       ├── private_embeddings.py
│       ├── protocol_diagnostics.py
│       ├── quote_atoms.py
│       ├── records.py
│       ├── rerank_budget.py
│       ├── resident_claim_judge.py
│       ├── retrieval_diagnostics.py
│       ├── scoped_retrieval.py
│       ├── semantic_inputs.py
│       ├── splits.py
│       ├── task_references.py
│       ├── task_taxonomy.py
│       ├── task_workflows.py
│       ├── workbench.py
│       └── workflows.py
├── reports/
│   ├── .gitkeep
│   ├── retrieval_results.csv
│   ├── v2/
│   │   ├── ACCEPTANCE.json
│   │   ├── ANALYSIS.json
│   │   ├── DELIVERY_MANIFEST.json
│   │   ├── ERROR_CASES.manifest.json
│   │   ├── ERROR_CASES.md
│   │   ├── FINAL_REPORT.md
│   │   ├── GENERATION_TABLES.md
│   │   ├── RESOURCE_REPORT.md
│   │   ├── RETRIEVAL_TABLES.md
│   │   ├── ROUTING_REPORT.md
│   │   ├── TRAINING_REPORT.md
│   │   ├── figures/
│   │   │   ├── context_stress_resource.data.json
│   │   │   ├── context_stress_resource.png
│   │   │   ├── context_stress_resource.svg
│   │   │   ├── e0_reconstructed_retrieval.png
│   │   │   ├── e1_dev_chunk_counts.png
│   │   │   ├── e1_dev_evidence_matrix.png
│   │   │   ├── e1_dev_expansion_delta.png
│   │   │   ├── evidence_answer.data.json
│   │   │   ├── evidence_answer.png
│   │   │   ├── evidence_answer.svg
│   │   │   ├── lora_training_loss.png
│   │   │   ├── main_quality.data.json
│   │   │   ├── main_quality.png
│   │   │   ├── main_quality.svg
│   │   │   ├── quality_latency.data.json
│   │   │   ├── quality_latency.png
│   │   │   └── quality_latency.svg
│   │   ├── resource_interim/
│   │   │   ├── BUILD_RECEIPT.json
│   │   │   ├── RESOURCE_DATA.json
│   │   │   ├── RESOURCE_REPORT.md
│   │   │   └── build_e3_resource_report.py
│   │   └── retrieval_interim/
│   │       ├── CHART_CONTRACT.json
│   │       ├── DEV_RESULTS.md
│   │       ├── FIGURE_MANIFEST.json
│   │       ├── SVG_STRUCTURE_CHECK.json
│   │       ├── VISUAL_QA.json
│   │       ├── build_dev_figures.py
│   │       ├── dev_chunk_counts.data.json
│   │       ├── dev_chunk_counts.png
│   │       ├── dev_chunk_counts.svg
│   │       ├── dev_evidence_matrix.data.json
│   │       ├── dev_evidence_matrix.png
│   │       ├── dev_evidence_matrix.svg
│   │       ├── dev_expansion_delta.data.json
│   │       ├── dev_expansion_delta.png
│   │       ├── dev_expansion_delta.svg
│   │       └── summarize_expansion_pairs.py
│   └── v3/
│       ├── ERROR_ATTRIBUTION.md
│       ├── FINISH_ACCEPTANCE.json
│       ├── FINISH_REPORT_ZH.md
│       └── METRIC_SPEC.md
├── requirements-reports.in
├── requirements-reports.lock
├── requirements-v2.in
├── requirements-v2.lock
├── requirements.txt
├── scripts/
│   ├── .gitkeep
│   ├── accept_generation_v2.py
│   ├── analyze_error_cases_v2.py
│   ├── analyze_generation_v2.py
│   ├── apply_embedding_cpu_threads.py
│   ├── audit_chunk_preflight_scope_v2.py
│   ├── audit_embedding_indexes_v2.py
│   ├── audit_environment.py
│   ├── audit_final_report_v2.py
│   ├── audit_generation_numbers_v2.py
│   ├── audit_gpu_job.py
│   ├── audit_lora_v2.py
│   ├── audit_model_pilots.py
│   ├── audit_retrieval_v2.py
│   ├── audit_storage_v2.py
│   ├── benchmark_embedding_batches.py
│   ├── benchmark_embedding_cpu.py
│   ├── bootstrap_v2.ps1
│   ├── build_final_report_v2.py
│   ├── build_indexes.py
│   ├── build_local_reports_v2.py
│   ├── build_ui_model_pins.py
│   ├── configure_main_sources_v2.py
│   ├── configure_ui_v2.py
│   ├── evaluate_lora_v2.py
│   ├── export_retrieval_source_map_v2.py
│   ├── freeze_e2_v2.py
│   ├── freeze_e3_v2.py
│   ├── freeze_generation_v2.py
│   ├── generate_eval_report.py
│   ├── ingest.py
│   ├── inspect_model_sources.py
│   ├── manage_previous_jobs.py
│   ├── materialize_routing_contexts_v2.py
│   ├── materialize_simple_cache_only_v2.py
│   ├── materialize_squad_v2.py
│   ├── monitor_cpu_embedding.py
│   ├── package_results_v2.py
│   ├── pilot_generation.py
│   ├── plot_results_v2.py
│   ├── preencode_aux_v2.py
│   ├── prepare_cpu_assets.py
│   ├── prepare_generation_models.py
│   ├── prepare_generation_tokenizers.py
│   ├── prepare_lfm_model.py
│   ├── prepare_simple_scope_v2.py
│   ├── prepare_squad_v2.py
│   ├── prepare_text_only_models.py
│   ├── prepare_v2_data.py
│   ├── reenter_structured_after_host_restart.py
│   ├── resume_embedding_after_training.py
│   ├── resume_structured_after_low_ram.py
│   ├── run_all.py
│   ├── run_context_stress_v2.py
│   ├── run_generation_campaign.py
│   ├── run_laya_v2.py
│   ├── run_legacy_v2.py
│   ├── run_online_routing_v2.py
│   ├── run_rerank_gpu_v2.py
│   ├── run_rerank_v2.py
│   ├── run_retrieval_eval.py
│   ├── run_retrieval_v2.py
│   ├── run_routing_v2.py
│   ├── run_support_judge_v2.py
│   ├── run_ui_v2.ps1
│   ├── run_ui_v3.ps1
│   ├── run_v2.ps1
│   ├── run_v2.sh
│   ├── runtime_env.ps1
│   ├── serve_embedding.py
│   ├── serve_embedding_wsl.py
│   ├── serve_generation.py
│   ├── serve_online_embedding_v2.py
│   ├── summarize_retrieval_auxiliary_v2.py
│   ├── summarize_retrieval_statistics_v2.py
│   ├── test_generation.py
│   ├── test_retrieval.py
│   ├── v3/
│   │   ├── attribute_old_outputs.py
│   │   ├── audit_campaign.py
│   │   ├── audit_development_claim_gate.py
│   │   ├── audit_document_candidates.py
│   │   ├── audit_document_dense_candidates.py
│   │   ├── audit_final_resources.py
│   │   ├── audit_hierarchical_real_queries.py
│   │   ├── audit_identities.py
│   │   ├── audit_lora_runtime_diagnostics.py
│   │   ├── audit_lora_supervision_distribution.py
│   │   ├── audit_natural_reranker.py
│   │   ├── audit_online_cost.py
│   │   ├── audit_protocol_cases.py
│   │   ├── audit_protocol_results.py
│   │   ├── audit_protocol_supplements.py
│   │   ├── audit_retrieval_funnel.py
│   │   ├── audit_scoped_generation.py
│   │   ├── audit_sources.py
│   │   ├── audit_v2_preservation.py
│   │   ├── audit_workflow_utility.py
│   │   ├── bootstrap_campaign.py
│   │   ├── bridge_and_prepare_query_vectors.py
│   │   ├── build_acceptance.py
│   │   ├── build_chunk_dense_index.py
│   │   ├── build_chunk_dense_index_resource_v3.py
│   │   ├── build_chunk_sparse_indices.py
│   │   ├── build_delivery_reports.py
│   │   ├── build_development_task_references.py
│   │   ├── build_document_bm25.py
│   │   ├── build_document_dense.py
│   │   ├── build_task_references.py
│   │   ├── calibrate_bge_reranker.py
│   │   ├── calibrate_claim_completion.py
│   │   ├── calibrate_gpu_claim_judge.py
│   │   ├── calibrate_nli.py
│   │   ├── calibrate_qa_nli.py
│   │   ├── calibrate_qwen_reranker.py
│   │   ├── checkpoint_campaign.py
│   │   ├── close_claim_resource_block.py
│   │   ├── close_report_pending_notes.py
│   │   ├── complete_dense.py
│   │   ├── download_metric_models.py
│   │   ├── download_reranker.py
│   │   ├── export_human_review_package.py
│   │   ├── fetch_generation_cards.py
│   │   ├── finalize_delivery.py
│   │   ├── finalize_delivery_state.py
│   │   ├── finish_deployment_answers.py
│   │   ├── finish_evaluation.py
│   │   ├── finish_experiments.py
│   │   ├── finish_global.py
│   │   ├── finish_online.py
│   │   ├── finish_prompt_control.py
│   │   ├── freeze_candidate_gate.py
│   │   ├── freeze_scoped_contexts.py
│   │   ├── polar_prefix_diagnostic.py
│   │   ├── prepare_atomization_groups.py
│   │   ├── prepare_auxiliary_splits.py
│   │   ├── prepare_context_judgements.py
│   │   ├── prepare_development_task_contexts.py
│   │   ├── prepare_protocol_study.py
│   │   ├── prepare_retrieval_sources.py
│   │   ├── prepare_scoped_vectors.py
│   │   ├── prepare_semantic_inputs.py
│   │   ├── prepare_workflow_contexts.py
│   │   ├── prepare_workflow_labels.py
│   │   ├── recheck_runtime_identity.py
│   │   ├── repair_development_task_paragraphs.py
│   │   ├── repair_task_reference_whitespace.py
│   │   ├── report_development_source_tasks.py
│   │   ├── report_generation_diagnosis.py
│   │   ├── report_lora_diagnostics.py
│   │   ├── report_retrieval_funnel.py
│   │   ├── report_root_cause_verdicts.py
│   │   ├── report_workflow_diagnostics.py
│   │   ├── rescore_old.py
│   │   ├── resume_atomization.py
│   │   ├── run_atomization.py
│   │   ├── run_campaign.py
│   │   ├── run_context_judge.py
│   │   ├── run_development_task_generation.py
│   │   ├── run_lora_diagnostic_sequence.py
│   │   ├── run_lora_fixed_token_benchmark.py
│   │   ├── run_lora_runtime_diagnostics.py
│   │   ├── run_lora_supervision_smoke.py
│   │   ├── run_lora_supervision_smoke_v2.py
│   │   ├── run_natural_reranker.py
│   │   ├── run_online_cost.py
│   │   ├── run_protocol_study.py
│   │   ├── run_protocol_supplements.py
│   │   ├── run_protocol_supplements_v2.py
│   │   ├── run_scope_validation.py
│   │   ├── run_scoped_generation.py
│   │   ├── run_scoped_generation_resource.py
│   │   ├── run_workflow_generation.py
│   │   ├── save_environment_lock.py
│   │   ├── score_bertscore.py
│   │   ├── score_claim_metrics.py
│   │   ├── score_development_source_tasks.py
│   │   ├── score_development_source_tasks_resident_v2.py
│   │   ├── score_scoped_development.py
│   │   ├── screen_chunk_sparse_candidates.py
│   │   ├── screen_old_core_packing.py
│   │   ├── seal_splits.py
│   │   ├── serve_full_workbench.py
│   │   ├── serve_workbench.py
│   │   ├── smoke_full_api.py
│   │   ├── smoke_private_embeddings.py
│   │   ├── summarize_context_diagnosis.py
│   │   ├── verify_protocol_supplement_projection.py
│   │   └── verify_workbench_application.py
│   ├── validate_gpu_lease.py
│   ├── validate_laya_raw_v2.py
│   ├── validate_r3_v2.py
│   ├── verify_aux_preencode_v2.py
│   └── verify_ui_candidate_v2.py
├── tests/
│   ├── __init__.py
│   └── unit/
│       ├── test_analysis_report_v2.py
│       ├── test_array_index_v2.py
│       ├── test_aux_preencode_v2.py
│       ├── test_chunking_v2.py
│       ├── test_configure_ui_v2.py
│       ├── test_context_stress_v2.py
│       ├── test_data_v2.py
│       ├── test_e2_shard_v2.py
│       ├── test_error_cases_v2.py
│       ├── test_final_report_audit_v2.py
│       ├── test_freeze_generation_v2.py
│       ├── test_generation_campaign_v2.py
│       ├── test_generation_number_audit_v2.py
│       ├── test_generation_resources_v2.py
│       ├── test_generation_v2.py
│       ├── test_gpu_lease_v2.py
│       ├── test_indexing_v2.py
│       ├── test_laya_campaign_v2.py
│       ├── test_laya_report_v2.py
│       ├── test_laya_supervision_v2.py
│       ├── test_laya_validation_v2.py
│       ├── test_legacy_reconstruction_v2.py
│       ├── test_legacy_statistics_v2.py
│       ├── test_local_reports_v2.py
│       ├── test_metrics_v2.py
│       ├── test_online_embedding_service_v2.py
│       ├── test_online_remote_gpu_v2.py
│       ├── test_online_retrieval_v2.py
│       ├── test_online_routing_v2.py
│       ├── test_package_results_v2.py
│       ├── test_production_acceptance_v2.py
│       ├── test_rerank_campaign_v2.py
│       ├── test_rerank_gpu_v2.py
│       ├── test_rerank_sources_v2.py
│       ├── test_rerank_validation_v2.py
│       ├── test_retrieval_audit_v2.py
│       ├── test_retrieval_auxiliary_v2.py
│       ├── test_retrieval_campaign_v2.py
│       ├── test_retrieval_statistics_v2.py
│       ├── test_routing_campaign_v2.py
│       ├── test_routing_context_sources_v2.py
│       ├── test_routing_record_sources_v2.py
│       ├── test_routing_v2.py
│       ├── test_runner_v2.py
│       ├── test_scoped_index_v2.py
│       ├── test_simple_cache_only_v2.py
│       ├── test_squad_v2.py
│       ├── test_squad_window_v2.py
│       ├── test_storage_audit_v2.py
│       ├── test_structured_resume_v2.py
│       ├── test_support_judge_v2.py
│       ├── test_task_catalog_v2.py
│       ├── test_training_v2.py
│       ├── test_ui_live_generation_v2.py
│       ├── test_ui_model_identity_v2.py
│       ├── test_ui_offline_qa_v2.py
│       ├── test_ui_routing_v2.py
│       ├── test_ui_v2.py
│       ├── test_v3_atomization.py
│       ├── test_v3_budget.py
│       ├── test_v3_claim_judge_cache.py
│       ├── test_v3_claim_metrics.py
│       ├── test_v3_compressed_index.py
│       ├── test_v3_core_packing.py
│       ├── test_v3_development_source_paragraph.py
│       ├── test_v3_finish.py
│       ├── test_v3_generation_protocols.py
│       ├── test_v3_gpu_guard.py
│       ├── test_v3_hierarchical_retrieval.py
│       ├── test_v3_judge_protocol.py
│       ├── test_v3_lora_runtime_inputs.py
│       ├── test_v3_nli_contract.py
│       ├── test_v3_online_uncached_embedding.py
│       ├── test_v3_oracle_padding.py
│       ├── test_v3_protocol_diagnostics.py
│       ├── test_v3_protocol_transport.py
│       ├── test_v3_quote_atoms.py
│       ├── test_v3_records.py
│       ├── test_v3_rerank_budget.py
│       ├── test_v3_resident_claim_judge.py
│       ├── test_v3_retrieval_diagnostics.py
│       ├── test_v3_scoped_retrieval.py
│       ├── test_v3_semantic_inputs.py
│       ├── test_v3_source_audit.py
│       ├── test_v3_splits.py
│       ├── test_v3_task_references.py
│       ├── test_v3_task_taxonomy.py
│       ├── test_v3_workbench.py
│       └── test_v3_workflows.py
└── ui/
    ├── .gitkeep
    ├── chat_app.py
    ├── chat_app_v3.py
    ├── full_workbench.py
    ├── live_generation.py
    ├── model_identity.py
    └── routing_options.py
```

本次文档资产：`docs/architecture.md`为单一正文来源；`docs/architecture.html`由它生成；`docs/_architecture/`含轻量统计、证据核验、离线转换、图源、预览和验收收据。阅读时无需运行脚本或访问网络。

### 附录 C：重要Python文件与函数索引

| 文件/位置 | 关键符号 | 读它可以回答什么 |
|---|---|---|
| `ragscope/schema.py:38` | Document / EvidenceUnit / Query / Chunk / ContextBundle | 各层数据身份、坐标和字段 |
| `ragscope/ingestion/qasper.py:38` | make_document / convert_question / duplicate_groups / freeze_splits | 数据怎样规范化、隔离、映射与冻结 |
| `ragscope/ingestion/cleaner.py` | clean_text / clean_documents / evidence_units | 原文结构与legacy清洗区别 |
| `ragscope/ingestion/chunkers.py:17` | TokenizerAdapter | 真实tokenizer与离线码点计数的区别 |
| `ragscope/ingestion/chunkers.py:175` | FixedSizeChunker | token预算、空白边界与overlap |
| `ragscope/ingestion/chunkers.py:208` | SentenceAwareChunker | 句子合并与超长句fallback |
| `ragscope/ingestion/chunkers.py:273` | StructureRecursiveChunker / SemanticChunker | 结构切块为何不是相似度切块 |
| `ragscope/ingestion/chunkers.py` | ParentChildChunker / _assemble | 父子映射、chunk ID和证据投影 |
| `ragscope/indexing/embedding.py` | EmbeddingService | 固定BGE、显式设备、真实长度限制 |
| `ragscope/indexing/local_embedding.py:15` | HFJsonTokenizer / LocalEmbeddingService | 旧本地CPU编码协议与query指令 |
| `ragscope/indexing/array_index.py:14` | ArrayCache / ArrayIndex | 哈希缓存、npy及V2精确检索 |
| `ragscope/indexing/dense_index.py` | DenseIndex | Chroma旧/兼容路径与manifest校验 |
| `ragscope/indexing/identity.py` | artifact / signature / atomic_json / verify_artifacts | 字节身份与原子写入 |
| `ragscope/v3/compressed_index.py:12` | CompressedBM25 / CanonicalIndex | BM25稀疏矩阵与规范块检索 |
| `ragscope/v3/hierarchical_runtime.py` | load_hierarchy / DocumentSparse | 文档层与旧scoped资产加载 |
| `ragscope/v3/full_retrieval.py:11` | FullCorpusRetriever / search | 当前全库/范围/RRF/文档先验 |
| `ragscope/v3/finish_runtime.py` | ReusableEncoder / task_kind / evidence_bundle / answer_messages | query编码、基础任务证据与Prompt |
| `ragscope/v3/deployment.py:10` | deployment_kind / DeploymentClient / deployment_context | 当前普通与复杂任务真实分支 |
| `ragscope/v3/core_packing.py:6` | pack_core_first | 保核心、限邻居，避免挤占 |
| `ragscope/generation/context.py:97` | pack_context / validate_bundle / render_context | 最终实际材料、预算、坐标去重 |
| `ragscope/generation/prompts.py` | visible_output_schema | 引用枚举约束的能力与局限 |
| `ragscope/generation/llm_client.py:41` | LocalLLMClient / chat / _request | 本地HTTP、设备授权、token一致性 |
| `ragscope/generation/pipeline.py:165` | RAGPipeline.parse_output | JSON合法性，非事实裁判 |
| `ragscope/v3/full_workbench.py:50` | FullWorkbench.answer | 当前问答总编排与应用缓存/日志 |
| `scripts/v3/serve_full_workbench.py` | main / Handler | API入口、启动校验和正常退出 |
| `ui/full_workbench.py` | api / main | 用户范围、任务、引用与历史回放 |
| `ragscope/generation/resources.py` | RequestResourceSampler | 性能/资源采样的测量边界 |
| `ragscope/evaluation/generation_metrics.py` | exact_match / token_f1 / rouge_l | 规范化、多参考与LCS公式 |
| `ragscope/evaluation/evidence.py` | evidence_metrics / merge_intervals | 证据覆盖而非文档命中 |
| `ragscope/v3/practical_quality.py` | fact_checks / numbers / polarity | 字面疑点，真值仍UNKNOWN |
| `ragscope/v3/nli.py` / `claim_metrics.py` | NLI与主张评分原语 | 高级事实评价研究，非默认在线 |
| `ragscope/routing/core.py:118` | build_proxy_labels / RuleRouter | 三路代理标签与规则 |
| `ragscope/routing/learned.py:32` | TfidfRouter | linear/MLP的特征与持久化 |
| `ragscope/routing/lstm.py:24` | LSTM路由实现 | 小序列分类器，而非生成器 |
| `ragscope/routing/laya_adapter.py:39` | Laya适配层 | 真实choice、置信阈值和fallback |
| `ragscope/training/data.py:39` | prepare_examples | train/dev隔离、assistant-only监督 |
| `ragscope/training/run.py:63` | discover_attention_targets / train_loop | LoRA目标、优化与训练循环 |
| `ragscope/training/checkpoint.py:26` | checkpoint状态保存/恢复 | 原子提交、RNG和样本位置 |
| `ragscope/training/evaluate.py:89` | paired评估 | 同BF16只切adapter的对照 |
| `ragscope/experiments/ledger.py` | RecordStore / read_records / append_jsonl | 逐条账本、哈希和续写 |
| `ragscope/experiments/runtime.py` | configure / require_ram / validate_gpu_lease | 子进程路径、内存与当前boot租约 |
| `scripts/v3/complete_dense.py` | main | 补齐完整矩阵、复用与位图恢复 |
| `scripts/v3/finish_experiments.py` | screen / original_client / questions | 收尾开发选型与固定客户端 |
| `scripts/v3/finish_evaluation.py` | 最终评分入口 | 答案重合/语义/证据/规则汇总 |
| `scripts/v3/finish_online.py` | 在线专项入口 | 热请求与缓存控制 |

这是导航索引，不意味着每个列出的模块都进入默认链路。完整当前调用顺序以第14章为准，完整受控路径以附录B为准。

### 附录 D：模型与运行环境定位

完整模型/环境表见第13章。所有相对 `models/`、`envs/`、`tools/` 路径均相对于 `D:/RAGScope_Runtime`，不是项目根。核心定位如下：

| 资产 | 实际定位/身份 |
|---|---|
| 当前生成manifest | `D:/RAGScope_Runtime/models/generation/text_only/qwen3.5_2b-q4_K_M.json` |
| 当前embedding | `D:/RAGScope_Runtime/models/qwen3-embedding-0.6b/Qwen3-Embedding-0.6B-Q8_0.gguf` |
| 私有Ollama | `D:/RAGScope_Runtime/tools/ollama-0.34.4/ollama.exe` |
| 当前默认解释器 | `D:/RAGScope_Runtime/envs/core/Scripts/python.exe`，Python3.12.13 |
| 训练解释器 | `D:/RAGScope_Runtime/envs/train/Scripts/python.exe`，BF16/CUDA研究 |
| 源依赖身份 | `configs/v3/dependencies.lock.json`及各env的实际metadata/锁文件 |
| 模型/训练证据核验 | `docs/_architecture/training_routing_evidence.json` |

“模型文件总和”“目录逻辑大小”“硬链接去重逻辑大小”“磁盘物理分配”“进程RAM”“设备VRAM”应分别统计。此前运行目录去重逻辑约32.35GB是历史整个运行资产口径，不是当前向量矩阵，也不是程序运行必需一次装入内存。该历史存储数字不能当成本次2026-10-15重新扫描的最新占用。

### 附录 E：数据与实验结果索引

|任务|原始证据|核验方式|
|---|---|---|
|V1原始四检索|`rerun/per_query.csv`|按cfg分组200题均值；与summary.csv交叉|
|固定权重E0|`L0/{dense,hybrid}.jsonl`|real_model、200分母、metrics均值及模型身份|
|E1切块|`R2/records/E1_*.jsonl`|非rerank41条件、共4100；92适用|
|E2主模型矩阵|`G2E2/records/E2_qwen_*.jsonl`|按method/model分组，失败保留|
|Oracle/No Context|`G2E3/records/E3_qwen_*.jsonl`|每组200、182适用|
|E4—E7|`G2/records/E{4,5,6,7}_*.jsonl`|独立题库/模型/位置分别聚合|
|E8全库|`R2/records/E8_R512_*.jsonl`|正式GPU重排与其他CPU条件分开|
|路由真实路径|`G2/records/ROUTE_{TRAIN,DEV}_qwen_2b.jsonl`|200/60×3真实生成|
|路由重放|`D:/RAGScope_Runtime/runs/20261014_v2_routing/REPLAY_*.json`|八策略dev60/test100；不是新生成|
|路由实际在线|`D:/RAGScope_Runtime/runs/20261014_v2_routing/online/records.jsonl`|两个policy各30，actual_online|
|训练|`T2/steps.jsonl`、`steps.config.json`、`resume_from_step_*.json`|122连续步、配置、实际恢复|
|LoRA QA|`T2/paired_test100/{baseline,adapted}.jsonl`|同100/92适用、BF16/prompt-only|
|V3协议|`V3D/S2_PROTOCOL_RESULTS.jsonl`与`S2_G0_REUSED_RESULTS.jsonl`|720新主请求+144真实复用|
|V3工作流|`V3D/S4_WORKFLOW_GENERATION.jsonl`|180真实，不把clarify计新调用|
|LoRA机制|`V3D/S5_RUNTIME_*.jsonl`、`S5_FIXED64_*.jsonl`|24自然停止/3固定长度各分条件|
|收尾开发|`V3F/DEV_RETRIEVAL.jsonl`、`DEV_GENERATION.jsonl`、`QA_PROMPT_CONTROL.jsonl`|6recipes×220；80+40生成|
|最终答案质量|`V3F/FINAL_SCORES_RESIDENT8K.jsonl`及其来源响应|440评分、每组200/20、F1分母170/20|
|全库最终检索|`V3F/FINAL_GLOBAL_RETRIEVAL.jsonl`|200×3×2=1200|
|最终在线|`V3F/FINAL_ONLINE_RESIDENT8K.jsonl`|30×2=60；查询/答案缓存全false|

复算本稿使用的已完成轻量命令（只读运行数据，输出文档审计，不运行模型）：

```powershell
& 'D:/RAGScope_Runtime/envs/core/Scripts/python.exe' docs/_architecture/audit_experiment_evidence.py
```

当前源码/默认值以架构主文源码核验为准。历史记录中的源码hash是当时版本，不要求等于当前HEAD；同一方法名在不同设备、语料、scope、窗口、协议下也不是同一实验条件。

### 附录 F：V1—V3变化汇总

| 维度 | V1 | V2 | V3当前 |
|---|---|---|---|
| 主要语料 | SQuAD段落/arXiv摘要 | 509篇QASPER及独立技术/SQuAD | 同源全1585篇、原509全覆盖 |
| 证据对象 | 文档/chunk命中为主 | 独立EvidenceUnit与canonical区间 | 继续复用并记录完整实际上下文 |
| 索引 | BGE384+Chroma/BM25 | Qwen1024数组/范围索引 | mmap全块+压缩BM25+文档先验 |
| 切块 | 代码多种、实际短文难区分 | 长论文结构/父子/overlap受控实验 | R512默认，其他候选与历史保留 |
| 生成 | 有客户端/UI，早期无完整质量评测 | 四模型、Oracle/无材料/位置/切块对照 | 2B普通QA保守策略+任务证据组织 |
| 路由/训练 | 无已证实部署收益 | 三路代理、linear/MLP/LSTM/Laya；BF16 LoRA | 机制诊断、默认不用学习路由/adapter |
| 评价 | 检索Recall/MRR/nDCG | 答案F1/EM/ROUGE、覆盖与统计 | 另有BERTScore/NLI研究、最终余弦和规则 |
| 界面 | chat_app.py | chat_app_v2.py | full_workbench.py及独立API |
| 实测主要结论 | 短文本Hybrid召回较高 | 复杂模块有负结果，2B在所测协议更稳 | 总结比较改善较清晰，普通QA小增，拒答弱 |

### 附录 G：尚未解决的技术问题

1. 如何在不依赖题目给定论文的条件下，提高完整证据覆盖，并兼顾文档定位？
2. 如何校准“材料足够回答”与“需要拒答”，避免可答/不可答两端失衡？
3. 自动评价如何正确惩罚数字、否定和比较对象错误，同时认可正确复述？
4. 哪些LoRA退步来自输入分布、监督目标、停止条件、adapter开销或引用协议，各占多少？
5. 对“局部事实/跨段比较/全文总结”的语义分类，在足够监督数据下是否值得训练？旧三路标签不能回答这个问题。
6. 如何在固定预算内选择互补证据，而不只优化单块相关排序？
7. 多篇/中文/极长论文、并发与取消流程的真实用户验证还缺多少？
8. 新的独立题集和人工审核如何建立，避免在已经反复分析的题上继续选型？

这些是待研究问题，不是本次文档任务偷偷新增的开发目标。第16章记录了各项现有证据、影响和处理状态。

### 附录 H：面试前优先学习清单

**第一优先级：能手算。** 给两路排名算一次加权RRF；给参考/预测词集合算F1；给原文区间算交集覆盖；算1024维float32×19673行的存储；写出LoRA参数量r(m+n)与alpha/r缩放；说清8K窗口如何扣除prompt和输出预留。

**第二优先级：能沿代码讲。** 从UI payload追到FullWorkbench、deployment_context、search、pack_context、chat和parse；指出每个环节读取哪一份manifest、为什么不允许旧向量空间混入；解释为什么当前不是通过旧RAGPipeline.answer全包编排。

**第三优先级：能读一条失败。** 打开旧LoRA拒答题、一个全库漏证据题、一个重排后打包丢失题、一个数字规则疑点，分别检查原文、实际上下文、模型输出和参考。不要只背均值。

**第四优先级：能承认证据边界。** 把统计显著、工程容差、配置冻结、盲测、人工事实正确率和组件测试区分开。回答“为什么没有启用”时用具体实验，不用“高级技术不适合小数据”这种未经验证的总解释。

**第五优先级：扩展理论但不冒认实现。** 进一步学习ANN/HNSW与精确搜索取舍、cross-encoder训练目标、对比学习embedding、KL/校准与拒答、QLoRA的NF4/分页优化器、多证据组合优化。它们能帮助思考替代方案，但不应写成RAGScope已经实现。

### 附录 I：本手册的证据与生成方式

Markdown是HTML的单一正文来源，六幅SVG直接内嵌同一文本；HTML包含完整目录、响应式样式、系统中文字体与打印样式，无CDN、外部字体、在线图表或运行脚本依赖。

`repository_snapshot.json`保存源码计数/工作区初态和源码SHA；`experiment_audit.json`保存105个实验与配置来源、分母和逐条复算；`training_routing_evidence.json`保存训练/路由/环境的核验；`REFERENCE_STYLE_AUDIT.json`保存真实QuantMind范例结构；`CONSISTENCY_CHECK.json`保存MD→HTML章节、表格、代码、SVG、锚点和内容一致性；最终 `CONTENT_AUDIT.json`逐项对应用户要求，并声明未知事项。

本次只做资料核验和文档编写。没有把未进行的人评、未运行的模型、未完成的合并或被拒绝的应用浏览器验收写成完成；历史源码、模型、数据、报告和断点保留。
