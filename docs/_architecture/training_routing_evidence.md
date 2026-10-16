# LoRA、路由研究、模型资产与 Python 环境核验稿

核验日期：2026-10-15。本稿供架构手册整合；全部数值来自当前文件读取和轻量聚合，没有启动训练、模型服务、GPU 任务或重新下载资产。机器可读核验清单为同目录 `training_routing_evidence.json`。该清单还保存本次读取的关键原始文件大小和 SHA-256。

本稿中的运行目录简称：

|简称|实际目录|
|---|---|
|`V2`|`D:/RAGScope_Runtime/runs/20261014_v2`|
|`LORA`|`D:/RAGScope_Runtime/runs/20261014_v2_lora`|
|`TRAIN`|`LORA/train_2b_1024_seed42`|
|`ROUTE`|`D:/RAGScope_Runtime/runs/20261014_v2_routing`|
|`LAYA`|`D:/RAGScope_Runtime/runs/20261014_v2_laya`|
|`V3D`|`D:/RAGScope_Runtime/runs/v3_20261015_014506`|
|`V3F`|`D:/RAGScope_Runtime/runs/v3_finish_20261015_1415`|

## 可整合到第 10 章：Routing 与任务工作流

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
|Jev|没有本地可验证实现或本轮请求|[代码/配置] `configs/v2/routing.json` 为 `NOT_APPLICABLE`，理由是可选付费外部API被排除；不能把它算作完成的实验|

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

[代码/配置] 最新V3F默认并未接入V2这些学习分类器，而是范围控制加可审计的任务规则：普通事实/解释保留原有可靠短证据协议；比较、总结、多篇任务使用更广的证据组织。`V3F/FINAL_CONFIGURATION.json` 冻结 `SAFE_QA_TASKS` 和选型依据；`configs/v3/default.json` 指向当前已验收发布。用户可以手动指定任务。真实语义问题在于“这道题需要哪种证据范围”，而不是任何复杂问题都去重排。当前规则不是经过训练并证明泛化更好的语义分类器。

## 可整合到第 11 章：已有 LoRA 实验

### 通俗原理与本项目的实际选择

LoRA可以理解成给原模型的一些线性变换加一组很小的可训练“修正层”。原权重W冻结，只训练低秩矩阵A、B，使输出近似使用 `W + (alpha/r)BA`。这样需要优化和保存的参数少很多，但底座仍须常驻并执行前向/反向，并不等于训练几乎不占显存。

[代码/实测] 本项目实际是**BF16底座LoRA，非QLoRA**。底座是 `Qwen/Qwen3.5-2B` 固定revision `15852e8c16360a2fea060d615a32b45270f8a8fc`，本地完整safetensors约4.55GB；它与部署的GGUF Q4模型属于不同表示与后端。配置有0.8B后备路径，不等于实际下载或训练了0.8B；最终选中 `selected_path_index:0`、真实122步来自2B。模型只在取得GPU lease、通过资源/资产校验后加载。

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

[未知/推断] BF16合并舍入、残差计算路径可能产生logits差，但未隔离原因，不应称PEFT bug。可进一步验证的方向是同分布的干净证据与检索噪声分层、长度匹配、拒答阈值/标注充分性审核、同协议语义与事实指标、不同checkpoint的冻结dev对照。这些是后续研究问题，不是本轮已完成的新训练。

[代码/实测] 当前没有把旧adapter并入Q4默认模型；V3正式新LoRA未触发，只做了监督链路smoke与机制诊断。原因是旧结果没有稳定质量收益，且预设“材料充分时仍有可训练缺陷/足够部署分布样本/可信主事实评价”等前置条件未同时满足。`configs/v3/default.json`、V3F选型记录与实际workbench均保留基础2B路线。历史 `LORA_DIAGNOSIS.md` 末尾“正式holdout受阻、默认v2”属于较早V3D阶段，V3F后续已实际完成新holdout与发布；训练资产未变化，不能把那一段当作当前全项目状态。

## 可整合到第 13 章及附录 D：模型与运行环境

### 模型资产不是同时加载的模型

[代码/资产] 下表记录现有本地资产及角色。GGUF与safetensors可能是同家族的不同表示；历史下载的视觉projector仍保留，但文本专用manifest不加载它。不能把“有模型文件”升级为“在默认问答中启用”。

|模型|资产与体积|角色/实际状态|
|---|---|---|
|Qwen3.5-2B Q4_K_M|`models/ollama/blobs/sha256-20cb277...`，语言权重1,274,396,992字节；文本manifest `models/generation/text_only/qwen3.5_2b-q4_K_M.json`|当前默认生成；模型tag `ragscope-qwen3.5:2b-q4_K_M`，私有Ollama0.34.4；非LoRA合并模型|
|Qwen3.5-4B Q4_K_M|语言权重2,648,593,216字节；同目录4b manifest|V2模型尺度实验、V3开发对照；当前非默认|
|Qwen3.5-9B Q4_K_M|语言权重5,629,109,120字节；同目录9b manifest|V2模型尺度实验、V3开发对照；单卡串行，当前非默认|
|LiquidAI LFM2.5-1.2B-Instruct Q4_K_M|语言权重730,895,168字节；`models/generation/lfm2.5_q4_k_m.json`|V2异构小模型对照；不是当前默认；许可清单写LFM Open License v1.0/local noncommercial research，不是所有模型统一Apache|
|Qwen3-Embedding-0.6B Q8_0|`models/qwen3-embedding-0.6b/Qwen3-Embedding-0.6B-Q8_0.gguf`，639,150,592字节|当前1024维/L2/last pooling检索编码；V2有CPU llama.cpp b11429实现，V3 GPU编码桥；具体当前调用由V3运行层决定|
|BGE-small-en-v1.5|`models/bge-small-en-v1.5/5c38ec7c405ec4b44b94cc5a9bb96e735b38267a`；config为BertModel/hidden384/max_position512|历史Embedding/CPU比较；当前主索引不是该384维向量|
|BGE-reranker-base|`models/bge-reranker-base/2cfc18c9415c912f9d8155881c133215df768a70`；权重1,112,206,140字节，download_manifest完整|V2实际重排/CPU成本对照；当前默认关闭重排|
|Laya|`models/laya/7b928d828b7b0e022f929d9bd2e44165aa270148`；权重payload约842.6MB，421,293,830个元素（含buffer）；ModernBERT encoder hidden1024/layers28|实际CPU路线选择实验；加载时FP32，不能把F16文件体积当驻留内存；当前未启用|
|Qwen3.5-2B BF16|`models/Qwen3.5-2B/15852e8c16360a2fea060d615a32b45270f8a8fc`；官方单分片4,548,221,488字节，12文件总4,571,272,453字节|真实LoRA训练/同精度比较；与Q4部署文件独立，不能共用权重哈希|
|旧LoRA adapter|`TRAIN/checkpoints/step-000122/adapter/adapter_model.safetensors`，14,798,456字节，SHA `a532dc22942d84e4dceabacb64b936c191b37ebdfa9fda092f3f87464d0bbae4`|已训练/恢复/同精度运行，但不默认使用；未成功发布合并副本|
|RoBERTa-large、DeBERTa-v3-base MNLI/FEVER/ANLI|`models/v3_metrics/FacebookAI__roberta-large/722cf37b1afa9454edce342e7895e588b6ff1d59`；`models/v3_metrics/MoritzLaurer__DeBERTa-v3-base-mnli-fever-anli/6f5cf0a2b59cabb106aca4c287eed12e357e90eb`|高级评价资产。仅文件存在不代表NLI/ BERTScore所有正式行已完成；最新实用语义分数也不能冒称BERTScore|
|Qwen3-Reranker-0.6B|`models/v3_reranker/Qwen__Qwen3-Reranker-0.6B/e61197ed45024b0ed8a2d74b80b4d909f1255473`，权重1,191,588,280字节，config为BF16 Qwen3ForCausalLM|V3候选重排资产/研究路线；不在最终默认问答路径|

### Python 为什么分环境

[代码/资产] 三个主要uv环境core/models/train的 `pyvenv.cfg` 都为CPython3.12.13、uv0.11.19、`include-system-site-packages=false`，base位于 `D:/RAGScope_Runtime/tools/python/cpython-3.12-windows-x86_64-none`。不是WSL Python，也不是系统全局Python。环境分开实际解决的是依赖版本冲突及轻量UI与模型库的隔离：CPU BGE/Laya保留Transformers4.57.6，Qwen3.5训练需要5.19.0/CUDA版torch。

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

[文档冲突] `docs/execution/TRAIN_ENV_AUDIT.md` 末尾的“当前46包”描述准备阶段；现有train已补入rank-bm25，直接分发包47，旧 `requirements.lock.txt` 的SHA为 `e2cd66b2...`，而新 `requirements.r3_gpu.lock.txt`才对应后续GPU重排需要。旧锁没有删除，但不能拿旧46包记录宣称当前完整环境一致。两个metrics环境也不应称为再下载两套完整Torch，.pth明确复用已有依赖；其独立性是新增包与入口隔离，而非全部底层包物理隔离。

### 运行资产为何集中D盘

[代码] `scripts/runtime_env.ps1:1` 设置当前PowerShell进程及子进程环境：HF/PyTorch/SentenceTransformers/uv/pip/tokenizer缓存、TEMP/TMP、CUDA/torch扩展/inductor/Matplotlib目录都指向 `D:/RAGScope_Runtime`。不修改系统全局环境。默认置 `CUDA_VISIBLE_DEVICES=''`、OMP/MKL/OpenBLAS/NumExpr四线程；显式GPU分支先通过lease验证再开放设备。`PYTHONDONTWRITEBYTECODE=1`避免额外pycache。

[代码/配置] 最新工作台用core Python通过HTTP控制本地私有模型后端，core没有torch并不妨碍GPU推理，因为真正计算在独立模型服务进程中。当前后端窗口统一8192，不代表每道题都输入8K：普通QA参考证据最多2048、输出512；比较/总结参考证据最多6144、输出768。窗口、参考证据预算、实际token输入是不同数字。

[实测/边界] 文件体积、目录逻辑大小、硬链接去重后的大小、磁盘物理占用与运行时RAM/VRAM不是同一指标。模型身份manifest+blob布局会复用相同文件；原视觉projector仍可落盘但文本tag不加载。只读现有清单能说明资产位置，不能把整D盘用量变化全部归因于RAGScope，也不能由“模型文件1.27GB”推出“推理只需1.27GB显存”。

## 明确的未知项与面试边界

- [未知] 当前LoRA退步主要由训练分布、拒答校准、参考答案表达、实现数值还是样本噪声造成，尚无单因素因果实验证明。
- [未知] BF16合并logits差异来源未隔离；不能说成功合并/量化发布，也不能归咎外部库bug。
- [未知] 新24题及旧100题均没有独立人工事实审核；语义/文字分数不能代替事实正确率。
- [实测不足] V2只有200条路由训练题，学习的是质量成本代理；这不足以证明LSTM/Transformer/SLM做“局部/比较/总结”分类普遍无效。V2没训练那个分类目标。
- [未执行] Jev外部API、新的正式V3 LoRA训练、合并模型24题及微基准没有完成；不能列进已完成成果。
- [已完成但不默认] V2 LoRA训练、BF16配对、linear/MLP/LSTM训练、Laya真实1080决策、路线重放和固定/规则真实在线对照都存在原始记录，不能概括为“这些只写了框架”。
