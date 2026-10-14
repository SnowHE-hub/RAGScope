# S4.1–S4.4 路由结果与在线验收

日期：2026-10-14（客户端 Australia/Sydney）。正式200train/60dev训练、100test重放及Laya校准已完成，来源见[ROUTING_REAL_RESULTS.md](ROUTING_REAL_RESULTS.md)。随后真实在线60次与6次独立预热也已完整验收。规则路由P50降低5.0446%，未达到预设10%门槛，默认保留固定Hybrid。

## 当前状态

| 分支 | 已实现且已实际验证 | 仍待真实实验 |
|---|---|---|
| Always-BM25 / Always-Hybrid / Always-Rerank | 各60dev/100test真实重放完成；逐条绑定冻结决策与真实生成来源 | 无新增实验；dev选择Hybrid进入实际在线比较 |
| 规则路由 | 仅用可见输入；真实dev F1 .426660、test F1 .477269；在线30题已完成 | P50降低不足10%，不设为默认 |
| TF-IDF + LogisticRegression | 200train/60dev真实代理标签训练完成，17次求解迭代；dev/test均预测BM25，完整重放 | 不因模型训练成功声称有部署收益 |
| TF-IDF + SVD + MLP | 真实200train/60dev训练，14epoch早停、最佳11；dev/test均预测BM25，完整重放 | 同上，保留预测塌缩事实 |
| 小LSTM | 真实200train/60dev、69,251参数、CPU4/seed42，9epoch早停、最佳6；dev/test均预测BM25，完整重放 | 不宣称隔离了网络架构效果 |
| Laya | 原官方SDK/固定权重已实际运行CPU4 FP32：360题×3顺序=1080条，0调用错误、646低信心回退，3条dev预热单列；完整逐条独立验收PASS，模型已退出 | 原order0质量及三顺序校准均完成；未被dev选择进入在线比较，不宣称其有在线收益 |
| 在线部署门槛 | 同30个dev题、两策略各30次真实请求，6次预热独立；0失败、66次输入token一致 | 完整dev质量容差通过；P50降低5.0446%不足10%，默认固定Hybrid |

## 固定协议与接口

配置：`configs/v2/routing.json`。路径ID仅 `bm25 / hybrid / hybrid_rerank`。路由选择已建好的路径，绝不对问题重新切分或重新编码整库。生成模型、上下文打包/父块扩展、数据scope必须与正式比较配置一致。

`RoutingRequest.from_query` 只提取 question、显式用户scope和白名单元数据 document_length_tokens/document_type/language。学习特征不使用scope中的文档ID作为捷径，也不使用答案、gold_doc_id、证据坐标、test分数。文档类型与语言限制为明确类别；问句长度不作为规则判定简单任务的条件。

正式200train、60dev、100test来自主run的`ROUTING_QUERY_MANIFEST.json`与正式生成manifest，使用同一sort-ID后seed42抽样协议，不重新调用早期通用`freeze_query_ids`助手另抽样。`build_proxy_labels` 对每个冻结问题强制要求三条路径完整记录，包括失败；要求 `real_model=true`、统一 `comparison_config_hash`（排除路径这个实验变量）。缺一条就报错，不能生成随机标签替代。

可回答题的离线标签按答案文字重合F1确定；不可回答题的F1保留N/A，单独采用正确拒答0/1作为代理质量。质量差≤0.01的路径先选实测总成本更低者，成本仍相同再选证据覆盖更高者，最终按固定路径顺序。所有路径都失败时仍保留该问题且标记all_paths_failed，其适用代理质量均为0。标签是 **基于质量代理与成本的路径选择**，并非真实任务绝对最优策略。不可回答的证据指标为N/A，不能当作覆盖0或1。

训练检查train/dev查询和论文不重叠，test禁止进入fit；模型保存配置、样本IDs/哈希、标签分布、单类塌缩、警告与训练历史。若train只有一类，保留诚实的单类常数回退 `PASS_FALLBACK_SINGLE_CLASS`，不伪造三类平衡。LSTM只是学习性对照，不能将其与生成LLM混成架构排名。

训练入口（必须先有真实标签文件；此处是执行接口说明，不声称命令已经针对正式数据运行）：

```powershell
. ./scripts/runtime_env.ps1
& 'D:/RAGScope_Runtime/envs/core/Scripts/python.exe' -m ragscope.routing.train --labels '<冻结真实标签JSONL>' --output '<新的D盘run目录>' --kind linear
# --kind mlp 使用同一环境；--kind lstm 需已安装并验证的隔离torch环境。
```

`replay` 复用现有真实生成记录评估最终答案F1/证据/失败/错误拒答，延迟标为component_replay_estimate，加入实际测量的路由器耗时。`online` 逐题真正调用执行器，保存actual_online端到端时间和模型驻留说明，每行flush+fsync；需要装卸模型时调用者必须把装卸放入计时执行器。部署门槛不能接受重放延迟，也不能用test调优。

## Laya 官方来源核验

读取官方 [作者仓库](https://github.com/NandhaKishorM/laya/tree/c73908d96ff73c0062c340a11bf7c54ec7d8bc4d) 与 [模型仓库](https://huggingface.co/convaiinnovations/laya/tree/7b928d828b7b0e022f929d9bd2e44165aa270148)，均标注Apache-2.0。SDK固定Git commit `c73908d96ff73c0062c340a11bf7c54ec7d8bc4d`，版本0.4.6；模型固定revision `7b928d828b7b0e022f929d9bd2e44165aa270148`。

模型API公布的English根目录权重为842,609,210字节，SHA-256 `891102d372688fc2a094dac56a384bc537b87c63f21f9f3dac0be2b7cbc8d86c`。本机已下载并实际验证文件大小和SHA均匹配。完整6文件清单、URL、大小与哈希位于 `D:/RAGScope_Runtime/models/laya/7b928d828b7b0e022f929d9bd2e44165aa270148/model_manifest.json`，状态VERIFIED；下载日志 `D:/RAGScope_Runtime/logs/laya_pinned_download.log`。仅使用stdlib公开HTTPS下载，未导入SDK/Torch、未扫描凭证、未加载模型；下载完成不能记为真实路由实验完成。

已保存并检查加载路径的官方代码：LICENSE、pyproject.toml、laya/__init__.py、agent.py、router.py、common.py。审阅重点是设备选择、下载范围、tokenizer/model构建、safetensors加载及predict输出预算；不是整库所有可选插件的安全认证。原文与哈希清单位于 `D:/RAGScope_Runtime/downloads/laya_audit/c73908d96ff73c0062c340a11bf7c54ec7d8bc4d/manifest.json`，模型元数据快照为同目录 `hf_model_info.json`。

当前SDK的Router默认preload=False、max_loaded=2；preload=True会加载多个检查点。项目不使用该Router，选择 `laya.load(local_checkpoint, device='cpu', backend='eager', fast=False, compile=False)`，只准备根目录的rl_agent_config.json、model.safetensors、tokenizer/*、encoder/*。本地适配器先要求全部必要文件及哈希，临时启用HF_HUB_OFFLINE，防止缺失目录触发远程fallback；不传token、不读取凭证值。

该固定加载路径用本地encoder config构造Transformers内建模型，未开启trust_remote_code。SDK `_fix_tokenizer_config` 可能原子更新本项目tokenizer配置；适配器保存加载前后哈希，不能把变更后的文件仍冒称原始hash。正式安装必须由主执行器固定SDK来源/依赖后再做真实加载预检。

Laya返回合法choice不代表任务路由正确。项目使用三种循环候选顺序保存全部概率与选择，检查选择是否变化；ECE/Brier只相对生成F1/成本代理标签，不能引用上游“calibrated”描述就声称本任务已校准。低信心、非法概率、输入/候选描述被截断均回退固定Hybrid。Jev按任务书保持NOT_APPLICABLE，未调用收费API。

## 固定SDK实装核验（后续）

已按固定Git提交下载必要源码，用`--no-deps --no-build-isolation`在独立models CPU环境安装Laya0.4.6。44个安装源码文件与下载manifest逐个SHA匹配；`verify_sdk_installation()`通过importlib.metadata读取路径并复验文件，不import Laya/Torch。实际验证二者均未出现在sys.modules。详情见`docs/execution/MODELS_ENV_ADDITIONS.md`和models环境`laya_sdk_installation.json`。

`load_local`现在在SDK导入前强制检查实装源码凭据。加载期间同时启用HF_HUB_OFFLINE、TRANSFORMERS_OFFLINE和HF_HUB_DISABLE_IMPLICIT_TOKEN，之后恢复原值；始终传本地完整检查点，避免官方远程checkpoint分支读取HF_TOKEN。实际offline loader保护测试1 passed，23 deselected，1.96秒，日志`D:/RAGScope_Runtime/logs/laya_fixed_sdk_guard_tests.log`。此安装核验阶段未调用SDK或加载权重；后续真实1080条执行见本目录LAYA_REAL_RUN.md。安装测试本身不能记真实路由质量。

## 离线验证与缺口

当前测试文件：`tests/unit/test_routing_v2.py`。24项离线测试已通过，包括泄漏字段不进特征、冻结ID、F1容差与成本优先、失败标签留存、真实记录门槛、train/dev论文不重叠、线性/MLP/LSTM小夹具实际拟合、二分类概率、JSON+npz序列化、单类塌缩、LSTM预算、三顺序Laya测试对象、加载路径哈希/offline控制、概率校准计算、重放与在线计时区分、部署拒绝。模型环境中的小LSTM只用64词上限、4维embedding/hidden、最多3epoch，全部CPU；这验证代码执行，不证明真实路由质量。

最近覆盖率报告：`D:/RAGScope_Runtime/runs/routing_models_coverage.json`，隔离环境Python `D:/RAGScope_Runtime/envs/models/Scripts/python.exe`。24 passed in61.23s；整体92%，core94%、evaluate91%、Laya适配器96%、learned96%、LSTM97%、CLI0%。没有隐藏正式CLI未执行分支。先前core环境22项测试也通过，旧报告仍保留。此次测试期间Windows平台信息WMI查询打印0x8007000e内存不足诊断，但进程继续，最终24项全部通过；该诊断不被当作模型实验结果。上述是早期功能测试阶段的记录；后续Laya已实际完成1080条，正式代理标签训练与质量重放现已完成，见真实结果报告；历史夹具成绩不能替代它。

## 正式结果接线与防泄漏补充

ROOT新增`ragscope/routing/campaign.py`与`scripts/run_routing_v2.py`，早期querymanifest位于主run/ROUTING_QUERY_MANIFEST.json，实际冻结200train/60dev/100test，来源SPLIT_MANIFEST哈希核验。只有真实正式生成完整260×3时才输出labels；不可答F1仍N/A，使用明确的correct_abstention代理单列。全部7个必需策略测试决策写成不可变带哈希DECISIONS文件后才允许读取test路径分数做replay。回放只报告实际组件计时之和。正式生成manifest、ROUTE780、三个学习模型训练、七策略冻结及八策略质量重放、Laya三顺序校准均已完成。在线60次使用独立真实请求完成；离线数字见[ROUTING_REAL_RESULTS.md](ROUTING_REAL_RESULTS.md)，Laya原始1080记录见[LAYA_REAL_RUN.md](LAYA_REAL_RUN.md)。

正式固定权重 train600 诊断已补齐，见 [ROUTING_TRAIN_DIAGNOSTIC.md](ROUTING_TRAIN_DIAGNOSTIC.md)。原权重和dev/test决策未变；分类gap与类别占比变化一并解释。

## 训练诊断的准确解释

这里的accuracy是匹配质量与成本代理标签的比例，不是生成答案F1。线性模型train accuracy为0.675、dev为0.516667；MLP和LSTM分别为0.64、0.516667。三者在dev/test全部选择BM25；训练集上只有线性模型选择了9次Hybrid，其余191次为BM25，MLP/LSTM的200次全为BM25。

train的BM25标签占128/200=64%，dev占31/60=51.667%。因此MLP/LSTM的0.123333准确率差恰等于多数类占比变化，不能直接称为过拟合。线性模型训练集比多数类基线高3.5个百分点，dev仍未超出多数类基线；这是有限训练收益没有在这批dev中体现的观察，不是架构优劣或显著性结论。原dev损失曲线、早停和600条固定权重预测共同保留，不凭test重训。

## 离线结果与部署决策

三个固定策略、rules、linear、MLP、LSTM和Laya各有dev60/test100，共1280次完整重放选择；它们复用真实路径生成结果，不能计为1280次新生成，也不能称在线请求。dev答案F1分别为BM25 0.380598、Hybrid 0.429777、Rerank 0.370988、rules 0.426660；三个学习模型均为0.380598。rules相对Hybrid差−0.003117，处在预设0.01容差内；真实在线结果独立列于下表，不以重放P50代替。

Laya原order0在dev全部选择或回退Hybrid，F1同为0.429777，组件重放P50却由约1.218秒增加至1.900秒，57/60次为低信心回退。原1080条三顺序决策完整保留；dev三顺序raw choice变化59/60，最终路线变化26/60。order0的dev ECE约0.1254、Brier约0.7567，只相对本次代理标签解释；不能由SDK声明或合法概率输出推断本任务已可靠校准。默认order0未根据test更换。

**本轮学习路由未显示优于固定Hybrid的dev答案质量；rules满足完整dev质量容差，但实际在线P50只降低5.0446%，未达到预设10%，因此默认保留固定Hybrid。** S4.4实验本身已完整执行，负向部署结论也是有效结果，没有为了达标更换题目、阈值或重复测量。

|在线策略|正式题数|可答F1（26题）|P50秒|P95秒|调用失败|
|---|---:|---:|---:|---:|---:|
|固定Hybrid|30|0.408830|1.277657|2.026595|0|
|rules|30|0.411395|1.213204|2.201954|0|

同30个dev题包含4个不可回答题，两策略正确拒答率均0；不能因为调用成功就称答案正确。最终质量门槛取完整dev60（其中56可答）的冻结F1，而不是在线子集F1；延迟取上述真实在线30题/策略。顺序按题交替，6次预热独立。2B驻留，生成应用缓存关闭；实时CPU4 embedding、CPU4 BGE、索引/上下文打包都按原配置。BGE加载28.903秒、索引加载2.883秒另列，不能把驻留延迟称为冷启动时间。观测客户端峰值RSS2,473,873,408字节，2秒采样最低host available1,764,184,064字节；采样可能错过峰值，保护线未降低。

第一轮因CPU客户端未传递已核验远程GPU授权，3次预热在HTTP生成前失败、0正式行，预算6191–6193原样保留；旧实现、manifest、失败行和退出2证据按原SHA归档。最小修复增加默认关闭的授权回调，每请求重新验证自有服务与当前boot lease，CPU CUDA mask、模型identity和payload不变；66项定向测试通过。重新冻结只改变implementation和online_hash，题目/策略/顺序/模型/生成配置全部相同。唯一修复后运行完整60+6、0失败、66次token计数一致，实际exit0；此前失败不能藏入“从未出错”的陈述。

线上记录位于`D:/RAGScope_Runtime/runs/20261014_v2_routing/online`；当前online_hash为`0c82613047a1b8e8041d2a3129faa0627aaea29e571b84e0a7f30ca0065cc5f1`。独立签收`INDEPENDENT_COMPLETION_AUDIT.json` SHA `57f895d3329ebb6670b536c0841b9ecb6ebe9675cbd3240ddc660814c4a1b708`，核完整ID、交替顺序、逐条/config哈希、66预算票、token一致、指标和门槛。CPU模型/客户端精确退出、两锁释放收据SHA `d8f7981a69a703ad495ccaa99a49cc6d371903116c72d9391668d7b632afe308`；根任务的两个服务继续用于独立UI验收。本轮线上没有读取test进行选择，CPU与GPU重放来源的成本差异仍按[真实结果报告](ROUTING_REAL_RESULTS.md)披露。
