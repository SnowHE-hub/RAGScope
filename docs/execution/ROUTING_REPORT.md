# S4.1–S4.4 路由实现与待执行实验账本

日期：2026-10-14（客户端 Australia/Sydney）。本文件是当前实现/验证事实记录，**不是正式路由质量结果**；后续 campaign 真实标签与在线结果到位后必须补充实际 run_id。

## 当前状态

| 分支 | 已实现且已实际验证 | 仍待真实实验 |
|---|---|---|
| Always-BM25 / Always-Hybrid / Always-Rerank | 固定策略、合法路径验证、重放计时标识、错误保留分母 | 同100个冻结test问题的真实输出比较 |
| 规则路由 | 只用可见问题文字；比较/why等走rerank，明确literal/identifier走BM25，其余Hybrid；没有按问句短直接判断容易 | dev最终答案质量与实际在线延迟 |
| TF-IDF + LogisticRegression | sklearn真实运行18条离线合成夹具的train/dev功能验证；版本化JSON+npz往返一致 | 200train/60dev真实生成标签；100test正式结果 |
| TF-IDF + SVD + MLP | 小MLP部分拟合、最多20epoch、dev log-loss选最佳、patience3；真实运行合成夹具测试 | 同上，不能把夹具预测当作效果成绩 |
| 小LSTM | CPU/lazy torch入口、词表仅train、<2M参数上限、固定seed42；隔离models环境已实际训练18条合成夹具、前向/反向、JSON+npz往返与哈希篡改拒绝验证 | 正式200train/60dev生成代理标签训练与100test评估未完成；夹具测试不是任务质量成绩 |
| Laya | 官方来源/revision/license/下载范围核验；已下载并校验固定官方模型、encoder配置和tokenizer，共6文件846,218,761字节；3候选顺序置换、概率验证、低信心/非法输出/截断回退；本地loader哈希与offline边界用明确Mock验证 | 未加载模型权重；同dev/test真实推理、3种顺序分布、校准分析仍未完成 |
| 在线部署门槛 | 禁止用重放估计当成真实在线延迟；dev质量容差0.01、P50至少降低10%、至少30条实际在线记录的门槛 | 尚无真实生成在线记录，默认固定路径，不部署学习路由 |

## 固定协议与接口

配置：`configs/v2/routing.json`。路径ID仅 `bm25 / hybrid / hybrid_rerank`。路由选择已建好的路径，绝不对问题重新切分或重新编码整库。生成模型、上下文打包/父块扩展、数据scope必须与正式比较配置一致。

`RoutingRequest.from_query` 只提取 question、显式用户scope和白名单元数据 document_length_tokens/document_type/language。学习特征不使用scope中的文档ID作为捷径，也不使用答案、gold_doc_id、证据坐标、test分数。文档类型与语言限制为明确类别；问句长度不作为规则判定简单任务的条件。

`freeze_query_ids` 在任何路由成绩产生前以seed42固定200train、60dev、100test（不足时明确actual_counts）。`build_proxy_labels` 对每个冻结问题强制要求三条路径完整记录，包括失败；要求 `real_model=true`、统一 `comparison_config_hash`（排除路径这个实验变量）。缺一条就报错，不能生成随机标签替代。

离线标签按最大答案文字重合F1确定，差≤0.01的路径先选实测总成本更低者，成本仍相同再选证据覆盖更高者，最终按固定路径顺序。所有路径都失败时仍保留该问题且标记all_paths_failed，F1均为0。标签是 **基于文字重合与成本的代理最优路径**，并非真实任务绝对最优策略。不可回答的证据指标为N/A，不能当作覆盖0或1。

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

`load_local`现在在SDK导入前强制检查实装源码凭据。加载期间同时启用HF_HUB_OFFLINE、TRANSFORMERS_OFFLINE和HF_HUB_DISABLE_IMPLICIT_TOKEN，之后恢复原值；始终传本地完整检查点，避免官方远程checkpoint分支读取HF_TOKEN。实际offline loader保护测试1 passed，23 deselected，1.96秒，日志`D:/RAGScope_Runtime/logs/laya_fixed_sdk_guard_tests.log`。此轮仍未调用SDK或加载Laya权重，不能记真实路由质量。

## 离线验证与缺口

当前测试文件：`tests/unit/test_routing_v2.py`。24项离线测试已通过，包括泄漏字段不进特征、冻结ID、F1容差与成本优先、失败标签留存、真实记录门槛、train/dev论文不重叠、线性/MLP/LSTM小夹具实际拟合、二分类概率、JSON+npz序列化、单类塌缩、LSTM预算、三顺序Laya测试对象、加载路径哈希/offline控制、概率校准计算、重放与在线计时区分、部署拒绝。模型环境中的小LSTM只用64词上限、4维embedding/hidden、最多3epoch，全部CPU；这验证代码执行，不证明真实路由质量。

最近覆盖率报告：`D:/RAGScope_Runtime/runs/routing_models_coverage.json`，隔离环境Python `D:/RAGScope_Runtime/envs/models/Scripts/python.exe`。24 passed in61.23s；整体92%，core94%、evaluate91%、Laya适配器96%、learned96%、LSTM97%、CLI0%。没有隐藏正式CLI未执行分支。先前core环境22项测试也通过，旧报告仍保留。此次测试期间Windows平台信息WMI查询打印0x8007000e内存不足诊断，但进程继续，最终24项全部通过；该诊断不被当作模型实验结果。完整reranker/Laya模型仍须等待embedding结束、资源协调后串行加载，正式标签与真实Laya推理仍待补验。

## 正式结果接线与防泄漏补充

ROOT新增`ragscope/routing/campaign.py`与`scripts/run_routing_v2.py`，早期querymanifest位于主run/ROUTING_QUERY_MANIFEST.json，实际冻结200train/60dev/100test，来源SPLIT_MANIFEST哈希核验。只有真实正式生成完整260×3时才输出labels；不可答F1仍N/A，使用明确的correct_abstention代理单列。全部7个必需策略测试决策写成不可变带哈希DECISIONS文件后才允许读取test路径分数做replay。回放只报告实际组件计时之和，在线门槛另等真在线实验。当前正式generationmanifest未就绪，因此READINESS为WAITING_FOR_FROZEN_GENERATION，不声称路由实验已有质量结果。
