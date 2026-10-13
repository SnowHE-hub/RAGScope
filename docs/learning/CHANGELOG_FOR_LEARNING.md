# 按本地提交理解这次改动

这是代码学习记录；实验是否完成以执行账本与逐条真实结果为准。测试通过表示某种行为受到了检查，不能替代模型质量实验。

## 9c43163：先建立可复现的数据和工程基础

过去只保存一个输出文件，很难判断它属于哪一份数据、哪一种切块参数，也难以安全续跑。此提交建立结构化文本、稳定证据单元、可验证索引身份、不可变任务配置和逐行落盘的执行账本，并修复切块的实际预算和前进边界。

可以从 `ragscope/schema.py`、`ragscope/ingestion/cleaner.py`、`ragscope/ingestion/chunkers.py` 与 `ragscope/experiments/ledger.py` 开始读。源坐标指向规范化后的全文，而不是原 PDF 的页坐标；“覆盖证据”必须回到这些坐标计算。

验证包括中文、长单词、代码、重叠、父子块、证据位置、split 隔离和 runner 中断恢复。真实 QASPER 数据另有冻结清单与隔离记录，详见 `docs/execution/DATA_REPORT.md`。

## 496354d：让检索和生成的结果身份可以被检查

旧缓存只有模型名，无法区分同名模型的新权重；旧生成解析也可能把失败包装成带引用的回答。现在索引、缓存和结果携带数据、模型、参数与文本内容的哈希；不同向量空间不能混查；生成错误保留明确错误类别，引用必须来自实际发送的上下文。

重点文件是 `ragscope/indexing/identity.py`、`ragscope/indexing/array_index.py`、`ragscope/generation/context.py`、`ragscope/generation/pipeline.py` 和 `ragscope/evaluation/evidence.py`。教学例子：查询取到了正确论文的标题，并不代表取到了回答问题所需的实验段落；文档命中与证据覆盖需要分别计算。

本阶段运行过旧 BM25 200 题、新 BM25 dev 切块矩阵和独立坐标检查。完整结果与限定条件见 `docs/execution/RETRIEVAL_REPORT.md`，不要从代码支持 Dense 推断 Dense 实验已经全部完成。

## af94414：从逐条结果重建统计，而不是从汇总表抄数字

`ragscope/experiments/analysis_report.py` 在读取结果时核对任务 ID、配置哈希、权重身份、上下文和真实调用票据。缺失的任务仍保留在计划分母中，未完成组不生成最终均值或置信区间；已执行但失败的可回答题按零分保留。

`configs/v2/analysis.json` 在正式 test 生成前声明主要指标、对照和 2,000 次论文/文章聚类 bootstrap。聚类是指抽样时把同一来源文章的问题一起抽取，避免把它们误当成完全独立的观察。多项对照明确标为探索性，不把未经多重比较控制的区间包装成确定的胜负。

8 项离线分析测试实际通过，覆盖缺失分母、失败零分、单论文置信区间和篡改拒绝；它们使用临时教学夹具，未计入真实模型成绩。正式命令为 `scripts/analyze_generation_v2.py --campaign <已冻结运行目录> --output <结果目录>`。

## 尚未写入里程碑提交的运行工作

正在执行的真实模型预检、CPU Dense/Hybrid、路由和 LoRA 不以本页的代码说明代替验收。当前状态与准确续跑命令见 `docs/execution/STATUS.md`、`DATA_RETRIEVAL_CHECKPOINT.md` 和 `GENERATION_HANDOVER.md`。后续提交完成后在此追加实际验证结果。

用户最新要求原训练手动续跑；结束时只释放本项目资源并通知用户，不自动重启原任务。历史停止与断点核验保留在 `docs/execution/RECOVERY.md`。

## 3b49ac4：真实实验的完整执行链

固定模型与上下文身份、统一10,000调用账本、单模型串行服务、逐请求资源采样、训练断点和路由接线落入代码。346项离线测试曾统一通过，Torch专属项在独立环境单列；最终模型20dev预检是实际GPU调用，必须与Mock契约测试区分。LFM输出长度截止的失败保留。代码提交并不等于正式2400+1200核心实验完成，只有逐行记录和完整分母能证明完成。
