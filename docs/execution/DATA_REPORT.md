# S1.5 数据准备实测报告

执行日期：2026-10-14（Australia/Sydney）；campaign：`20261014_v2`；状态：`PASS`。
本阶段设置 `CUDA_VISIBLE_DEVICES=""`，未加载模型、未请求 GPU 服务、未干预原 GPU 任务。

## 实际产物与冻结范围

运行目录：`D:/RAGScope_Runtime/corpora/20261014_v2`。
配置：`configs/v2/data.json`，SHA-256：`d8d89a1d1685cdb27675182f306e2143b8264fd532d351969b2abf1f85fa5594`。
冻结清单：`SPLIT_MANIFEST.json`，SHA-256：`901bb1485cc2490fac9f149e02b7f3b48f48e39e0b581852022390473634f5e8`。

| 范围 | 论文 | 冻结问题 | 可回答 | 相对指定论文不可回答 |
|---|---:|---:|---:|---:|
| train | 398 | 1000 | 869 | 131 |
| dev | 36 | 100 | 92 | 8 |
| test | 75 | 200 | 182 | 18 |
| 合计 | 509 | 1300 | 1143 | 157 |

全库检索另冻结100条来自test的可回答问题，保留相同query_id用于配对，scope改为固定509篇全库。
`gold_doc_id_evaluation_only`仅用于评估，不能交给检索器或实时路由。
主任务明确允许用户指定论文作为scope；不可回答标注仅对该论文成立。

源数据包含1585篇、5049问：train 888篇/2593问、dev 281篇/1005问、test 416篇/1451问。
实际官方test档案包含答案与证据，所以直接沿用官方论文级划分，没有另拆validation。
每个split内以seed42及SHA-256排序选择论文，再选择问题。600篇上限内优先满足dev、test，再取train；全部目标配额得到满足。
选择规则在模型结果产生前冻结，未按表现筛题。

## 官方来源、版本、许可

- QASPER v0.3，数据许可CC-BY-4.0。已核验[官方数据卡](https://huggingface.co/datasets/allenai/qasper)及[官方加载脚本](https://huggingface.co/datasets/allenai/qasper/blob/main/qasper.py)。
- [官方train/dev档案](https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-train-dev-v0.3.tgz)：10,835,856字节，SHA-256 `a28fdf966db827bcee3d873107d6b6669864fb7ca8fbf73a192f5e39191bdb5a`。
- [官方test/evaluator档案](https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-test-and-evaluator-v0.3.tgz)：3,865,061字节，SHA-256 `72a52a41193e2838b8074f80ac074b94f956b84886c36a61c58a7df4171bdd72`。
- 技术资料取[CPython v3.12.7官方教程源码](https://github.com/python/cpython/tree/v3.12.7/Doc/tutorial)，固定tag及逐文件SHA-256，许可文件一同保存在raw/technical/LICENSE。

`source_manifest.json`保存每个URL、时间、字节数、SHA-256和实际可用的ETag/Last-Modified；raw目录保留原始tgz、解出的原始JSON、官方数据卡和原始技术文档。
本轮所有下载首次成功，没有切换到镜像；实现单URL至多3次、60秒网络超时及Range续传。

## 证据、隔离与去重

稳定unit_id由论文ID、源章节序号、源段落序号、段落内容哈希构成，在切块前生成。
冻结语料共34,615个证据单元。NFC规范化与换行统一保留段落和代码，所有位置均明确为canonical文档字符坐标、半开区间；不声称是原始JSON字节位置。
证据匹配只对Unicode规范化后折叠空白的完整段落进行精确匹配，不进行模糊猜测。
同一标注匹配多个重复段落时隔离，不能任选一个位置提高覆盖率。
多参考答案与每位标注者的完整证据集合都保留，评估聚合规则为最好合法集合。

全部源问题中1,286条进入`quarantine.jsonl`；原因可重叠，不能把下表相加当独立样本数。

| 隔离原因 | 问题数 |
|---|---:|
| FLOAT SELECTED图表证据 | 737 |
| 可回答但无有效文本证据 | 564 |
| 可回答性标注冲突 | 246 |
| 段落证据匹配不到 | 208 |
| 同一证据文本对应多个位置 | 19 |

原问题、原答案、原始证据、每个失败原因均保存在`all_queries_audit.jsonl`，没有静默丢弃。
核心测试范围是提前声明的纯文本可评估子集；本结果不能外推为含图表的完整QASPER成绩。
冲突问题保留为独立审计组，未混入核心test。

对全部1585篇进行正文精确哈希、相同规范化标题，以及bottom-128词三元组sketch候选+精确Jaccard≥0.9筛查。
实测未发现重复组，别名表及跨split隔离列表均为空。
实现仍明确支持同split别名记录及跨split整组隔离；原始副本始终保留。
近重复是保守的近似筛查，不保证识别所有语义改写；未宣称“不存在任何潜在语义泄漏”。
独立核验确认train/dev/test论文ID交集为空。

## 长度与技术工程集

固定reference tokenizer为`cl100k_base`，包版本在SPLIT_MANIFEST记录。
它用于长度分组和统一文本量，不能代替后续embedding或生成模型的真实tokenizer。
切块前全文长度：medium（513～4096）178篇，long（>4096）331篇，short（≤512）0篇。
连续长度范围790～34,780 tokens；每篇实际值在Document.metadata中保留。
本次QASPER子集没有short，不构造假短论文；短文本回归使用独立legacy SQuAD。

技术功能集实际下载3页官方RST：datastructures、errors、inputoutput，合计32条问答。
每条病例在固定源码内唯一匹配证据，保存unit_id及canonical坐标。
全部标记`agent_authored_not_human_validated`、`independent_quality_benchmark=false`。
用途是工程验收与演示，不冒充人工审核的独立质量基准。
中文/代码切块夹具由切块测试模块单列，不计真实问答成绩。

## Legacy保护

原`data/raw`与`data/processed`中的JSON/JSONL复制到本轮D盘`legacy_snapshot`，逐文件核验源与副本SHA-256一致。
原文件未修改；`legacy_manifest.json`保存源路径、副本路径、大小和哈希。
旧arXiv内容是摘要且没有已标注问题，不能充当已标注长全文问答集。
旧索引和报告的保护清单由主执行账本记录，数据准备脚本未改写它们。

## 实际验证及恢复

命令（项目根目录）：

```powershell
. ./scripts/runtime_env.ps1
& 'D:/RAGScope_Runtime/envs/core/Scripts/python.exe' -u scripts/prepare_v2_data.py --home 'D:/RAGScope_Runtime' --campaign 20261014_v2
& 'D:/RAGScope_Runtime/envs/core/Scripts/python.exe' -m pytest tests/unit/test_data_v2.py -q
```

准备命令exit0；离线15个测试全部通过。
确定性核心覆盖率：qasper.py 98%，technical.py 100%，合计99%；证据在`D:/RAGScope_Runtime/runs/20261014_v2/data_coverage.json`。
另行逐条检查document checksum、所有单元位置回环、query/unit唯一性、scope与证据指向、官方split无交集、所有冻结JSONL文件哈希以及100条全库题的可回答性，见`DATA_VALIDATION.json`。

重新运行准备命令exit0，约29.9秒，冻结manifest哈希完全不变，见`IDEMPOTENCE_VALIDATION.json`。
输出函数比较已存在内容，内容有变化立即拒绝，不覆盖冻结split；下载成功记录复用原始文件且校验哈希。
流式JSON解析逐论文处理，未将全部源数据字典载入内存。
另一次资源核验约30.0秒，采样的进程树RSS峰值160,104,448字节（约153MiB），最低系统可用RAM711,479,296字节，见`RESOURCE_VALIDATION.json`。
该指标是200ms采样的launcher与子进程RSS之和，可能重复计算共享页；最初仅测launcher的5MiB不作为实际解释器峰值。
数据产物约319MiB，包含legacy副本，全部在D盘；无模型权重下载、无C盘大型数据产物。

数据准备已经完成；后续CPU检索直接使用冻结`documents.jsonl`和`queries_dev.jsonl`，不重新选split。
