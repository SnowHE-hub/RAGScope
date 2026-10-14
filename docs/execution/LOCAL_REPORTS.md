# 四份局部报告构建入口

`scripts/build_local_reports_v2.py` 是只读消费者，只生成 `RESOURCE_REPORT.md`、`TRAINING_REPORT.md`、`ROUTING_REPORT.md`、`RETRIEVAL_TABLES.md` 和局部构建收据。它不调用模型、不计算新的bootstrap、不训练/选择策略，不生成 `FINAL_REPORT.md` 或最终 `ACCEPTANCE.json`，也不改变UI默认。

本次只完成工程准备与9项离线契约验证（1.90秒），没有读取主矩阵test质量分析或构建这四份真实报告。实际 `--prepare` 已exit0，返回 `PREPARED_NO_EXPERIMENT_READS`；九个来源绑定暂缺，明确列出UNBOUND，不能据此声称分析完成。凭据在 `D:/RAGScope_Runtime/runs/20261014_v2/local_report_preparation/ENGINEERING_VALIDATION_V2.json`，模板为同目录 `SOURCE_SPEC_TEMPLATE_V2.json`。root后续确认全部matrix完成和七策略冻结后提供真实路径/哈希，再进入正式构建。

## 来源与复用边界

先用CodeGraph定位并复用既有 `scripts/build_final_report_v2.py` 的来源结构，以及以下现成验证/汇总函数，没有修改root报告器或analysis模块：

- `analysis_report.validated_rows`：检查主清单、所有真实记录、config/record hash、完整task分母与外部shard归属。资源表直接取既有ANALYSIS的performance和latency字段。
- `ledger.read_records`：原JSONL完整行、record hash与config身份；检索统计每个PASS组还核实际query ID全体及失败数。
- `routing.campaign.validate_decisions` 与 `online_campaign.verified_body`：必须先验证七个策略dev/test完整冻结，才打开test-bearing分析；重放summary通过现成 `routing.evaluate.summarize` 从原rows复算。没有新标签/排序或路由选择。
- `evaluation.v2.summarize_records`：完整BF16 paired两组100题逐行复核和原summary比对，不另写质量算法。训练状态仅消费实际training_result/TRAINING_AUDIT及其已绑定原始artifact。
- RETRIEVAL_STATISTICS中的原bootstrap均值、区间、适用N、cluster数、重采样次数直接展示，不增加对照或重算统计。

每个显式来源须有完整 `{path, bytes, sha256}`，会核原文件及其内嵌artifact依赖。文件缺失/未绑定保持UNAVAILABLE；哈希变化直接拒绝构建，不吞掉错误。paired或online未完成不得展示为完整质量结论，失败和pending保留。来源实际命令只从显式绑定的已有receipt中的command/argv/worker_command等字段读取；没有收据就写UNAVAILABLE，不把建议命令冒充历史执行。

## 规格与准确命令

source spec包含 `version=local_reports_v1`、明确 `campaign_dir`、`routing_dir`、`sources`、`replays` 和 `command_receipts`。sources键为 analysis、retrieval_statistics、training_result、training_audit、paired_summary、online_summary、online_manifest、gpu_r3_migration、gpu_r3_validation；缺项可以为null。replays按七策略名和可选laya分别绑定实际REPLAY文件artifact，未绑定就UNAVAILABLE，不自行打开猜测路径。root提供最终来源时，应另存新规格而非覆盖准备模板。正式构建要求main ANALYSIS状态COMPLETE、全部真实task/各组分母完整，并与routing protocol的manifest一致。

已实际执行的准备命令：

```powershell
. ./scripts/runtime_env.ps1
& 'D:/RAGScope_Runtime/envs/core/Scripts/python.exe' scripts/build_local_reports_v2.py --prepare --spec 'D:/RAGScope_Runtime/runs/20261014_v2/local_report_preparation/SOURCE_SPEC_TEMPLATE_V2.json'
```

最终操作由root确认窗口后使用 `--build --spec <实际已绑定spec> --output <全新报告目录>`。输出目录已存在即拒绝，保留旧报告和收据。每份报告附同次完整source hashes、清单范围/分母、实际构建命令与来源命令receipt，所有输出再写进 `LOCAL_REPORTS_BUILD.json`。本次没有执行此正式命令。

## 必须保留的解释限制

GPU R3的E8/ROUTE训练成本与原CPU E2 R3及CPU实际online条件不同，不能直接作部署延迟收益；重放组件成本之和不是真实在线端到端时间。TTFT保持N/A，device-total显存采样包含其他进程且可能漏峰，PyTorch allocator量不能冒称物理显存。训练step耗时不含checkpoint/dev/load。BF16 paired与Q4 schema协议不同，跨协议差不能归因于LoRA或精度。全部多重对照为探索性、非因果，不读test选默认。所有人工审核为NOT_REVIEWED，局部构建成功不代表终验。
