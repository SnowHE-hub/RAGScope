# Laya 固定官方模型真实执行

2026-10-14 05:17:38 UTC 独立验收通过。运行目录为 `D:/RAGScope_Runtime/runs/20261014_v2_laya`。这是原始结构化路由决策的完整执行验收；真实路径代理标签的校准、答案质量重放与部署判断仍待生成结果齐备。

固定200 train、60 dev、100 test问题，每题3种原定候选顺序，共1080条正式记录；另有3条固定dev预热，不混入分母。模型调用错误0条；固定0.5信心门槛下646条回退Hybrid；142/360题在三个顺序下最终选择发生变化。这些是执行与顺序敏感性事实，不能解释为答案质量提升。

模型revision `7b928d828b7b0e022f929d9bd2e44165aa270148`，SDK 0.4.6 commit `c73908d96ff73c0062c340a11bf7c54ec7d8bc4d`，权重SHA `891102d372688fc2a094dac56a384bc537b87c63f21f9f3dac0be2b7cbc8d86c`。实际backend eager、CPU4、FP32、compile=False、Torch 2.9.1+cpu。SDK文件与全部检查点文件加载前后哈希一致，没有下载或改权重。

SDK实际打印choice:11+温度条目0.10058280825614929超出其支持区间，运行时按SDK自身规则夹到0.5，并警告相关条目信心不可视为已校准。日志保留该警告；本项目没有改SDK或温度。正式任务仍需自己的ECE/Brier检验，不能用上游声明代替。

实际加载75.774秒，含加载的正式工作段1003.967秒。正常≥5GiB预载路径成功；唯一4.5–5GiB备选未消耗。attempts=1、backend_failures=0。1秒资源采样观测峰值RSS 2,928,300,032字节，最低host available 1,697,701,888字节；原1.25GiB运行保护线保持不变。后段与本项目2B GPU生成并行，耗时只能按这个执行条件理解。

exec session76916退出0，精确PID11880/create1791954014.0780013已不存在，Laya writer.lock实际重新获取并释放；签收时host available 4,259,045,376字节。没有操作其他模型、原用户任务、WSL或GPU资源。

| 证据 | SHA-256 |
|---|---|
| records.jsonl，1080条 | `5ba2606cc9aab3e0c88d8d060d178b23ba6c6386b790ec42d35850240f906666` |
| RAW_ACCEPTANCE.json | `292a33d2d02b09f5e143ef80f94c6c07de31ea0c5d6f234fd2cb770f4a0d3b8b` |
| RELEASE_AFTER_RAW1080.json | `0b3d95a0893ba56b24b623edf1f40100198ed7bd1c001247437f0b423f75cdef` |

独立验收入口 `scripts/validate_laya_raw_v2.py` 核验所有360×3人口、题目可见输入哈希、逐条记录哈希、原始与归一化概率、固定回退规则、真实模型/SDK身份、3个预热、汇总分母和实际进程退出。31项轻量验收/报告测试已通过，日志 `D:/RAGScope_Runtime/logs/laya_report_validation_tests_20261014.log`；真实验收日志为 `laya_raw_independent_acceptance_20261014.log`。

两个预加载入口曾被模型排他保护拒绝，均在导入模型前退出，未消耗加载或备选次数。第二次精确诊断证明Windows自有venv launcher的cwd是D盘tmp；以完整exe、argv、父子关系、创建时间窗口、cwd精确限制修复，未泛排Python进程。原日志及preload_guards目录保留，修复已提交4098625。

后处理 `python -m ragscope.routing.laya_report` 不加载模型。默认顺序始终是原order0；必须先冻结全部7个必需策略及Laya的dev/test决策，才读取真实路径质量。报告三个顺序的分布、选择变化、10固定bin的ECE、Brier与描述性重放；不挑最佳test顺序、不拟合额外温度。无效概率不能计算ECE/Brier时单列缺失数，仍保留完整调用分母。
