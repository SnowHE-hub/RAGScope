# 生成实验生产规模验收

`scripts/accept_generation_v2.py` 只读取已有产物，不调用模型、不修复日志、不改写冻结配置。小规模功能测试仍可使用通用 campaign harness；生产验收另有固定人数门槛。

主验收命令：

```powershell
. ./scripts/runtime_env.ps1
& D:/RAGScope_Runtime/envs/core/Scripts/python.exe scripts/accept_generation_v2.py --campaign-dir D:/RAGScope_Runtime/runs/20261014_v2_generation --scope main
```

验收从独立指定的冻结语料读取完整 QASPER train1000、dev100、test200 和 SQuAD test100，核验文件哈希、唯一 ID、split 和逐题原始内容。主 manifest 的来源必须与该语料一致，不能用缩小后的 manifest 自证完整。

- E2：三模型 × R0/R1/R2/R3 × 全部200个 test ID，即2400任务。
- E3：同三模型 × Oracle/No Context × 同全部200题，即1200任务。
- E4：同三模型各100个冻结 SQuAD问题。
- E5：LFM 的 BM25/Hybrid，各同一 seed42 的100题子集。
- E6：2B/4B 的 simple/structured，各同一100题子集。
- ROUTE：2B × 三路线，train200、dev60，按原规则重建子集。
- E7：2B/4B × 三位置，使用已冻结、最多40题的可行性集合；不足40必须有逐项排除并穷尽候选的证据。完整受阻时仍保留每条件40的目标分母。

每个模型/方法条件必须恰好声明一次。重新计算 manifest 哈希不能掩盖缺组、少题、少模型、重复 task 或改写问题。READY 任务按完整笛卡尔积独立重建；BLOCKED 必须有原因及仍匹配哈希的实际证据。所有来源由 `validated_rows` 校验，外部分片保留原 manifest/config/record hash，不重复复制；失败请求计入 observed 和失败数。

状态含义：`PENDING` 尚无完整冻结配置或尚未生成；`RUNNING` 有真实结果但仍缺任务；`BLOCKED` 仅用于带证据的受阻条件；`FAILED_VALIDATION` 表示完整性或生产人数校验失败；`COMPLETE` 需要全部预定条件有对应真实记录且没有未执行或受阻条件。完成不代表质量提升，失败仍可成为已执行结果。字段 `audit_generation_calls=0` 只说明验收程序没有新增调用，不是原实验调用数。

专用 `--scope e3` 只检查独立 E3 分片，明确不能授予主矩阵完成状态。实际运行结果：

- `20261014_v2_generation_e3/PRODUCTION_ACCEPTANCE_E3.json`：COMPLETE，1200/1200，0错误，0缺失。
- `20261014_v2_generation/PRODUCTION_ACCEPTANCE_MAIN.json`：PENDING，主 manifest 尚待最终检索输入冻结；退出码2。
- 10项隔离功能测试通过，覆盖缺组、199题、重复任务、少模型、重新计算哈希后改题、缺扩展组、BLOCKED证据漂移、无manifest和失败保留分母。

以上状态对应本次执行时刻；后续重跑验收应读取新生成 receipt。E3 原始完成证据仍由 `COMPLETION_RECEIPT.json` 封存，完整统计和独立复算由 `analysis_complete` 保存。
