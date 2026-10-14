# 当前可执行命令与阶段状态

运行目录 `D:/RAGScope_Runtime/runs/v3_20261015_014506`。这不是最终启动说明；S2–S6 尚未完成。原训练 USER_MANUAL_RESUME，不自动启动。所有实际结果、失败、预算和进程以运行目录原始 journal/收据为准。

当前覆盖下方历史命令的恢复点：分割实际完成558组且退出；GPU全量两次触发内存守卫，不再重启。CPU兼容全量已启动，session24620，输出 `claim_scores_cpu_v3`。先观察这个句柄，禁止同时加载其他模型。确实退出后核验全部5720条，再执行9B上下文判定。下方GPU命令保留执行历史，不是当前重启建议。

先观察当前任务，禁止因句柄观察超时重新启动分割：

```powershell
. ./scripts/runtime_env.ps1
& 'D:/RAGScope_Runtime/envs/core/Scripts/python.exe' scripts/v3/checkpoint_campaign.py --campaign 'D:/RAGScope_Runtime/runs/v3_20261015_014506'
```

分割实际后台句柄 session57682，对应 `resume_atomization.py`；确实终态后才可进入下一GPU任务。已有187组结果由V4复用，另外371组未发请求的资源阻断只续跑一次，不使用旧的V3入口重跑。

GPU评分前的实际栈校准（尚未运行时为待执行，不是完成证据）：

```powershell
. ./scripts/runtime_env.ps1
& 'D:/RAGScope_Runtime/envs/v3_metrics/Scripts/python.exe' scripts/v3/calibrate_gpu_claim_judge.py --campaign 'D:/RAGScope_Runtime/runs/v3_20261015_014506' --resume
```

只有GPU校准门槛通过、分割558组齐全且原任务资源释放，才执行12条评分冒烟；真实通过后再补齐5720条，不重复已有关系或答案：

```powershell
. ./scripts/runtime_env.ps1
& 'D:/RAGScope_Runtime/envs/v3_metrics/Scripts/python.exe' scripts/v3/score_claim_metrics.py --campaign 'D:/RAGScope_Runtime/runs/v3_20261015_014506' --resume --device cuda --max-records 12
& 'D:/RAGScope_Runtime/envs/v3_metrics/Scripts/python.exe' scripts/v3/score_claim_metrics.py --campaign 'D:/RAGScope_Runtime/runs/v3_20261015_014506' --resume --device cuda
```

然后由现有9B先运行60组构造校准，通过后执行117个独特问题/上下文代理判定。输入不含模型回答、参考答案或答案标签；所有新增请求接入v3预算。不能与评分或分割同时加载：

```powershell
. ./scripts/runtime_env.ps1
& 'D:/RAGScope_Runtime/envs/core/Scripts/python.exe' scripts/v3/run_context_judge.py --campaign 'D:/RAGScope_Runtime/runs/v3_20261015_014506' --resume
```

辅助范围40题及30篇任务来源已冻结，可复验身份；H_tasks参考仍待真实生成，不读取H_QA gold：

```powershell
. ./scripts/runtime_env.ps1
& 'D:/RAGScope_Runtime/envs/core/Scripts/python.exe' scripts/v3/prepare_auxiliary_splits.py --campaign 'D:/RAGScope_Runtime/runs/v3_20261015_014506' --resume
& 'D:/RAGScope_Runtime/envs/core/Scripts/python.exe' scripts/v3/prepare_context_judgements.py --campaign 'D:/RAGScope_Runtime/runs/v3_20261015_014506' --resume
```

现有评分契约、代理边界和证据见 `reports/v3/METRIC_SPEC.md`、`ERROR_ATTRIBUTION.md`；这些命令不代表最终候选或发布已经完成。
