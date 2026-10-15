# RAGScope v3：启动、复核与回退

在 `D:/MCS_26S1课程资料/个人项目/RAGScope` 执行。复用现有D盘环境和模型，不需要再下载。当前默认保留v2；core-first仅供手动开发诊断，未通过质量/效率发布门槛。人工审核NOT_REVIEWED。

## 查看结果和界面

```powershell
. ./scripts/runtime_env.ps1
./scripts/run_ui_v3.ps1 -CheckOnly
./scripts/run_ui_v3.ps1
```

页面监听 `http://127.0.0.1:8502`。未启动问答服务时，仍可查看经过哈希校验的真实实验记录和诊断报告。点击问答会明确报服务未就绪，页面不启动模型、不下载、不使用默认Mock。

## 手动启动真实问答

另开终端，先确认系统可用RAM至少3GiB、11435/11437未被其他服务占用，再执行：

```powershell
. ./scripts/runtime_env.ps1
& D:/RAGScope_Runtime/envs/core/Scripts/python.exe scripts/v3/serve_workbench.py --campaign D:/RAGScope_Runtime/runs/v3_20261015_014506
```

启动器核验当前GPU消费者、独占本项目GPU锁，使用D盘私有Ollama 0.34.4和现有权重。只允许一个0.6B编码器与一个2B生成器，逐次加载/调用，无并行推理。每个新用户任务再次检查3GiB可用RAM；不足时保留拒绝状态。原11434服务不使用。

页面需填写索引内的 `doc_id`。例如已实跑的开发输入：

```text
doc_id：qasper:1608.08738
问题：how did they ask if a tweet was racist?
```

该输入已经用于开发与应用测试，不能再当未见测试题。

查询模式B0沿用v2策略，但此工作台的查询编码使用GPU桥接，成本不是原CPU部署成本。目前只有141篇开发论文的完整范围内向量可用；其他论文的真实问答明确拒绝，不把稀疏回退冒充Hybrid。C1为明确标注的开发诊断。答案缓存默认关闭；打开后缓存命中只记独立应用测试，不并入180次正式未缓存计时。UI调用写入v3预算和APPLICATION_SUPPLEMENT journal。

问答服务终端Ctrl+C正常退出，或按 `WORKBENCH_SERVICE.json` 中实际记录的 `stop_file` 写入退出请求：

```powershell
$service = Get-Content -LiteralPath D:/RAGScope_Runtime/runs/v3_20261015_014506/WORKBENCH_SERVICE.json -Raw | ConvertFrom-Json
Set-Content -LiteralPath $service.stop_file -Value 'Own service exit requested' -Encoding utf8
```

只作用于当前v3服务的精确停止文件。退出后核验服务回执EXITED、对应PID/创建时间和GPU锁，不依进程名称批量结束。UI终端Ctrl+C退出。

## Campaign接口和独立核验

```powershell
. ./scripts/runtime_env.ps1
& D:/RAGScope_Runtime/envs/core/Scripts/python.exe scripts/v3/run_campaign.py --help
& D:/RAGScope_Runtime/envs/core/Scripts/python.exe scripts/v3/run_campaign.py --plan configs/v3/campaign.yaml --resume --dry-run --until acceptance
& D:/RAGScope_Runtime/envs/core/Scripts/python.exe scripts/v3/run_campaign.py --plan configs/v3/campaign.yaml --resume --until acceptance
& D:/RAGScope_Runtime/envs/core/Scripts/python.exe scripts/v3/audit_campaign.py --campaign D:/RAGScope_Runtime/runs/v3_20261015_014506 --independent
```

支持 `--stage S0` 至 `S6` 或 `acceptance`。已完成阶段复用身份一致的证据，不重跑旧矩阵；配置/源码身份改变会拒绝向旧执行身份追加。`configs/v3/campaign.yaml` 是合法YAML的JSON子集，不需要新安装解析器。

exit0表示命令定义的检查成功，不表示性能达标；exit3表示缺失、受阻或未完成证据；非法配置/异常会非零退出。必须同时查看ACCEPTANCE_V3.json中的workflow_finished、execution_complete、performance_status、release_status和human_review_status。

没有通过开发门槛的v3候选，正式H_QA配对保持BLOCKED。此入口不会绕过门槛、读取封存gold或重新训练router/LoRA。报告复核不调用生成模型。

## v2回退

退出v3问答服务及UI，再使用原入口：

```powershell
. ./scripts/runtime_env.ps1
./scripts/run_ui_v2.ps1 -Environment models -Port 8501
```

真实v2生成服务的核验、启动、退出仍遵循LOCAL_RUN_V2.md。v2默认配置未被v3修改，无需reset、复制模型或覆盖旧结果。

原训练策略USER_MANUAL_RESUME：本项目结束不自动恢复训练，不重建定时器，不修改日更、WSL或内存上限。
