# v3 恢复记录

Campaign：`D:\RAGScope_Runtime\runs\v3_20261015_014506`。当前模型进程：`[{"cmdline": ["D:\\RAGScope_Runtime\\tools\\python\\cpython-3.12-windows-x86_64-none\\python.exe", "scripts/v3/score_bertscore.py", "--campaign", "D:/RAGScope_Runtime/runs/v3_20261015_014506", "--resume", "--device", "cpu"], "pid": 17764, "create_time": 1791992410.1747198, "parent_pid": 35192, "rss": 1511981056, "cpu_seconds": 527.75}, {"cmdline": ["D:\\RAGScope_Runtime\\envs\\v3_metrics_cpu\\Scripts\\python.exe", "scripts/v3/score_bertscore.py", "--campaign", "D:/RAGScope_Runtime/runs/v3_20261015_014506", "--resume", "--device", "cpu"], "pid": 35192, "create_time": 1791992410.1526277, "parent_pid": 18756, "rss": 4419584, "cpu_seconds": 0.0}]`。

先读取STATUS.json/CAMPAIGN_IDENTITY.json/BUDGET.jsonl；不要重复bootstrap、重新封存holdout或重跑旧5520条生成。

CPU BERTScore句柄：unified exec session31866。先调用write_stdin(session_id=31866,chars="",yield_time_ms=1000)，再核PID/create/argv、日志和落盘；观测超时不是退出，不重复起模型。准确只读命令：
```powershell
Get-Content -LiteralPath 'D:/RAGScope_Runtime/runs/v3_20261015_014506/S1_BERTSCORE_CPU.log' -Tail 12
```

只有真实旧进程gone且锁释放后才用以下同身份命令恢复：
```powershell
. ./scripts/runtime_env.ps1
& 'D:/RAGScope_Runtime/envs/v3_metrics_cpu/Scripts/python.exe' scripts/v3/score_bertscore.py --campaign 'D:/RAGScope_Runtime/runs/v3_20261015_014506' --resume --device cpu
```

GPU partial64/6342与两次失败日志保留单列，不能拼进CPU正式结果。CPU全6342对/5720答案完成后保存真实exit/资源与校验，再执行已固定NLI的校准和claim/context/citation proxies。人工NOT_REVIEWED；新LLM请求0，S6保留1800。H_QA gold仍封存，H_scope/H_tasks待构造；S2–S6未开始。原训练USER_MANUAL_RESUME，不自动恢复原训练/WSL/定时/日更。
