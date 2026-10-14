# v3 当前恢复点

2026-10-14T19:03:15.905070+00:00

Campaign `D:\RAGScope_Runtime\runs\v3_20261015_014506`；原训练 USER_MANUAL_RESUME，不自动恢复。

当前实际预算：558 次请求；输出已扣除/保留 211975 tokens；以 BUDGET.jsonl 为准。

已核验运行进程：`[{"cmdline":["D:\\RAGScope_Runtime\\tools\\python\\cpython-3.12-windows-x86_64-none\\python.exe","scripts/v3/score_claim_metrics.py","--campaign","D:/RAGScope_Runtime/runs/v3_20261015_014506","--resume","--device","cpu"],"create_time":1792004226.868376,"exe":"D:\\RAGScope_Runtime\\tools\\python\\cpython-3.12.13-windows-x86_64-none\\python.exe","name":"python.exe","pid":30144},{"cmdline":["D:\\RAGScope_Runtime\\envs\\v3_metrics_cpu\\Scripts\\python.exe","scripts/v3/score_claim_metrics.py","--campaign","D:/RAGScope_Runtime/runs/v3_20261015_014506","--resume","--device","cpu"],"create_time":1792004226.838538,"exe":"D:\\RAGScope_Runtime\\envs\\v3_metrics_cpu\\Scripts\\python.exe","name":"python.exe","pid":33984}]`。句柄未知时以 PID、创建时间、exe、argv 联合核验，不能因观察超时重启。

旧 V2 分割保存 83 组（含 3 个失败）；V3 实际 187 请求、183 成功、4 契约失败，另 371 组资源预检阻断而未发请求。V4 只续跑未发请求的组，所有原记录保留。

NLI QA 校准：完整句 24/24，短片段 6/24，问题答案包装 12/24；后两者不得承担自动事实门槛，人工 NOT_REVIEWED。

参考答案不可见的窄 QA 完句规则实际 80/80 构造校准通过；其余语法/短片段仍 UNKNOWN，不宣称自然语料准确率。严禁与分割同时加载 CPU 裁判，以免再次耗尽内存余量。

下一条准确命令：
```powershell
. ./scripts/runtime_env.ps1
& 'D:/RAGScope_Runtime/envs/core/Scripts/python.exe' scripts/v3/checkpoint_campaign.py --campaign 'D:/RAGScope_Runtime/runs/v3_20261015_014506'
```

V4实际终端session57682已exit2：558真实请求齐全，548有效、10契约失败保留；进程已不存在，私有服务卸载/退出。不得重跑分割。

实际GPU裁判210构造检查全部通过，session46755 exit0。评分12条冒烟因均UNKNOWN没有关系调用；原空缓存汇总bug session29339 exit1，已保存源与日志，修复后复用12条且actualexit0。

GPU全量session80992/66941均actualexit1，分别在可用内存3153125376/3206303744字节触发3GiB守卫；其部分结果保留。流式修复后仍不足，GPU全量停止有界重试。

唯一CPU兼容全量正在session24620执行，输出 claim_scores_cpu_v3/；使用已校准CPU依赖栈与相同FP32模型，不混入GPU部分分数。只观察实际PID/创建时间/argv，不并行加载任何裁判。

40范围题已封存，30任务来源论文已冻结且与主分割241篇零交叉；H_tasks参考未生成。117个独立问题/上下文输入已冻结。

分割确认完整558组并实际退出后，依次执行（一个GPU模型，严禁并发）：
```powershell
. ./scripts/runtime_env.ps1
& 'D:/RAGScope_Runtime/envs/core/Scripts/python.exe' scripts/v3/run_context_judge.py --campaign 'D:/RAGScope_Runtime/runs/v3_20261015_014506' --resume
```

每条实际终态和检查通过后才执行下一条。GPU工程门槛失败不能自动进入全量。S2–S6尚未执行，Goal active。
