# v3 当前恢复点

2026-10-15T00:33:30.116039+00:00

Campaign `D:\RAGScope_Runtime\runs\v3_20261015_014506`；原训练 USER_MANUAL_RESUME，不自动恢复。

当前实际预算：2269 次请求；输出已扣除/保留 394103 tokens；以 BUDGET.jsonl 为准。

已核验运行进程：`[]`。句柄未知时以 PID、创建时间、exe、argv 联合核验，不能因观察超时重启。

旧 V2 分割保存 83 组（含 3 个失败）；V3 实际 187 请求、183 成功、4 契约失败，另 371 组资源预检阻断而未发请求。V4 只续跑未发请求的组，所有原记录保留。

NLI QA 校准：完整句 24/24，短片段 6/24，问题答案包装 12/24；后两者不得承担自动事实门槛，人工 NOT_REVIEWED。

参考答案不可见的窄 QA 完句规则实际 80/80 构造校准通过；其余语法/短片段仍 UNKNOWN，不宣称自然语料准确率。严禁与分割同时加载 CPU 裁判，以免再次耗尽内存余量。

下一条准确命令：
```powershell
Get-Content -LiteralPath 'ragscope/v3/local_runtime.py' -Raw
```

V4实际终端session57682已exit2：558真实请求齐全，548有效、10契约失败保留；进程已不存在，私有服务卸载/退出。不得重跑分割。

实际GPU裁判210构造检查全部通过，session46755 exit0。评分12条冒烟因均UNKNOWN没有关系调用；原空缓存汇总bug session29339 exit1，已保存源与日志，修复后复用12条且actualexit0。

GPU全量session80992/66941均actualexit1，分别在可用内存3153125376/3206303744字节触发3GiB守卫；其部分结果保留。流式修复后仍不足，GPU全量停止有界重试。

CPU兼容session24620也actualexit1；335实际评分/230关系判断保留。全量旧NLI已到有界阻断，不再加载。所有5720条覆盖/UNKNOWN及盲评120条部分包保存，未执行行不冒充模型结果。

40范围题已封存，30任务来源论文已冻结且与主分割241篇零交叉；20有效H_tasks参考/29失败任务分支保留，全部人造AI标签。117个独立问题/上下文输入已冻结。

9B上下文独立校准59/60。session21014在104/117后资源exit1，6771只续跑13条actualexit0；117条完成。23新增拒答：YES8/NO12/UNKNOWN3，均为proxy，不是人评。

S2输入冻结24题×2上下文，G0真实144复用。72新direct/pipeline调用session65647 exit0，36配对完全一致、无token/思考通道差异。主矩阵session84289 exit0，864完整记录（新720）独立计分验证；含6prior探针。

H_tasks整篇来源session71248 actualexit0，30请求全部完成；初版16有效任务/20失败论文。一次零生成成本的空白格式修复和独立任务验证后，20有效任务、29失败分支保留；H_tasks.questions_v2/gold_v2为当前辅助人造任务集，不是官方QASPER或操作者盲测。无后台模型进程。

当前下一项：Quality admission cannot be proved: retain all UNKNOWN questions and v2 default. Complete independent real online cost and UI/campaign interfaces, then apply candidate locks/gates; never substitute a known-only subgroup or read sealedH_QA to tune.

S2_SUPPLEMENT_RESULTS_verified.jsonl是42条补充结果的当前校验投影。v2封装首行遗留旧record_hash参与新hash的缺陷有确证，原两份journal保留，逐个raw请求/输出完全相同；不得重发调用。

S3旧500记录漏斗与9519窗口核验、400同预算core-first对照已落盘；仅旧诊断，不能当成新D_confirm或holdout。Goal active，execution_complete仍false。
