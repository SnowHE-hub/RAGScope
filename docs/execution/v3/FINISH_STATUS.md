# V3 收尾交付状态

2026-10-15 Sydney。功能、真实实验、评价和资源释放已完成，最终本地提交由 GIT_DELIVERY.json 记录。

运行目录 D:/RAGScope_Runtime/runs/v3_finish_20261015_1415；旧 v3_20261015_014506 保留。

- 全部1585篇、19673块完整，原509篇目标缺失0；文档层和块层BM25/Dense/Hybrid全部接通。复用13045、新编码6628向量。
- 默认2B/G0普通问答、结构化长证据总结/比较；全库metadata+document prior Hybrid。生成统一8K驻留，普通2K证据/512输出，复杂6K证据/768输出。
- H_QA200+H_tasks20，最终440同题评分；全部真实，失败保留。原440初版、24长上下文新增、196统一驻留新增均保留；最终220中24严格相同请求复用。
- QA F1 .390401→.406514；任务F1 .427401→.458162、语义余弦 .812471→.859379、源关键点可见57.45%→100%。全库Hybrid完整证据11.18%→17.65%，doc_hit@10却40.5%→38%。不宣称全面改善或显著性。
- 全库1200检索、最终在线60+2预热完整。4K/8K反复重载已修复；最终P50基线1.166/最终1.187秒，P95基线1.479/最终1.988秒；问题版本10.132秒已保留。
- 82回归通过；真实API9场景、实际组件调用模型并显示引用、离线读取220条记录通过。1条跨论文引用过多引起的截断失败保留，修复后复测通过。
- 浏览器视觉验证被自动审批拒绝（用户拒绝访问权限），没有通过其他浏览器/CDP绕过。用本地Streamlit组件↔已授权API11437的真实集成验证替代，不宣称浏览器截图验收。
- 全部14个自有后端已退出、11435/11437/8502端口释放、GPU锁可用；只余既有Codex12MiB/0%。原训练USER_MANUAL_RESUME，没有恢复训练/定时/日更或WSL。
- query缓存24条已按原后端SHA逐位恢复，恢复前字节也保留。新在线编码不写旧缓存。原预算2457条前缀已归档，本轮沿原预算追加，当前3361/6000、511318/4000000输出token。

主要结论 reports/v3/FINISH_REPORT_ZH.md；8交付验收 reports/v3/FINISH_ACCEPTANCE.json；准确使用 LOCAL_RUN_V3.md。不可答正确拒答仍仅10%，人审未完成，事实真值UNKNOWN保留。

本轮没有剩余自动推理任务，不再重跑开发矩阵。日常使用命令：

```powershell
. ./scripts/runtime_env.ps1
& D:/RAGScope_Runtime/envs/core/Scripts/python.exe scripts/v3/serve_full_workbench.py --check-only
& D:/RAGScope_Runtime/envs/core/Scripts/python.exe scripts/v3/serve_full_workbench.py
```

另终端 `./scripts/run_ui_v3.ps1`。停止按当前 FULL_WORKBENCH_SERVICE.json 的精确 stop_file；只释放自有进程。用户原训练自行续跑。
