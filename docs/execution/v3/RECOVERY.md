# v3最终恢复与复核

Campaign：D:/RAGScope_Runtime/runs/v3_20261015_014506。
workflow_finished=true，execution_complete=false，performance_status=INCONCLUSIVE，release_status=KEEP_V2_DEFAULT，human_review_status=NOT_REVIEWED。

180正式在线请求、6独立预热、40合成范围查询、两版本合计2真实应用回答与2答案缓存命中已完成；72v3功能回归通过。全部成功且身份一致的结果复用，不重跑旧矩阵、原子化、旧全量NLI或全库Dense。

必需受阻：旧完整claim/support/citation评分、120自然盲评全量proxy、完整块Dense、足够比较/总结标签及各3案例、adapter合并比较、无入围候选的正式H_QA/H_tasks配对及最终统计。失败、部分结果和UNKNOWN全部保留；不可替换裁判或挑已知子集追分。

H_QA200封存，未创建FINAL_EVALUATION_AUTHORIZATION.json；不能用未入围候选读gold。后续研究应使用新campaign和新holdout，当前不会继续调参。

本项目私有服务、11435/11437/8502及GPU/writer/orchestrator锁均已释放，精确PID/创建时间核验见FINAL_RESOURCE_STATE.json。原训练USER_MANUAL_RESUME，不自动恢复、重建定时器或修改日更/WSL。

下一条准确复核命令（不调用模型）：
```powershell
. ./scripts/runtime_env.ps1
& D:/RAGScope_Runtime/envs/core/Scripts/python.exe scripts/v3/audit_campaign.py --campaign D:/RAGScope_Runtime/runs/v3_20261015_014506 --independent
```

启动和v2回退见LOCAL_RUN_V3.md。验收事实以ACCEPTANCE_V3.json及独立审计为准；Git提交本地保存，不push。
