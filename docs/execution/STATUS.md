# RAGScope v2 最终执行状态

更新2026-10-14 09:38:00 UTC。验收终态：**COMPLETE_WITH_DECLARED_DEVIATIONS**。25/25任务已签收（24PASS、1PASS_FALLBACK），15/15终验已签收，强制主矩阵5520/5520完整，0missing/0blocked，32条解析失败保留。Goal完成签收及本地提交证据见runtime的FINAL_GOAL_COMPLETION.json与LOCAL_GIT_DELIVERY.json；原任务恢复由用户手动进行。

主矩阵39组、2000次论文聚类bootstrap、错误账本与图实际完成；E2 2400/E3 1200均0error。dev41条件4100条，QASPER509篇/34615证据单元；SQuAD独立100题。旧分片引用不重跑。四模型Qwen3.5 2B/4B/9B与LFM2.5 1.2B按单卡串行实跑，Mock只用于功能测试。

同BF16 LoRA972train/59dev/122步与5→10→122真实恢复完成，100+100配对F1下降、P50升至约3.62倍、错误1+2保留；原OS退出UNKNOWN，独立206票据audit actual0。七路由1120决策先冻后读test、八策略1280重放、Laya1080+3真实概率、train600固定模型诊断、在线60+6完整，规则P50提速5.0446%未达10%门槛。原3次推理前预热拒绝全部保留。默认保持dev预选2B/固定Hybrid/neighbor，不启用LoRA/动态路由，不按test再选型。

最终浏览器实际2问答/5引用/空问题/资源页/历史回放，UI配置发布；8学习文档更新。当前220Python源前后稳定，core710PASS/2skip + models24PASS，去重711PASS/唯一Windows symlink权限skip，两actualexit0。真实集成用实验/在线/UI凭据，不虚构integration/gpu pytest。FINAL_REPORT实际build0并source-bound补结论；随机10/全156表格单元与195raw数字重算全部PASS，人审仍NOT_REVIEWED。

所有本项目CPU/GPU/UI服务、进程、11435/11436/18501端口和锁已释放；设备12MiB/0%是既有Codex，未停止。USER_MANUAL_RESUME覆盖原自动恢复请求，未自动启动原训练/定时/日更/WSL，没有宣称原日志继续前进。原断点历史lease哈希链保留，最终未启动WSL读新checkpoint；.wslconfig原SHA不变。

本地轻量包：D:/RAGScope_Runtime/RAGScope_v2_20261014_results_light.zip，33494364B，SHA4a3e41ab50e0e8dda4e32d4b5e0c1fac0646ae684b431a7e7c4e73d100f269df。实际42members/CRC/SHA/逐条5520原hash与raw输出/errors/scores/timings全等，32失败全部保留，完整FROZEN_INPUTS不变。首次85MB超限包和22MB完整错误账本保留；轻量投影省略逐行重复上下文，不冒充原RecordStore。无权重/环境/原任务快照/上传/push。

最终D运行目录清点：179395文件，逻辑39074233173B、硬链接去重逻辑32346975137B（非物理分配）；整D盘较初始用量增加19884892160B、整C盘空闲增加1372360704B，均含OS/其他程序，不能作项目独占归因。下载/env/cache/index/log/checkpoint在D，C例外为既有Codex/uv/字体/系统Ollama等清单，未声称零C写入。

已披露偏差：E0原embedding版本UNKNOWN，只官方固定重建；R3CPU300与后期GPU360设备/后端不同，旧CPU29partial保留不混算；用户任意阶段GPU追加授权覆盖原规则与历史阶段重叠；symlink权限skip；原BF16 OSexit UNKNOWN/独立审计0；原训练手动恢复。没有剩余强制阻断项。

原始证据：D:/RAGScope_Runtime/runs/20261014_v2；逐任务completion与plan_status，FINAL_CURRENT_REGRESSION、FINAL_REPORT_REVERSE_AUDIT、FINAL_OWNED_RESOURCE_RELEASE、FINAL_PROTECTION_KNOWN_SCOPE、FINAL_LIGHT_PACKAGE_INDEPENDENT_AUDIT、storage_final_delivery_0933。项目最终报告/15验收/四子报告见reports/v2；精确应用启动/退出见根LOCAL_RUN_V2.md。

无需继续模型调用。日后只读核验准确命令：. ./scripts/runtime_env.ps1；& D:/RAGScope_Runtime/envs/core/Scripts/python.exe -m ragscope.experiments.run --profile full --campaign 20261014_v2 --resume --cpu-only --reconcile-only。不要运行原任务restore；原训练由用户自行续跑。最后本地提交与Goal签收由根随后记录，不创建新会话、不规避额度。
