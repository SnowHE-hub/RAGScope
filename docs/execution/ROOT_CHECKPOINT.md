# 主执行器恢复点

2026-10-14 04:03 Australia/Sydney。Goal ACTIVE，无预算；全部可执行任务继续，尚未最终验收。

用户最新恢复规则：原训练和定时检查已停、日更不动；RAGScope释放资源后通知，由用户手动续跑。gpu_lease=USER_MANUAL_RESUME，禁止自动恢复、不得再动原项目或任务。

当前进程/代理：
- generation_evaluation持自有Ollama11435，exec54636/PID14564，GPU单实例锁。正式E3独立shard已冻结1200tasks，2B400/400真实完成、error0；E0窗口已交回，4B exec78438正在运行，之后串行9B400。root不得并开GPU模型。主generation完整manifest尚缺最终retrieval上下文。
- chunking_core的E0续跑exec37101 exit0，实际PID37108已退出；22254/22254向量、Dense200/Hybrid200真实完成，无错误，独立复算及8个原索引SHA不变PASS。该代理还负责R3 CPU真实50→10源物化、Laya真实分类，以及等待窗口期间有界验收差距审阅。
- data_preparation CPU embedding服务36892/create1791909754.8890424、Dense worker28832/create1791909810.3224635、session13666，当前6/11dev组完成/1200neuralrows，S512编码中。先完成11dev和扩展选择，再structured全509/test200/global100/ROUTE200+60物化，然后在正常命令结束边界停止自有CPU服务给R3/Laya内存窗口，再继续simple/SQuAD。禁止重复worker或中途切断HTTP。已消耗2/3恢复（包含已记录计划内暂停导致的10054）。
- 正式seed42 LoRA122/122完成，972train/59dev/1epoch，TRAINING_AUDIT PASS，三init相同SHA、5/10步精确恢复、3完整断点、best=step122。training进程/GPU已退出。最终devloss .1356631101不是问答质量改善；相同BF16 base/adapter100test仍缺最终R2上下文。

E3 shard：D:/RAGScope_Runtime/runs/20261014_v2_generation_e3，manifest_hash8c6a71031d25f0a0218a85a3abbd9d8dc35d3c0f56dafa046e8249beeffb5599。E2仍0/2400。2B每方法200题，其中182可答；oracle F1 .5785604023，no_context .0414930593，CI2000paperbootstrap；18unanswerable另列。root已实际analysis+独立10数字raw/source审计PASS。保持e3/analysis/不可变快照不覆盖；4B/9B齐后另用analysis_complete。global budget615reserved/core400（之后代理会继续增长）。

独立分组联合：generation_campaign.iter_record_sources(directory,manifest,inputs)返回{directory,manifest,inputs,owned_task_ids}。最终freezer自动显式引用E3 shardmanifest；所有external task即使missing也不在main重复跑。严格验task/input/model/gen/protocol字段、禁止嵌套/重复来源；analysis和独立auditor都已接入。不可复制改写E3原行冒充main记录。

最新工程：core全套414PASS、2明确skip(Torch隔离models、Windows无symlink创建权限)，D/logs/unit_suite_20261014_0355.log与同runJUnit，unit_suite_latest.json持证据。随后online新增1项、Laya等专项单列。模型测试/真实结果不能由此推断。root package_results_v2.py已2fixture测试PASS，仅最终15项ACCEPTANCE均终态时本地allowlist打包，保留失败和原hash、清楚标portable路径改写、不含权重/私有任务快照，未真正打最终包。

Windowsbootstrap现固定3.12.13+require-hashes，已有env只dry-run/check不重建；真实74包无变化兼容PASS。run_v2.ps1新增-Gpu/-ReconcileOnly/-CoordinatedTask，Windows实际对账exit2（8PASS其余等待）。WSL bash-n通过，实际wrapper明确Linuxenv缺失exit2，未安装/重启WSL，CPU预检历史被RAM阻止；不能宣称Linux完整跑通。

存储中期实际runtime117247files，逻辑33.319874515GB、hardlink去重27.320451700GB；D全盘净增27.550240768GB、C空闲净增.901668864GB，不作独占归因。storage_mid_training_0338.json保存快照；final需重跑新snapshot。已知C例外：原uv/Ollama/Codex可执行资源与系统字体读取，模型/cache均D。PRESERVATION_AUDIT.json旧2报告+master副本3项SHA一致。MASTER_PLAN原件SHA157040e71eab86b44bb3007bf43c1ba25240403b892a7cd5f93c1fb37e280117。

接下来：CPU矩阵/两候选冻结；structured源→R3/Laya；simple/SQuAD；完整生成manifest引用E3并运行E2与扩展/route780；真实七路由学习与dev选择/paired在线30×2；LoRA100pair；supportjudge≤200；正式统计/图/错误/独立反核≥10/中文FINAL_REPORT/15项ACCEPTANCE/轻包；最后退出本项目CPU/GPU并通知用户手动续跑。LayaINPUTS已实际冻结train200dev60test100三order1080决策，hashb82d4b56253f1773672fc35a49402b38409256a6e892777cd55fa64c20b270f5，尚未加载；需5GiB free且留1.25GiB，等待上述RAM窗口。

S1/S2计算始终CPU-only，但完整S2尚未完成时已根据用户追加资源优先授权交接S3/S4；DECISIONS已明示阶段顺序偏差，最终criterion14不能虚构原任务直到阶段二结束仍运行。

本地分支codex/ragscope-v2-autonomous-20261014；已有9c43163/496354d/af94414/3b49ac4/fd836f4，root准备新里程碑。不要stage用户原untracked plan/zip/startprompt、architecture.md/html、rerun/、svg/。无push。

下一条准确命令（只读，不加载模型）：
```powershell
Get-Content -LiteralPath D:/RAGScope_Runtime/runs/20261014_v2/generation_service.json -Raw
```
