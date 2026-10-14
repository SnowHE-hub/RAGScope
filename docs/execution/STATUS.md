# RAGScope v2 执行状态

**07:27 UTC 最新覆盖（旧节保留历史）：** 主生成5520/5520已实际完成，32解析失败全部保留；E2 2400完整0错误，main9B100完整0错误。支持性judge200真实完成、1 UNKNOWN解析失败、人审均NOT_REVIEWED，原私有Ollama31596已正常卸载/关闭（Windows退出码1如实保留），11435空闲。当前唯一项目GPU任务是原gen的同BF16配对：PID9824/create1791962055.3476353/session4583；固定E2/R2同100输入/base100+adapter100+各3预热，67/100 baseline时1解析失败，余预计15–25min，不能据中间行称完成。根当前无exec/模型服务。当前源码完整离线回归：core700PASS/2skip + models24PASS，去重701PASS/1 symlink权限skip；219源前后同SHA。两个实际退出码0，最终receipt D:/RAGScope_Runtime/runs/20261014_v2/FULL_OFFLINE_REGRESSION_20261014_0658.json SHA55d9637ef34c9797ea20bccb7beef8c76e12e258fb6b58586aa22759ee232c0e。两次预载RAM拒绝另存，未开suite/模型。models CPU4实际测试与BF16前几条短重叠，不能声称所有GPU延迟为独占测量。

**准确下一动作：** 等原gen BF16完整206 tickets审计并全部释放；根随后在CPU embedding仍OFF时执行全main生产验收和ANALYSIS/2000 paper-cluster CI/图表，保存fresh分析释放receipt；原data凭receipt才实际启CPU11436→原gen单2B/11435→原chunk真实online60+6→根IAB实际生成与发布UI→所有自身服务正常释放→四正式报告/最终反查/15验收/25签收/轻量zip/本地Git。原训练任务USER_MANUAL_RESUME，最终只通知用户GPU已释放，不自动恢复；WSL内存上限/原11434/日更不动。硬截止仍10:29:06.606796Z。


**06:52UTC最新：** E2 9B完整行观察767/800/owner766，进入尾段，最终800哈希/token审计待owner签收。根的大对象主分析/图表将在BF16全部release且embedding未启动窗口执行；之后在线与最终UI服务，再全部释放才构建四报告/终验。新CPU embedding生命周期入口只读准备0load/0HTTP；不会抢占当前GPU或再次正文编码。本文下面0633行数是历史快照。Goal仍ACTIVE。

更新2026-10-14 06:33UTC / 17:33 Sydney。Goal ACTIVE，未终验。硬截止10:29:06UTC /21:29 Sydney，campaign20261014_v2，详细恢复/准确命令见ROOT_CHECKPOINT.md。

用户最新允许任何阶段GPU，单卡本项目模型仍串行。原任务USER_MANUAL_RESUME：结束释放并通知用户，不自动恢复。WSL配置不变且off；不动原11434/日更/驱动/安全工具，不上传/push/付款/搜密钥。大件全部D:/RAGScope_Runtime。

## 正在实际运行

原gen：私有Ollama0.34.4/11435 PID31596/session55731，E2 9B23892/session44959，最近453/800、0error；首50平均3.996s/记录间隔4.078s，余约20–30min。main2B1200/4B420/LFM200已完整、保留1/2/29条长度解析失败。E3全1200、E2 2B/4B各800完整，主5520矩阵已观察约92%但尚未全量验收。队列继续9B剩余→main9B100→judge200→正常服务退出→原E2/R2 BF16paired200，不等待阶段确认、不重跑已封存数据。

root无常驻服务/exec，chunk/data无模型。UI18501已正常请求退出/实际exit1保留、26204gone/port可bind；第一轮真实浏览器四片/证据/资源提示及截图存BROWSER_PHASE1_QA，最终实时UI生成尚未做。

## 已完成实证

- dev4100/41条件0error；全509论文/6551向量/34615证据坐标，simple600/SQuAD100上下文完成。R3 CPU300有效+29历史保留，GPU360有效/0error/13222完整窗，GPU与CPU后端/cost可比性限制明示。12完整检索统计组/5声明cluster对照actualPASS，RETRIEVAL_STATISTICS SHAc300b148...bbff。
- 七mandatory各dev60/test100先冻后读test；三学习router真实训练。8策略共1280重放/80数字独立核验；Laya1080+3warmup/3顺序校准完成，仅proxy诊断。train600固定模型诊断完成，MLP/LSTM train/dev都全BM25，不能称学习模型提升。
- LoRA972train/59dev/122步/1epoch及5→10→122真实恢复完成；同BF16 paired QA仍待。4K/8K共22真实压力/图和QA已完成；E0官方CPU重建400检索/CI/旧索引保护完成。
- 全离线基线679不同PASS/唯一symlink能力skip/216源当时前后稳定；后实际外部E2路径修复25focused、在线RAM边界修复16focused（各新增7），原216中213仍同SHA，不冒称当前已经再全套。最后source稳定需一次最终完整回归。
- 全新D盘core_acceptance环境74锁包offline安装/pipcheck/importsmoke均actualexit0；验收准备759引用核验/3历史before差异公开，并归档实际exit99→0→0恢复fixture。它不是最终15ACCEPTANCE。
- 第一轮真实IAB浏览器BM25证据及资源状态验证0模型调用；旧UI配置尚未正式发布。中文学习材料已更新实际完成事实，最终结果段继续待补。

## 在线准备与终验剩余

完整60dev选择fixed_hybrid/rules各30相同题，rules为28Hybrid/1BM25/1Rerank。当前online_hash16a0c607082e95cd54336c86889cf7f887f8bbdaf32f4b6419b7339e0043a3f4；安全补丁只改implementation hash，旧manifest及prepare原SHA归档，样本/策略/生成配置不变。预载3GiB和每warmup/任务边界1.25GiB保护已16测试验证。正式online仍0run；等gen BF16全释放，根协调CPU embedding和2B私有GPU服务后原chunk实际执行60+6warm，再同窗口最终UI。

最近实际runner reconcile0633：18PASS/1PASS_FALLBACK/6PENDING/exit2；19/25已签收，不等于最终15验收。尚S3.5/S3.6/S3.7/S4.4/S4.5/S4.6。安全且可执行工作持续推进。

必须完成：剩余真实生成/裁判/BF16→主完整分析/clusterCI/图/错误→在线门槛与真实UI/配置→四实际子报告/learning终稿→current-source最后回归→FINAL_REPORT/随机10raw反查/15ACCEPTANCE→最终存储/原件保护/GPU全释放/手动恢复通知→allowlisted轻量本地包/审包/提交。预览、准备框架、离线回放均不冒充最终实际效果。

SOURCE保护：MASTER_PLAN SHA157040e71eab86b44bb3007bf43c1ba25240403b892a7cd5f93c1fb37e280117未改。整理前完整STATUS/ROOT逐字归档主run/status_history，SHA在ROOT_CHECKPOINT首段。分支codex/ragscope-v2-autonomous-20261014。下一准确命令读取 ..._generation_e2/OPERATOR_CHECKPOINT.json；不poll其他agent句柄，不重复起活服务。
