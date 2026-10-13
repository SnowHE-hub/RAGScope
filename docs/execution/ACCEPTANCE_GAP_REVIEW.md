# 第0/11节有界验收差距复审

复审日期：2026-10-14，Australia/Sydney。依据保存的`MASTER_PLAN.md`第0、11节及实际源码/任务证据；这是终验前检查，不是最终`ACCEPTANCE.json`。当前仍有真实检索和生成进程运行，不能因资源排队标BLOCKED，也不能据此提前标COMPLETE。

## 本次已修复并实测的门槛

1. 历史执行ID即使恰好等于实验任务ID，或配置错误声明`completes_task=true`，都不能使实验/交付任务直接PASS。保留历史记录，只允许工程功能任务从成功命令推导完成；实验须取得真实完整子矩阵或明确签收凭据。
2. S2.3的自动PASS除了全部11组chunk和33组检索结果、相同模型/语料/问题身份，还核验每组恰好100个唯一dev问题、逐行hash/config/index/CPU身份、真实神经检索标记及汇总失败数。BM25的`real_model=false`合法，因为它是确定的真实算法实验。错误行保留在100题分母中。
3. 账本新增`completion_scope`，明确25项任务状态不等于第11节最终验收。最终GPU释放、用户手动恢复政策、轻量分享包、随机数字反查仍须单独证据。

`tests/unit/test_task_catalog_v2.py`真实36 PASS，7.48秒，日志`D:/RAGScope_Runtime/logs/runner_acceptance_guard_tests.log`。新逐行守卫额外核验当时已完成29个E1 run/2900条真实结果，0失败；其余组仍运行，没有把29当33。日志`runner_actual_matrix_row_audit.log`。随后主账本仍8 PASS、1 RUNNING、16 PENDING，非零退出。没有加载模型或运行新实验。

## 需要最终签收人继续执行的检查

| 第11.1条 | 本次复核结论与终验所需证据 |
|---|---|
| 1 环境 | Windows主环境、models/train隔离锁存在；WSL脚本语法已验证，专用core-linux缺失时实际退出2。不能写WSL全功能通过。依赖版本见各env manifest，不能仅凭requirements文件声称重装成功。 |
| 2 测试 | 已有离线和隔离Torch实际日志，主执行器最新414 PASS/2 skip须逐项列原因。CPU功能夹具不作为真实模型质量证据。后续代码修正须相应定向测试。 |
| 3 原件保护 | E0独立审计核验原8个索引和副本SHA不变；源码基线/用户diff由主执行器留存。E0历史权重UNKNOWN，Dense/Hybrid是固定官方版本重建，不是历史权重的精确复现。 |
| 4 结构正确性 | CHUNKING_AUDIT和性质测试覆盖Unicode、预算、offset、父子及scope；正式index manifest仍须随最终选择保存。不能把实际父块metadata漏掉的结果当已扩展。 |
| 5 矩阵/预算 | S2.3新守卫阻止缺行/Mock/换模型。必须完成33组固定dev结果、两个候选与none/neighbor/parent选择，R3真实50→10；最终三模型真实tokenizer统一重打包仍须主manifest及逐行来源。 |
| 6 全链路身份 | 现有模型/缓存/index/prompt/record hashes门槛有效。最终签收引用不可变来源；一次Git换行变动不应回写实验身份，应保留当时源码快照。 |
| 7 核心运行 | 不能用单个`MODEL_*.json status=COMPLETE`推导全任务完成，该标志只覆盖该模型可运行任务。最终需主manifest、独立E3分片的严格ownership导入、完整逐行重建及显式blocked组分母。 |
| 8 错误/人群 | 正式记录保留parse/timeout等错误；不可回答F1/证据N/A不得改0/1。Oracle空上下文拒答定义、图表排除/冲突样本应来自冻结清单。汇总应同时列总体与成功条件分母。 |
| 9 路由 | 七个mandatory策略须先固定全部dev60/test100决策，再读test质量；train200/dev60共780条三路径生成标签不能由Mock或训练准确率代替。Laya1080个原始顺序决策完成仍不代表dev概率校准/最终质量完成。 |
| 10 LoRA | `TRAIN_LOOP_COMPLETE`只证明训练，不替代保存恢复或100×2同BF16对照。正式路径为`train_2b_1024_seed42`；较早未在adapter初始化前seed的run只作可行性/恢复证据。签收需完整checkpoint SHA、有限梯度预检、训练步数、相同inputs/precision/backend的200条结果，或具体资源阻断。 |
| 11 报告/UI | 当前最终报告/ACCEPTANCE尚未形成，不能用代码入口代替交付。UI默认须dev选择且模型/index实际就绪；缺项明确未评测，loopback冒烟须真实，所有学习文件和结果图表须绑定run。 |
| 12 恢复幂等 | 已有专用人工进程exit99、半尾行恢复及重复运行不变测试；E0又真实恢复151批后完成174批。两类证据分别列明，不能把人工fixture写成模型中断恢复。 |
| 13 GPU/原任务 | 最新用户明确由自己手动续跑，优先于原任务书自动恢复段落。最终仅释放RAGScope拥有的GPU模型/服务并验证，记录`USER_MANUAL_RESUME`；不得擅自启动原任务，也不得声称已恢复日志推进。 |
| 14 阶段/磁盘 | CPU-only实际设备/服务证据和阶段交接时序由主执行器维护；当前并行仅是已授权阶段三后的CPU工作。最终核新增D盘用量和C盘例外，不从环境变量正确推断真实落盘正确。 |
| 15 反向核验 | E0已从400条记录独立复算10个方法×指标值，但这不是从尚未完成的FINAL_REPORT随机抽10个数字。最终报告生成后必须另做随机抽样、逐条重算和偏差记录。 |

## 必需组不能从完成标志中遗漏

任务书默认核心为3主模型×200问题×4种检索=E2 2400次，3×200×Oracle/No Context=E3 1200次。其他默认生成是E4 300、E5 200、E6 400、E7最多240、路由train/dev 780，共最多5520条冻结正式任务；另计pilot、warmup、支持性代理、在线与BF16对照，全部受同一10000调用预算约束。E8是100题全库检索证据，不应因没有生成任务而漏验。若E7源/预算可行数量少于40，必须保留生成前筛选和冻结实际N；不能运行后删失败题。

`freeze_generation_v2.py`默认列出E2/E3/E4/E5/E6/E7/ROUTE_TRAIN/ROUTE_DEV，E8由检索campaign单列；`analysis_report.py`发现blocked groups时保持`INCOMPLETE_OR_BLOCKED`。`validate_core_matrix`供小型测试使用时允许显式小样本，因此最终验收必须另对生产冻结split/model列表核验200题/3主模型及已声明替代，不能只引用该函数返回True。

生成模块负责人已新增生产专用`accept_generation_v2.py --scope main`，核验固定生产组、全部逐行来源与blocked分母，统一catalog挂在S3.5；不新增不存在的S4.7。当前主`PRODUCTION_ACCEPTANCE_MAIN.json`为PENDING，独立E3分片`PRODUCTION_ACCEPTANCE_E3.json`真实COMPLETE（1200/1200、0失败），后者明确不能替代主验收。入口无模型调用，仅完整时退出0，否则2。

通用`completion_receipt`核验task规格、requirements名称、artifact哈希和`real_experiment=true`，**并不会理解任意artifact的实验语义**。最终签收仍需使用各模块专用校验器和逐条数据；不要把手填布尔字段当作模型已执行的证明。当前S3/S4没有据此伪造PASS。

## 跨模块问题已修复

`routing/campaign.py`的已存决策读取和replay入口原先只验hash/protocol，没有重新核dev60/test100完整唯一ID集合。此问题已由主执行器修复：校验dev/test唯一完整ID、文件/策略名、合法route、非负耗时及fallback，并要求replay实参与已冻结文件完全一致；全部七策略在读取test指标前完成校验。主执行器新增缺行、重复、换ID、错误策略/路径、负耗时及内存改值七项测试，campaign+online合计19 PASS。本复审未改该模块，实际路由质量实验仍依赖待完成的正式数据。

下一条本代理实际命令仍由资源交接决定：R3源未齐时只运行`models/Scripts/python.exe scripts/run_rerank_v2.py --home D:/RAGScope_Runtime --campaign 20261014_v2`做只读就绪检查；只有完整structured来源和CPU RAM窗口交回后才加`--run`。Laya也保持已冻结输入、未加载状态，不与当前生成模型/embedding争抢内存。当前等待资源属于PENDING，不是硬阻断。
