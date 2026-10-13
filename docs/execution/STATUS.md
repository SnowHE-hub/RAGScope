# RAGScope v2 执行状态

campaign_id：`20261014_v2`；主目标 ACTIVE，尚未最终验收。正式 E3 已完成2B的400/1200条，E2仍为0/2400；不能把预检或代码支持写成实验完成。

项目分支 `codex/ragscope-v2-autonomous-20261014`。任务书完整副本 `MASTER_PLAN.md` SHA-256 为 `157040e71eab86b44bb3007bf43c1ba25240403b892a7cd5f93c1fb37e280117`。原 main、tracked diff 和用户未跟踪状态保存在 `D:/RAGScope_Runtime/backups/20261014_baseline`；用户原文件和旧报告保留。

## 当前执行与资源

- 正式 seed42 2B BF16 LoRA 完成122步/972例/1epoch，进程已退出。真实5→10步恢复验证了 adapter、optimizer、scheduler、RNG 和数据顺序；dev59例最终loss .1356631101。过长剔除28/1例有证据，保留3个完整断点。损失下降不能代替问答质量提升；同精度100题对照尚待最终检索上下文。
- 旧版固定 revision BGE-small CPU 重建完成22,254条向量、Dense/Hybrid各200题、失败0。曾在19,328条触发RAM保护并从第152批恢复；session37101 exit0，原索引与副本8个SHA不变、独立复算PASS。这是明确标注的重建替代，不冒称未知历史 embedding revision。
- v2 CPU Dense/Hybrid 完成5/11个dev切块组、1000条神经检索。CPU embedding服务36892、worker28832/session13666已恢复，缓存和记录保留。最终structured全库/主上下文/ROUTE260完成后安排R3/Laya内存窗口，再继续simple比较索引。
- 自有 GPU Ollama 11435由生成代理持有，PID14564/session54636；2B E3已完成400条并卸载，E0窗口已结束，4B/session78438正在运行，之后串行9B。原11434服务未重启。用户最新指示为原训练和定时检查由用户手动续跑；日更任务不动。禁止自动恢复原任务。
- S1/S2计算始终CPU-only。按用户追加的资源优先授权，S3预检/S4训练与尚未完成的S2矩阵有时间重叠；因此不能声称严格完成阶段二后才停止原任务/进入阶段三。这一执行偏差已记入DECISIONS，最终验收第14项须如实限定。

## 已实际验证

- 硬件：Windows11、32逻辑核、16.4GB RAM、RTX5060 Laptop8151MiB。Python3.12.13及core/models/train/reports隔离环境、下载与缓存均以D盘为目标。core74个锁定包再次只读检查：Would make no changes，pip check通过。
- QASPER509论文，train1000/dev100/test200；全库压力100；SQuAD100题；技术文档3页32个人工编写功能例（不称独立质量基准）。文档级split不交叉。
- BM25 dev1500条、test400条、全库100条合计2000条真实检索、失败0。Dense/Hybrid已有800条。独立审计26个完成run、2800行、12602证据片段、78个均值全部一致。完整神经检索矩阵和最终选型仍待完成。
- 最终rag-json-v2.2与visible-citation-enum-v1：四模型各20条dev实测，输入token80/80与后端一致。三个Qwen20/20解析；LFM19/20，1条length512截断保留、不挑选重试。累计212次请求含全部历史5次错误；预检只证明兼容性。
- CPU BGE-base真实50→10英文/中文及长窗口预检通过；未冒称R3正式矩阵已完成。
- 最新统一离线测试414通过、2项明确skip：Torch测试属于models隔离环境；Windows缺少创建symlink权限。日志和JUnit实证位于runtime/logs/unit_suite_20261014_0355.log与主run同名XML。models隔离环境中的小型LSTM训练恢复功能测试此前实际通过，不等同路由实效实验。
- runner真实子进程中断、半写入隔离、恢复、重复运行幂等验证通过。25任务目录以证据账本核验，阶段结束不是等待点。
- 本地图表已实际生成并目视核验pilot显存PNG/SVG；正式E2质量/延迟图等待真实完整结果。
- 真实E3 2B每方法200题均无请求错误。182可答题文字F1：Oracle .5785604、NoContext .0414931；18不可答单列。10个数字已从原始输出与源坐标独立重算PASS，原始快照在generation_e3/analysis，后续不覆盖。
- 存储中期快照：runtime逻辑文件长度33,319,874,515字节、按hardlink去重27,320,451,700字节；D全盘使用净增27,550,240,768字节、C可用空间净增901,668,864字节，不能把整盘变化全部归因本项目。源码原存档中的2个旧报告及任务书副本3项保留核验PASS。

## 尚待实际运行

最终CPU候选/上下文选择与全库索引；ROUTE与R3上下文；核心E2/E3及冻结扩展；真实路由标签、七策略、Laya、在线配对；LoRA精度一致100题前后对照；自动支持度proxy及人工待核验样例；完整分析、至少10个数字反向核验、中文FINAL_REPORT与15项ACCEPTANCE、轻量结果包、资源清理。

本地提交：9c43163、496354d、af94414、3b49ac4、fd836f4。无push。逐模块恢复点见 `ROOT_CHECKPOINT.md`、`DATA_RETRIEVAL_CHECKPOINT.md`、`GENERATION_HANDOVER.md`；本文件以前的逐时追加快照已留在runtime/status_history和Git历史。

下一条准确命令（只读）：
```powershell
Get-Content -LiteralPath D:/RAGScope_Runtime/runs/20261014_v2_lora/logs/train_2b_1024_seed42_resume10_to_end.log -Tail 2
```
