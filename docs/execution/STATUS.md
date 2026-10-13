# RAGScope v2 执行状态

更新时间：2026-10-14（Australia/Sydney）。campaign_id：`20261014_v2`。

当前：统一CPU冒烟234项通过；S2真实BM25矩阵、test上下文和E8已执行。阶段三开始资源交接；不是最终验收结果。

- 已完整读取 MASTER_PLAN.md；原件与副本 SHA-256：`157040e71eab86b44bb3007bf43c1ba25240403b892a7cd5f93c1fb37e280117`。
- 项目身份已核验；分支 `codex/ragscope-v2-autonomous-20261014`。
- 原 main 源码、HEAD、状态和 tracked diff 在 `D:/RAGScope_Runtime/backups/20261014_baseline`。
- 原始用户未跟踪文件保留，不纳入本轮提交。CodeGraph 已初始化。
- 阶段一/二严格CPU-only，原GPU程序在这两个阶段未被停止、暂停或修改。阶段三按用户追加授权开始安全交接（见RECOVERY与运行目录gpu_lease.json）。
- 机器：Windows 11、32逻辑核、约15.3 GiB RAM、RTX 5060 Laptop 8151 MiB。
- 初始剩余磁盘：D 232345247744 bytes，C 60073107456 bytes。空闲 RAM 初测约2.4 GiB，加载前重检。
- 并行实现：清洗/切块与结构、QASPER/技术数据、生成/评测正确性；主执行器负责隔离环境和可恢复账本。

新增已验证：
- Python3.12.13解释器和core环境实际base_prefix均在D；74包哈希锁安装成功。
- QASPER509论文，train1000/dev100/test200、global100；技术3页32例；数据复验与153MiB峰值证据见DATA_REPORT。
- 清洗/切块127测试PASS，确定性核心覆盖98%；生成/评测49测试PASS，覆盖97%。
- runner真实子进程中断→半写入隔离→恢复→重复执行测试PASS；ArrayIndex/caches+runner 7测试PASS。
- Qwen3 embedding Q8_0（639150592 bytes）与官方CPU llama.cpp b11429已下载校验；未加载任何模型。
- Windows可用RAM多次0.5～1.1GiB，低于安全加载预算；先完成其他CPU工作。没有停止或调整其他进程。
- 新账本主campaign已实际运行S1.1/S1.5；E0旧BM25子campaign执行中。

已补充验证：
- 统一离线单元/性质测试234 PASS；旧BM25实际200例，Recall@5=.845、Recall@10=.905，保留与旧CSV差异。
- 11组dev切块+上下文策略1500条真实BM25；后续test上下文400条与E8全库100条，合计2000条，失败0。
- E1独立重算33个数字全部一致；逐行context真实规范文本坐标核对1100条通过。
- Dense模型三次CPU内存预检均拒绝加载；此前最低可用约281MiB。这是资源阻断，不能记为Dense完成。
- 用户追加授权可以终止CPU/GPU计算并希望恢复；只对核验的QuantMind研究队列操作，原断点不修改。

下一条准确命令：`Get-Content D:/RAGScope_Runtime/runs/20261014_v2/gpu_lease.json -Raw`
