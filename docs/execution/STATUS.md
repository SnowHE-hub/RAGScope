# RAGScope v2 执行状态

更新时间：2026-10-14（Australia/Sydney）。campaign_id：`20261014_v2`。

当前：S1 数据/性质测试已落盘，S2 BM25 dev矩阵开始实施；S1整体验收尚待统一测试。不是最终验收结果。

- 已完整读取 MASTER_PLAN.md；原件与副本 SHA-256：`157040e71eab86b44bb3007bf43c1ba25240403b892a7cd5f93c1fb37e280117`。
- 项目身份已核验；分支 `codex/ragscope-v2-autonomous-20261014`。
- 原 main 源码、HEAD、状态和 tracked diff 在 `D:/RAGScope_Runtime/backups/20261014_baseline`。
- 原始用户未跟踪文件保留，不纳入本轮提交。CodeGraph 已初始化。
- 全程处于 CPU-only；原 GPU 程序未停止、暂停或修改，未请求任何模型服务。
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

下一条准确命令：`Get-Content D:/RAGScope_Runtime/logs/legacy_bm25.log -Tail 20`
