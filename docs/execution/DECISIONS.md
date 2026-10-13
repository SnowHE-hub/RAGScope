# 决策记录

## 2026-10-14 初始化
- 以当前已打开且含 ragscope/、scripts/build_indexes.py 的目录为原项目，不搬迁。
- 原 Git tracked 内容无修改，用户未跟踪文件按清单保留；只提交本轮明确文件。
- CodeGraph 无索引。按本次自主执行授权初始化本项目索引，不修改全局设置。
- Runtime 原不存在，创建 `D:/RAGScope_Runtime`；缓存只在本项目子进程设置，不使用 setx。
- 32逻辑核按任务书限制为4线程；初始空闲RAM较低，CPU模型需串行、低内存实现与加载前预算核验。
- 采用 Windows Python 3.12 隔离环境；现有 uv 可执行文件可用，但尚未发现受管已安装 Python。必要时由 uv 从官方 Python standalone 发行资产安装到 D 盘专用位置。
- 保留现有模块增量修改，不新建平行项目。Mock仅离线功能验证，正式结果必须有真实模型身份。
