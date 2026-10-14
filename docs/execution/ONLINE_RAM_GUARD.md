# 在线RAM边界补充与同协议重新绑定

原在线入口只在加载CPU BGE前检查内存。现保持该有效总3GiB门槛，在每个预热请求和每个尚未执行的正式任务开始前调用 `require_ram(0, reserve_bytes=1280*1024**2)`，要求1.25GiB余量。检查发生在router/retrieval/generation、预算ticket及计时开始之前，不改变计时口径。

不足时以 `INCOMPLETE_RESOURCE` 正常保存summary和pending数量；尚未开始的任务不写失败行、不申请预算、不调用检索/生成。资源恢复后保留旧行和统计，继续剩余正式任务；既定每次session的预热规则不变。CLI仍以非0状态2报告未完成。其他真实失败的原处理不变。

定向16项测试通过：原9项和新增7项；新增覆盖6个预热边界、正式任务边界、已落盘行逐字节保持、无额外请求/预算，以及恢复后完整60条无重复。测试中的客户端是明确功能夹具，未作为模型质量结果。

旧ONLINE_MANIFEST和准备目录文件逐个归档到 `D:/RAGScope_Runtime/runs/20261014_v2_routing/online/history/before_ram_boundary_guard`，原SHA核对一致。随后只读真实dev60路径成绩重新freeze；独立比较全部字段，仅 `implementation` 和 `online_hash` 变化。fixed_hybrid/rules、30个dev IDs、7策略来源、router identities、生成配置、交替顺序与指标规则完全相同，没有重训、调参或test选型。

旧online hash：`2fc8ddb254947304f1d67465362e35b4037542f7a5366cae131501440dbf9eda`。
新online hash：`16a0c607082e95cd54336c86889cf7f887f8bbdaf32f4b6419b7339e0043a3f4`。

实际重绑定进程PID15136正常exit0并核验退出，峰值RSS1,067,438,080字节、最低host available2,901,078,016字节；0模型加载、0 HTTP、0生成。产物 `RAM_GUARD_REBIND_VALIDATION.json`、`RELEASE_AFTER_RAM_GUARD_REBIND.json` 和更新的 `preparation/ONLINE_EXECUTION_PREPARATION.json` 包含源码、测试、原文件及新来源hash。正式60+6调用仍等待GPU/embedding资源交接，不据freeze把S4.4记为完成。
