# RAGScope 架构手册交付状态

日期：2026-10-15（Australia/Sydney）。本轮目标是文档与技术证据核验，不启动先前的训练或项目实现目标。

- 最终正文：`docs/architecture.md`，273,273 bytes，2,639行，41,263个汉字。
- 完整离线阅读版：`docs/architecture.html`，356,432 bytes，由同一 Markdown 确定性生成。
- 内容：0—18共19章、47个指定术语、36组完整面试问答、38表、10个代码块、6幅实际内嵌 SVG、附录A—H及补充索引。
- 证据：当前 HEAD `493dbdad37cd7f372c3121b8332a055d6faa8f7a`，405份Python源码哈希保持一致；105份原始实验文件重新核对哈希；源码引用路径及行号有效。
- 校验：`CONTENT_AUDIT.json` 的19项检查通过；MD/HTML全部表格单元格、代码块及正文转换一致；162个目录目标有效，无损坏锚点，无外部加载依赖。六幅SVG已实际离线渲染为PNG并检查。
- 功能源码未改动，没有重新训练、下载模型、重建索引或运行GPU实验，没有停止其他项目进程，没有push。

## 明确保留的限制

完整HTML的桌面／手机浏览器视觉排版未验证：Chrome未安装，Edge独立截图进程崩溃；内置浏览器随后因协议安全策略拒绝本地file URL，已停止尝试，没有绕过。实际完成的是内容与结构校验和六幅SVG渲染。准确证据见 `BROWSER_RENDER_STATUS.json`、`RENDERER_QA.json`、`PREVIEW_PROCESS_AUDIT.json`。

独立人工答案正确率、预训练污染、任务外泛化及LoRA／重排／规模效应的唯一因果解释仍未知，正文已明确标注；没有把规则、语义相似度或引用可见率当作事实正确率。

## 复核命令

仓库根目录：

```powershell
$env:PYTHONUTF8='1'
$env:PYTHONDONTWRITEBYTECODE='1'
& D:/RAGScope_Runtime/envs/core/Scripts/python.exe docs/_architecture/build_architecture.py --check-only
& D:/RAGScope_Runtime/envs/core/Scripts/python.exe docs/_architecture/audit_completion.py
```

如之后修改最终Markdown，先执行同一生成器的不带 `--check-only` 命令重新构建HTML，再运行以上复核。过程稿不是最终正文的替换来源。所有结果以 `CONTENT_AUDIT.json` 中最终文件哈希为准。
