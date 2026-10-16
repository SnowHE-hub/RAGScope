# 架构手册的离线构建与校验

正文的唯一事实来源是 `docs/architecture.md`。`docs/architecture.html` 由其确定性生成，不单独修改 HTML 正文。工具只读写文档，不调用模型、服务、网络或 GPU，也不修改功能源码。

在仓库根目录执行：

```powershell
& D:/RAGScope_Runtime/envs/core/Scripts/python.exe docs/_architecture/build_architecture.py
& D:/RAGScope_Runtime/envs/core/Scripts/python.exe docs/_architecture/build_architecture.py --check-only
```

使用 Python 标准库即可，不安装 Markdown、Node 或网页工具链。已有 HTML 内容发生变化时，生成器先把其原版按内容哈希备份到 `docs/_architecture/backups/`。本轮开始时两个 `docs/architecture.*` 目标文件尚不存在；仓库根目录原有同名文件完整保留，未覆盖。

最终正文只编辑 `docs/architecture.md`；本目录的 `manuscript_*.md`、`experiment_evidence.md`、`training_routing_evidence.md` 和 `interview.md` 是写作过程稿，不能覆盖已综合校订的最终正文。

`CONSISTENCY_CHECK.json` 记录 MD / HTML / 正文 SHA-256，章节、表格、代码、SVG 和目录数量。校验覆盖：

- 0—18 必需主章节存在，至少六幅实际内嵌 SVG；
- HTML 与当前 MD 的确定性构建完全相同；
- 所有标题与目录锚点一一对应，所有内部链接都有目标，HTML 与 SVG ID 无重复；
- 表格逐单元格文本一致，代码块完整文本一致，中文没有替换字符；
- SVG 是有 `viewBox` 与可访问标题的有效 XML，不含脚本、远程图像或事件处理器；
- 没有脚本、外部样式、字体、图像等离线依赖，也没有未替换的作者占位标记。

该校验证明结构与转换一致性，不代替正文的代码事实、实验口径及路径审查，也不冒充完整浏览器布局检查。

## 六幅同源 SVG

```powershell
& D:/RAGScope_Runtime/envs/core/Scripts/python.exe docs/_architecture/build_architecture.py --diagrams
```

图源在 `build_diagrams.py`，生成到 `diagrams/`。发布时将对应 SVG 的完整文本直接写入 MD，不使用图片链接；HTML 复制这段 SVG，不另取第二套图源。因此修改图后需同步更新 MD 的内嵌 SVG，再重新构建 HTML。

| 文件 | 说明 |
| --- | --- |
| `01-global.svg` | 当前在线链路、共享资产与独立研究链路 |
| `02-offline.svg` | 原文、切块、稀疏 / 向量索引与身份校验 |
| `03-online.svg` | 当前 UI → API → 检索 → 生成 → 解析调用链 |
| `04-retrieval.svg` | 全库文档先验、范围检索与任务证据预算 |
| `05-evolution.svg` | V1—V3 演进及真实结果边界 |
| `06-evaluation.svg` | 检索、生成、在线实验与工程功能验证 |

绿色实线框表示当前链路；紫色表示共享数据；黄色虚线表示独立实验；灰色虚线表示历史非默认实现。全库检索是在全部块候选上融合文档先验，没有画成强制只搜索前十篇论文。6144 token 是复杂任务证据上限，不能理解成每次都读完论文。

六幅 SVG 已用本机已有 `sharp` 实际离线渲染为 PNG 并逐图检查，中文文字、连线和节点可读。可选复现命令使用客户端已带的 Node / sharp，无须下载：

```powershell
& C:/Users/lenovo/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node.exe docs/_architecture/preview_diagrams.js C:/Users/lenovo/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/sharp
```

结果在 `previews/`，只是视觉检查产物，最终 HTML 不依赖这些 PNG。此操作不使用浏览器，也不访问旧工作台端口。

## 转换器自检与参考阅读审计

```powershell
& D:/RAGScope_Runtime/envs/core/Scripts/python.exe docs/_architecture/build_architecture.py --self-test
& D:/RAGScope_Runtime/envs/core/Scripts/python.exe docs/_architecture/audit_renderer.py
& D:/RAGScope_Runtime/envs/core/Scripts/python.exe docs/_architecture/audit_reference.py
```

自检覆盖中文重复标题锚点、转义和代码内竖线表格、代码及 Windows 路径保真、嵌套列表、引用、SVG，以及篡改表格后的失败检测。

扩展校验 `audit_renderer.py` 使用当前已安装的 Pillow 和 Windows 中文字体检查六图文字没有越出 viewBox，并对超过150KB的合成文档进行完整转换与一致性检查；涵盖路径行号、反斜杠、代码中的尖括号、管道和证据标记。结果在 `RENDERER_QA.json`。它不修改正式 MD / HTML，不是模型性能测试，也不替代最终浏览器排版检查。

`REFERENCE_STYLE_AUDIT.json` 记录用户实际提供的 QuantMind HTML 参考文件路径、完整文件 SHA、体积、全部解析的结构数量与样式。仅借鉴阅读设计，不把参考项目的金融模块或实验数字复制到 RAGScope。中文系统字体、暖白底、紫色强调、代码与引用底色、表格、打印和响应式设计沿用参考的阅读风格，并为更长的手册增加桌面粘性完整目录。

## 完整 HTML 浏览器预览尝试与边界

本轮实际尝试了已安装 Edge 155 的标准 headless 截图，只打开本地文档文件，使用当前文档目录内新建的独立 profile，显式关闭 GPU 与后台网络，不接管原有浏览器。准确参数在 `render_document.ps1`，尝试命令为：

```powershell
./docs/_architecture/render_document.ps1 -FirstOnly
```

首个1440×1000桌面截图进程以原生状态 `0xC0000005` 退出，未产生截图；随后停止手机和锚点截图尝试。该结果不是自动审批拒绝。stderr 中的 Edge catalog 与 USB 访问错误不能单独证明崩溃根因。直接启动的进程已确认不存在；CIM 子进程枚举不可用，不能宣称完成全部子进程审计，也没有停止其他进程。

准确命令、当时 HTML 哈希及错误保留在 `BROWSER_RENDER_STATUS.json` 指向的回执中。当前能够证明的是全文结构与内容一致、响应式和打印 CSS 已存在、六幅 SVG 已实际渲染并查看；没有声称完整 HTML 的桌面／手机截图验收通过。没有为此下载浏览器或访问旧工作台端口。

随后内置浏览器因协议安全策略拒绝本地 `file:` URL。本轮没有更换协议、启动 HTTP 服务或改用其他浏览器绕过这一拒绝。上面的 Edge 命令仅是此前尝试的历史记录，不作为继续尝试的建议。最终使用 psutil 精确匹配本轮工作目录内独立 profile 路径，未发现遗留的本轮预览进程；此检查不停止任何进程。结果见 `PREVIEW_PROCESS_AUDIT.json`。

## 最终内容验收

```powershell
& D:/RAGScope_Runtime/envs/core/Scripts/python.exe docs/_architecture/audit_completion.py
```

`CONTENT_AUDIT.json` 绑定最终 MD / HTML 哈希、19章、36组完整面试问答、47个指定术语、38表、6图、当前代码未改动、105份原始实验证据和引用路径行号。最终评分表按四组各自的列核对数值与分母，不把正文其他位置出现相同数字当作表格验证。当前验收状态为文档完成并明确保留完整浏览器视觉预览限制。
