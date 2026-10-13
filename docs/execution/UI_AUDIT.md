# 本地UI实测记录

本次为正式全部功能验收前的实际浏览器检查；最终dev选定的Dense/Hybrid/重排和实时生成仍须在对应产物完成后验证。

- 本地Streamlit监听127.0.0.1:18501；core CPU-only，关闭usage telemetry，没有启动模型或发出生成请求。
- Chrome工具因本机没有Chrome可执行文件不可用，改用已有Codex内置浏览器后台临时页，无额外浏览器安装。
- 论文qasper:1503.00841，实际输入“What problem does this paper address?”。R512 BM25打包4片段；第一片段ID为qasper:1503.00841:c:05c364c336182216a6f0935f。展开可见原论文文本和9个source span；页面显示检索/总计4065.7ms，生成N/A。该时间是一次含加载的UI观测，不是正式性能统计。
- 实验页加载E3_qwen_4b.jsonl，显示校验400条记录、第一题“How is the data labeled?”及真实拒答提示。1280×720截图实际目视，中文文字/选择/按钮与回放结果无主要遮挡。截图在工具输出中观察，未声称已另存本地截图。
- 检查修正了仅检索模式“模型实际收到片段”的不实措辞，统一显示已打包上下文。
- 临时浏览器页已关闭；服务器实际PID22472及venv launcher27044身份核验后退出，receipt记录释放成功。最初清理探针发现两个匹配项而拒绝行动，随后核验父子/创建时间，没有批量结束Python。

实际证据：D:/RAGScope_Runtime/runs/20261014_v2/UI_INTERIM_BROWSER_QA.json；日志D:/RAGScope_Runtime/logs/ui_visual_qa.log。本次不计入模型实验调用。

启动路径补正：README改用scripts/run_ui_v2.ps1；项目.streamlit/config.toml和CLI均明确127.0.0.1。启动器先检查环回端口，端口占用时不停止原进程。实际-CheckOnly在18501空闲端口exit0，在本项目embedding占用的11436端口exit1；未触及该服务。通过-CheckOnly不意味着已经启动UI或加载模型。
