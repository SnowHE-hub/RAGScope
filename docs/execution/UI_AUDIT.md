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

后续工程接入已完成22项离线契约测试（ui_routing_live_tests.xml）：实验路由只能绑定同一就绪索引与冻结扩展，逐次读取校验后的规范文档metadata，标准答案和evidence标签不会作为特征。规则/线性/MLP/LSTM按实际ready项显示；缺失项保留资源页未就绪说明。未经真实dev质量与至少30题/策略在线延迟门槛的路由，即使配置写为default，也不能成为默认选项。

交互生成现在共用generation-client.lock与原进程/租约核验；每个实际请求进入全局generation budget，以UI用途单列，不混入benchmark。带配置hash的交互结果持久化在主run/ui。显示的总耗时包含索引/路由来源核验、加载和请求，原pipeline时间另存。以上是实现及离线功能验证；尚未运行正式最终UI的路由/真实生成，不将fixtures当作结果。

独立代码审查又发现并修复：仅核模型名会漏掉同名tag替换；现同时核manifest digest、服务实际模型目录和weight blob字节，调用锁内再次核验。四模型实际本地包CPU核验通过，`UI_MODEL_PACKAGE_PINS.json`明确不等于服务就绪，生成调用0。两个准备脚本错误（未补E3未使用LFM的既定runtime policy、Qwen缺省counter_kind）在可复用build_ui_model_pins.py中修正，没有改旧实验或重新请求模型。

基础原始结果现在在answer_live返回及释放调用锁前保存；页面路由/时长信息另存补充记录，不覆盖基础记录。UI重排改用与正式在线相同的已审查loader和完整identity，不能仅加载同名目录默认参数。上述身份与落盘集成13项离线测试通过。

界面每次冷加载，与在线实验预热常驻运行方式不同；当前UI不得直接采用其延迟优势作为自适应默认部署证据。已就绪路由仍可作为实验选项，固定路径保留默认；后续报告必须披露这项运行方式限制。
