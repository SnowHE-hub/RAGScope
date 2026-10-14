# v3 决策记录

2026-10-15T01:45:06.552349+11:00：完整任务书/预算/诊断已核SHA并保存。纯审计CPU-only四线程，新campaign独立；旧v2与用户文件只读保护。先回原journal重算，后评分校准和有界实验；H01–H12均是待验证假设。S6保留1800请求，所有失败/预热/裁判计账。

2026-10-14T15:42:56.373763+00:00：实际CPU原journal审计5520/39/32，独立F1/EM全部一致；官方v0.3 evaluator重算5720条（含LoRA200），实际F1差异0。LoRA互斥分解：新增拒答23题贡献-0.0979273，改善28题+0.1239351，仍回答变差23题-0.0514381，格式/请求3题-0.0127749，不变15题，另8不可答；重构净差-0.0382051。EOS tokenizer248046与model.config248044不同仅为待验证线索，20数值mask检查PASS不能替代实际监督自检。
旧全局题真实文件为queries_global_test.jsonl，原SOURCE_INVENTORY错误候选名缺失保留并在身份补充报告解析；E8 Hybrid100全部可答/完整证据19，适用分母确为100。模型110声明文件全部HASH匹配，硬链接去重实际核19292924146字节；5环境读取未升级。首次身份审计把项目内stopwords来源误当模型缓存越界，原日志保留，一次修复后exit0。
封存新D_confirm100/50论文、H_QA200/100论文，扫描230旧journals保守排除510论文，包括本轮schema读取露出的all_queries第一条所在论文。H_scope/H_tasks仍待构造，未称S0全部完成。裁判60组构造在实际判分前发现购买不排除出售的歧义，V1保留，另立V2明确否定；120自然盲评上下文companion已完成，人工NOT_REVIEWED。
语义环境v3_metrics复用train/reports只读site-packages，仅在新env安装依赖；pip分发bert-score0.3.13但包内部__version__0.3.12，记录两者，不默改身份。GPU初次守卫误将Codex宿主ChatGPT.exe当未知，保留拒绝日志，核PID11276/create1791922058.5791745及WindowsApps/OpenAI.Codex准确路径后修复。第二次GPU已算64/6342对，RAM2850594816低于3GiB守卫而exit1/释放GPU到12MiB；不放宽门槛、不清理用户进程。唯一兼容替代为v3_metrics_cpu复用models/reports CPU torch2.9.1/TF4.57.6，独立cpu命名结果完整6342对，不混入GPUpartial。当前live exec31866；总新生成请求0，S6额度1800完整保留。
