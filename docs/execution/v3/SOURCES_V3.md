# v3来源与环境身份

模型、tokenizer、后端、数据的原始路径和固定SHA见运行目录SOURCE_INVENTORY.json、S0_IDENTITY_E8_AUDIT.json。最终重新逐字哈希398项：110个原模型引用及原5环境的相关包元数据，差异0；独立硬链接去重读取约19.29GB，未加载模型、未修改文件。证据：FINAL_RUNTIME_IDENTITY.json。

实际环境与关键包版本见configs/v3/dependencies.lock.json。原5环境core、core_acceptance_20261014、models、reports、train保持；本轮既存v3_metrics/v3_metrics_cpu隔离环境也单列，不能把7环境合成一份升级旧环境的pip命令。

实跑主生成器是原Qwen3.5-2B Q4_K_M，后端私有Ollama0.34.4。旧4B/9B用于有限协议和裁判诊断；旧2B BF16与adapter用于同精度LoRA诊断。没有新增4B/9B正式训练，也没有把BF16成绩冒充Q4部署成绩。

编码器复用Qwen3-Embedding-0.6B Q8_0，1024维、固定科学问题指令、L2归一化、4096输入不截断。GPU桥接与原CPU身份分开。唯一新增重排对照Qwen3-Reranker-0.6B固定revision及官方yes/no logits方法；BGE原窗口模型保留。模型卡和权重manifest记录于D盘downloads/models，不采用可变tag作为身份。

NLI采用固定MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli及真实id2label、FP32窗口规则；只称proxy。BERTScore官方默认因资源受阻，实际完成的兼容编码器、层、配置和完整5720答案分母见语义评分记录；两者不混成一个已验证事实裁判。

QASPER来自allenai/qasper固定数据revision；新H_QA论文与开发/训练互斥，gold仍封存。H_scope和H_tasks是规则/AI来源任务，非官方QA、人审或操作者盲测。部分任务源引用验证失败，全部失败分支保留。

研究依据及边界完整列于MASTER_PLAN.md第12节。核心来源：

- [QASPER官方数据](https://huggingface.co/datasets/allenai/qasper)与[官方基线/evaluator](https://github.com/allenai/qasper-led-baseline)
- [BERTScore原论文](https://arxiv.org/abs/1904.09675)
- [RAGChecker原论文](https://arxiv.org/abs/2408.08067)
- [ALCE引用评估](https://aclanthology.org/2023.emnlp-main.398/)
- [Sufficient Context](https://arxiv.org/abs/2411.06037)
- [Adaptive-RAG](https://aclanthology.org/2024.naacl-long.389/)
- [RAFT](https://arxiv.org/abs/2403.10131)
- [Qwen官方模型卡](https://huggingface.co/Qwen/Qwen3.5-2B)

这些来源帮助提出实验，不保证本机提升；没有声称完整复现RAGChecker、FActScore或RAFT训练配方。没有付费API、上传或push。
