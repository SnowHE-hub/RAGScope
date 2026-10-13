# 有限 LoRA 实施与状态

当前状态：真实官方tokenizer准备已完成，train接受972/1000、dev接受59/60，完整监督边界审计通过。2B BF16 1024长度的真实GPU预检进程已启动，计划5步保存后精确恢复至10步；**训练完成、真实断点恢复与100题同精度对照尚待运行结果**。此前低RAM保护失败不是CUDA OOM实测。

## 官方核验与冻结

2026-10-14 读取 [Unsloth Qwen3.5 微调说明](https://unsloth.ai/docs/models/qwen3.5/fine-tune)：支持小档 BF16 LoRA，要求当前 Transformers v5；本项目不采用该系列 4-bit QLoRA。上游给出 2B 约 5 GiB、0.8B 约 3 GiB 的参考显存，这不是本机测量，也不保证当前环境能训练。

读取 [Qwen 官方 2B config](https://huggingface.co/Qwen/Qwen3.5-2B/raw/15852e8c16360a2fea060d615a32b45270f8a8fc/config.json)、[Transformers 模型实现](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5/modeling_qwen3_5.py) 与 [PEFT LoRA 文档](https://huggingface.co/docs/peft/main/package_reference/lora)。官方架构是 Qwen3_5ForConditionalGeneration，包含语言和视觉组件，语言侧包含混合 attention。代码会检查实际实例化的模块树，再选择语言 attention 祖先下的二维 Linear；排除视觉、MLP、embedding 和输出头。没有猜测 q_proj 等硬编码目标名。

两条冻结候选最多：

| 模型 | revision | config SHA-256 |
|---|---|---|
| Qwen/Qwen3.5-2B | 15852e8c16360a2fea060d615a32b45270f8a8fc | ed1c1723241f23f7f4e23430759cbd7dcfb4103cbdfe052bfe7626b57c2615b4 |
| Qwen/Qwen3.5-0.8B | 2fc06364715b967f1860aea9cf38778875588b17 | b90b86f35c8e6925ef74ee04d0e758f0a845c83a42089ad82bbaa948de9b4204 |

官方2B BF16完整权重已下载并校验，位于 `D:/RAGScope_Runtime/models/Qwen3.5-2B/15852e8c16360a2fea060d615a32b45270f8a8fc`，完整manifest为VERIFIED；0.8B只保留预先指定的候补路径，没有因结果好坏更换模型。

## 实现约束

`configs/v2/lora.json` 固定 r=8、alpha=16、dropout=0.05、batch=1、accumulation=8、长度1024（允许统一512预检）、gradient checkpointing、lr=1e-4、warmup3%、seed42、最多1epoch/200 optimizer steps。`ragscope/training/run.py` 校验这些边界。

`data.py` 从已冻结 train 文档和合法标注生成最多1000条训练例。无答案题使用真实标注和原论文实际段落，不制造随机假证据。模型自己的 chat template 生成全部 token，prompt 必须是完整序列的精确前缀，只有 assistant 后缀产生标签；超过长度预算整条拒绝并保留原因，不截去答案或必要证据。dev 最多60条仅用于 best adapter 的 loss 选择；test 不进入准备或训练入口。

GPU 训练入口要求 `ACQUIRED`、`owner=ragscope`、`safety_verified=true` 的租约且不是 CPU-only 进程，再取得项目 GPU 文件锁。模型必须已有本地 model_manifest.json，记录官方 model/revision 和所有下载产物哈希；不在训练脚本中联网或下载。

断点在 optimizer step 边界保存：adapter、optimizer、scheduler、Python/NumPy/torch/CUDA RNG、step、epoch、data_order/data_position、配置与训练数据哈希。完整产物经过校验才原子发布 latest；保留 best 和 latest，总共至多3个本轮自有断点，绝不清理原 GPU 任务目录。未完成的临时目录不会被当作成功。此设计参照 [PyTorch 保存与恢复说明](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html)，实际 torch 恢复仍须实测。

预检建议先停在5步，再同配置 `--resume` 到10步，确认损失、有限梯度、数据进度与日志连续；随后才允许正式 train 从该10步断点继续。预检和正式训练必须传入同一份 dev 文件；没有完成10步时正式入口拒绝运行。

`evaluate.py` 对同一已加载 BF16 基础模型切换 adapter，使用完全相同 frozen ContextBundle、tokenizer、prompt、后端与 greedy 输出预算。允许模型实际声明的 FP32 数值稳定性参数，但要求基础参数 dtype 映射在两种模式保持一致。固定100个test问题（不足时事先冻结实际N），所有失败保留。当前该真实对照尚未执行。

## 已执行的软件测试与资源证据

`tests/unit/test_training_v2.py`：10 passed，3.18秒；数据和断点确定性模块行覆盖92%（data91%、checkpoint93%），证据 `D:/RAGScope_Runtime/runs/training_coverage.json`。测试实际验证原子断点 roundtrip、重启读取全部模拟状态、至多3个自有断点、损坏/半写入/无归属目录拒绝、标签边界、split与GPU租约限制。这里的模拟状态测试不是一次真实神经网络训练恢复。

实际资源 guard：

| 运行目录末段 | 配置 | 可用 RAM 字节 | 所需 RAM 字节 | 结果 |
|---|---|---:|---:|---|
| guard_2b_1024 | 2B / 1024 | 1558700032 | 2147483648 | BLOCKED_RESOURCE |
| guard_2b_512 | 2B / 512 | 885649408 | 2147483648 | BLOCKED_RESOURCE |
| guard_08b_512 | 0.8B / 512 | 781205504 | 2147483648 | BLOCKED_RESOURCE |

证据根目录 `D:/RAGScope_Runtime/runs/20261014_v2_lora`。这些是低 RAM 时加载前的真实保护检查，不能宣称已试过 CUDA OOM。用户后续扩大了其他计算任务停止与恢复授权，根代理正在核验；资源条件改变后可继续受控预检，本报告不能永久替代真实可行实验。

可执行入口：

```powershell
. ./scripts/runtime_env.ps1
& 'D:/RAGScope_Runtime/envs/core/Scripts/python.exe' -m ragscope.training.run --mode guard --config configs/v2/lora.json --path-index 0 --max-seq-len 1024 --run-dir D:/RAGScope_Runtime/runs/20261014_v2_lora/guard_recheck
```

真实训练使用后续建立的独立 train 环境，执行 `python -m ragscope.training.prepare --help` 与 `python -m ragscope.training.run --help` 获取精确参数。训练入口不会把已实现的框架标作已完成的实验。

后续数据准备修正：多标注问题使用与选定完整证据集合配对的同一条真实annotation答案，保存annotation_id；无答案题选择实际paragraph，不能只拿标题充当上下文。未匹配或存在未解决证据的annotation整条拒绝。准备产物记录接受/拒绝分母、拒绝原因、tokenizer产物哈希、最大实际长度与监督token数；再次执行只接受完全相同内容，不覆盖不同结果。模型快照必须status=VERIFIED，并核验所有weight_map分片，且这些检查发生在导入Torch之前。

实际数据准备产物：`D:/RAGScope_Runtime/runs/20261014_v2_lora/prepared_2b_1024_v2`。train972接受（841可回答、131不可回答），28完整长度超预算拒绝；dev59接受（53可回答、6不可回答），1超预算拒绝。最大训练长度999，最少assistant监督token27。所有972条实际token数组的prompt掩码、assistant标签与序列长度一致；CPU准备CUDA mask为空且cuda_initialized=false。数据manifest固定官方tokenizer文件哈希、来源SPLIT哈希、prompt版本rag-json-v2.2及examples哈希。

保留初始工程失败：缺少diskcache时进程没有进入准备；补齐隔离env应用依赖后，Transformers5.19默认返回dict导致旧prepared_2b_1024的1000条被类型守卫拒绝。显式return_dict=False修复后用新目录生成最终样本。实际探针证明USE_TORCH=0并未阻止该版本导入Torch，因此最终报告使用cuda_initialized与实际设备mask区分CPU准备和GPU训练，不宣称该步骤未导入Torch。

GPU预检日志：`.../20261014_v2_lora/logs/preflight_2b_1024_stop5.log`，训练run固定`.../20261014_v2_lora/train_2b_1024`。预检后正式训练受972样本的一轮122 optimizer steps上限约束（全局上限200），同一配置、同一dev、同一数据顺序，从10步断点继续。恢复时adapter、optimizer、scheduler、全部RNG、数据顺序和进度将逐项真实核验，输出resume_from_step_*.json；尚未产生时不得宣称恢复已验证。
