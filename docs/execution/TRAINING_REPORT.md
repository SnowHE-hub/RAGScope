# 有限 LoRA 实施与真实结果

官方 Qwen3.5-2B BF16 LoRA 已实际完成 **1 epoch、122 optimizer steps、972 条 train 样本**。正式 seed42 run 的损失与梯度全部有限，5→10 和10→正式训练的完整状态恢复逐项精确核验通过。进程正常退出、GPU锁释放，nvidia-smi观测为0/8151MiB。**100题同精度基础模型/adapter对照仍等待最终E2/R2上下文，因此不能宣称问答质量提升，也尚未决定是否启用adapter。**

最终审计：D:/RAGScope_Runtime/runs/20261014_v2_lora/train_2b_1024_seed42/TRAINING_AUDIT.json。
GPU释放证据：同目录GPU_RELEASE_RECEIPT.json。
数据审计：D:/RAGScope_Runtime/runs/20261014_v2_lora/prepared_2b_1024_v2/MASK_AND_SPLIT_AUDIT.json。

## 冻结条件与实际训练

官方模型 Qwen/Qwen3.5-2B revision：15852e8c16360a2fea060d615a32b45270f8a8fc。完整12文件快照在D盘，经所有权重、配置、tokenizer哈希校验后才加载；禁止下载器不完整receipt进入GPU加载。config SHA为ed1c1723241f23f7f4e23430759cbd7dcfb4103cbdfe052bfe7626b57c2615b4。

配置固定r8、alpha16、dropout0.05、batch1、梯度累积8、BF16、最大长度1024、gradient checkpointing、lr1e-4、warmup3%、seed42、最多1epoch/200步。实际972条样本对应122步。没有使用512回退或0.8B候补；没有新增训练超参组合。

实际模型模块检查选出114个语言attention祖先下的二维Linear，共3,691,008个LoRA可训练参数。排除视觉、MLP、embedding和输出头，所有基础参数冻结。没有猜测q_proj等硬编码模块列表。独立环境为Torch2.9.1+cu128、Transformers5.19.0、PEFT0.21.2、Accelerate1.15.0。缺少可选快速内核时使用上游参考PyTorch路径，未安装未验证替代内核。

[官方微调说明](https://unsloth.ai/docs/models/qwen3.5/fine-tune)、[官方模型配置](https://huggingface.co/Qwen/Qwen3.5-2B/raw/15852e8c16360a2fea060d615a32b45270f8a8fc/config.json)及[PEFT LoRA文档](https://huggingface.co/docs/peft/main/package_reference/lora)用于实现核验；它们的参考显存不是本机测量。

## 真实样本与完整监督

| 集合 | 原始数 | 接受 | 可回答/不可回答 | 完整长度超预算拒绝 |
|---|---:|---:|---:|---:|
| train | 1000 | 972 | 841/131 | 28 |
| dev | 60 | 59 | 53/6 | 1 |

train涉及394篇论文，dev31篇，无论文交叉；test不进入准备或训练。最大实际训练序列999tokens，dev825，最少assistant监督27tokens。使用官方chat template；prompt为完整序列的精确前缀且labels全部设为-100，assistant后缀保留真实监督。必要证据或答案放不下时整条拒绝，不截断或虚造证据。多标注问题使用同一条annotation的答案与完整证据集合；无答案例也来自真实标注和实际论文段落。

train examples SHA：3a015dc3121732ef266d6450fd6a75666b4ce1c55d11a97d81d53c32fb3ca405。
dev examples SHA：a1606aecc83c9b9c0286d4c173d0d165707ab0444b174fff4a361db3f7651592。

## 训练观察与dev选择

| optimizer step | 59条dev平均loss |
|---:|---:|
| 25 | 0.1581860474 |
| 50 | 0.1428376289 |
| 75 | 0.1394188490 |
| 100 | 0.1372079927 |
| 122 | 0.1356631101 |

按预先规定的dev loss选择step122为best；latest也是step122。保留本轮自有完整断点120/121/122，共3个。Adapter文件14,798,456bytes，SHA a532dc22942d84e4dceabacb64b936c191b37ebdfa9fda092f3f87464d0bbae4。loss下降是训练/验证观察，不能替代独立test问答指标。

所有122条optimizer-step日志有配置hash与逐条record_hash，序号无缺失/重复。实际每步耗时中位数9.942秒，step计算时间合计1215.023秒；这些数字不包含模型加载、dev评价、保存断点与进程间暂停，不能当总墙钟时间。

## 实际恢复核验与历史失败

正式seed42训练在模型/PEFT构造前固定Python/NumPy/Torch/CUDA随机种子。三个独立初始化进程得到同一LoRA参数初始SHA：c730a4fa702c2a45717a7d810f472fecd2e90a78f4b4f76b7eb4ca93cf5d685f。

正式run先5步保存退出，再新进程恢复5→10，第三个进程恢复10→122。两个resume receipt实际验证adapter张量、optimizer、scheduler、全部RNG、data_order和data_position完全一致。每一步在optimizer边界原子发布完整adapter+训练状态+tokenizer+progress+哈希manifest；只清理本轮已有归属标记的多余断点，没有触碰用户原任务断点。可用scripts/audit_lora_v2.py --run <正式目录>重新核验，审计本身不加载Torch或模型。

首轮train_2b_1024完成真实10步，但发现PEFT初始化早于seed设置。经root授权，只修复该可复现性缺口，重做同模型同参数seed42正式run；原10步及证据保留为诊断，未混入正式loss曲线。

其他保留的工程失败：首次准备缺diskcache，隔离依赖补齐后恢复；Transformers5.19的apply_chat_template默认返回dict，初始prepared_2b_1024的1000条被守卫全部拒绝，显式return_dict=False修复后使用新_v2目录，不覆盖拒绝记录。实际CPU准备探针显示该版本即使USE_TORCH=0仍导入Torch，但CUDA mask为空且cuda_initialized=false；报告没有宣称未导入Torch。

更早的资源检查：2B1024可用RAM1,558,700,032bytes、2B512为885,649,408bytes、0.8B512为781,205,504bytes，均未达到2GiB加载前门槛。这些是preload资源保护失败，不是已运行的CUDA OOM；后续资源释放后2B1024真实训练成功。

## 实际资源与释放

正式run跨三个进程的最大Torch allocator allocated为7,608,193,024bytes，reserved为8,283,750,400bytes；最高采样进程RSS2,556,239,872bytes，最低采样主机可用内存1,947,979,776bytes。采样发生在microbatch之后，可能漏过短暂峰值。allocator reserved不等于物理显存，Windows共享/分页可能参与，不能据此推断精确CPU offload层数。

诊断首轮默认缓存reserved曾达到14.48GB，同次nvidia-smi单点为7888/8151MiB。后续运行明确使用empty_unused_after_each_microbatch_and_dev_example，仅释放未使用allocator缓存，数学超参未改变。策略hash为3f3dab11c896cc7b7d6d5f2eb1bc786cf035f8770a0ea7b4de60ca5692f872fe。主机RAM低于512MiB时会在完整checkpoint边界恢复性退出，本轮没有触发。

exec72907最终exit0；2026-10-13T16:37:59Z确认训练进程不存在，gpu.lock实际重新获取/释放成功，GPU0MiB/0%，主机可用4,041,805,824bytes。已通知检索代理恢复本项目CPU embedding。用户原任务按最新要求保持USER_MANUAL_RESUME，没有自动恢复。

## 待完成的同精度对照

scripts/evaluate_lora_v2.py使用最终E2/R2固定100题，同一BF16基础权重、tokenizer、prompt、上下文、后端、greedy512输出预算，只切换adapter。每种条件每次进程先做3条单列warmup，全部调用进入全局预算。两组的prompt-only JSON协议相同；它们与Q4主矩阵的schema约束后端不同，不能做不控制协议的精度归因。

完整输入冻结和test推理尚未执行；失败会保留，OOM一次或连续后端失败三次停机，不生成虚假的完整对照结果。完成后才能报告F1、EM、ROUGE-L、拒答与引用指标的变化。

最新离线训练与在线路由回归22项通过；早期数据/checkpoint确定性模块覆盖92%。这些软件测试与上面的真实122步证据明确区分。
