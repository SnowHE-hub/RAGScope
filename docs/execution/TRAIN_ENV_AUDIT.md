# LoRA 独立训练环境与原始权重准备

日期：2026-10-14。当前范围仅安装、下载与静态核验；不加载模型、不初始化CUDA、不运行GPU训练。原GPU任务由用户决定手动恢复，本子任务不操作恢复。

## 固定输入与兼容性依据

读取 `ragscope/training/run.py` 和 `configs/v2/lora.json`：真实训练入口要求Transformers v5、BF16、GPU lease、仅本地模型、完整权重清单；模型架构必须是 `Qwen3_5ForConditionalGeneration`，LoRA的模块列表必须从实际模型线性层与祖先类型发现。

独立环境目标 `D:/RAGScope_Runtime/envs/train` 创建前确认不存在。基座Python3.12.13位于D盘；不修改core、models或其他项目环境。配置固定torch2.9.1+cu128、Transformers5.19.0、PEFT0.21.2、Accelerate1.15.0。其余直接依赖与传递依赖以安装后完整锁文件为准。

依据 [PyTorch官方历史版本安装表](https://pytorch.org/get-started/previous-versions/#v291)，torch2.9.1具有Windows CUDA12.8官方wheel；[PyTorch Blackwell支持说明](https://pytorch.org/blog/pytorch-2-7/)确认CUDA12.8构建引入Blackwell支持。选择cu128满足RTX5060/sm120构建需求；此处只确认发行渠道与构建版本，实际设备/内核前向反向需后续取得独占GPU时验证。未修改或安装系统驱动。Transformers的 [Qwen3.5官方接口](https://huggingface.co/docs/transformers/v5.19.0/model_doc/qwen3_5) 与 [PEFT官方快速教程](https://huggingface.co/docs/peft/en/quicktour)作为库接口依据。

模型唯一主路径 `Qwen/Qwen3.5-2B`，官方revision `15852e8c16360a2fea060d615a32b45270f8a8fc`，Apache-2.0。下载前官方API快照保存到 `D:/RAGScope_Runtime/downloads/qwen35_2b_bf16_info.json`。不下载0.8B fallback；只有真实2B路径失败后才允许后续执行器考虑它。

官方唯一safetensors分片 `model.safetensors-00001-of-00001.safetensors` 为4,548,221,488字节，LFS SHA-256 `aa33250c4fc64891ddfaba3a314fd9542ea371843c387178b425fbcc5ed680b1`；完整必要根目录快照（权重、索引、配置、tokenizer、模板、许可证与说明）预查询总大小4,571,272,453字节。源仓库没有下载执行任何Python代码。下载使用1MiB流式读取，每文件最多3次尝试；保留失败证据，不改动已有不匹配文件。

模型落盘目标 `D:/RAGScope_Runtime/models/Qwen3.5-2B/15852e8c16360a2fea060d615a32b45270f8a8fc`。下载过程写 `model_manifest.json` 的DOWNLOADING状态，完成后才能标VERIFIED，并包含 `model/revision/artifacts[{path,bytes,sha256}]`、源URL和官方LFS校验值。`config.json`必须匹配训练配置哈希 `ed1c1723241f23f7f4e23430759cbd7dcfb4103cbdfe052bfe7626b57c2615b4`；完成后以stdlib读取safetensors头部，验证实际tensor dtype，不载入tensor。

开始时D盘可用216,644,845,568字节；本项目Runtime逻辑大小16,895,948,195字节。加4,571,272,453字节模型和12GiB环境保守预留，仍低于80GiB项目上限。逻辑大小会重复计算硬链接，作为保守预算，不声称是物理占用。所有缓存/TEMP/基座Python均通过 `scripts/runtime_env.ps1` 指向D盘。

## 执行状态

- CUDA wheel安装：PASS，torch2.9.1+cu128来自官方索引；日志 `D:/RAGScope_Runtime/logs/train_torch_install.log`。安装准备8分02秒、安装6.45秒，没有加载CUDA库。
- BF16快照下载：PASS，12文件共4,571,272,453字节，全部逐文件SHA/大小核验通过；日志 `D:/RAGScope_Runtime/logs/qwen35_2b_bf16_download.log`；模型目录 `model_manifest.json` 为VERIFIED。真实头部596个BF16张量和36个F32张量，保留官方混合精度，未转换或加载张量。
- 38包完整锁文件 `D:/RAGScope_Runtime/envs/train/requirements.lock.txt`，SHA-256 `8f92040cbfcb2469c278f828c9496a206a0a1ad121b9950a1d4460eb51332554`；`uv pip check` PASS。剩余依赖来自官方PyPI，串行下载/安装；解析时修正上游元数据的两条带多余引号版本声明警告，日志保留，没有安装失败或重试。
- 包元数据与静态接口验证：PASS，记录为 `D:/RAGScope_Runtime/envs/train/environment_manifest.json`。以AST读取torch/version.py，确认构建CUDA12.8、git5811a8d7da873dd699ff6687092c225caffcf1bb；以AST确认Qwen3_5ForConditionalGeneration、Attention、GatedDeltaNet、LoraConfig、get_peft_model、set_peft_model_state_dict存在，保存对应源码哈希。没有导入torch/transformers/peft。
- 当前直接依赖实测：Transformers5.19.0、PEFT0.21.2、Accelerate1.15.0、safetensors0.8.0、numpy2.5.4、psutil7.2.2、filelock4.1.1；huggingface-hub2.2.0、tokenizers0.23.3。所有安装与清单读取发生在独立train环境。
- GPU兼容性前向/反向和10步预检：未执行，本次准备阶段明确不加载模型。

主执行器报告大文件安装期间主机可用内存下降，已暂停进一步GPU模型预检；后续依赖安装设置并发下载/安装各1。主执行器已给runtime脚本补充TORCH_EXTENSIONS_DIR/TORCHINDUCTOR_CACHE_DIR，均实测在D盘。最后预算复核与模型清单哈希保存于 `D:/RAGScope_Runtime/envs/train/preparation_receipt.json`。不以安装完成冒充实际训练成功。

下一步由主执行器安排独占资源：数据准备可调用 `D:/RAGScope_Runtime/envs/train/Scripts/python.exe` 与上述本地tokenizer；真正GPU预检必须先验证GPU lease、完成的权重清单、冻结train/dev样本，再调用 `ragscope.training.run --mode preflight`。本子任务没有执行该命令或下载0.8B权重。
### tokenizer准备后的应用依赖补齐

训练代理实际准备样本时发现训练环境缺应用导入依赖，已补装diskcache5.6.3、loguru0.7.3、tiktoken0.14.0及其传递依赖。当前共46包，`uv pip check`再次通过，完整新锁SHA-256为`e2cd66b2d9bb305c554c9e123e556870b663a4820abd84de784e0a24ee67f5e2`。旧38包锁与初始环境清单保存在同目录历史文件；新environment_manifest明确初始CUDA状态字段只描述准备阶段，不能用来推断后来训练进程的实时状态。安装日志在LoRA运行目录，依赖复验为`D:/RAGScope_Runtime/logs/train_dependency_check_after_application.log`。
