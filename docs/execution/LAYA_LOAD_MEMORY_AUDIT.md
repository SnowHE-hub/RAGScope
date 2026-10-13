# Laya固定加载路径的内存审阅

2026-10-14，Sydney。仅检查已固定SDK源码、已安装依赖源码和本地safetensors的21,544字节头部；没有读取权重数据区、导入Torch/Transformers/Laya、加载模型或下载新文件。默认5 GiB预载门槛和1.25 GiB运行余量保留；主执行器已授权下述唯一条件式4.5–5 GiB备选，实际加载尚未执行。

结论：**当前固定Laya 0.4.6没有额外公开的减少加载副本参数**。`Agent.__init__`与`laya.load`不接受`dtype`、`torch_dtype`、`map_location`、`low_cpu_mem_usage`或`assign`。不能把Transformers另一条加载API的参数直接传给它，也不能将CPU BF16计算开关当作减少常驻权重的方法。

证据为`D:/RAGScope_Runtime/runs/20261014_v2_routing/LAYA_LOAD_MEMORY_SOURCE_AUDIT.json`，包含SDK/安装源/依赖源及配置SHA、完整API参数、张量头统计。复算脚本为`D:/RAGScope_Runtime/downloads/audit_laya_load_memory.py`，日志`logs/laya_load_memory_source_audit.log`，实际退出0。版本为Laya0.4.6、Torch2.9.1+cpu、Transformers4.57.6、safetensors0.8.0。

## 精确存储量与复制路径

| 项目 | 从本地头部/配置导出的字节数 |
|---|---:|
| 官方F16参数：205张量、421,293,827元素 | 842,587,654 |
| `temperature`持久buffer：3个F32元素 | 12 |
| 全部checkpoint张量payload | 842,587,666（0.785 GiB） |
| SDK默认FP32参数及该buffer | 1,685,175,320（1.569 GiB） |
| 复制期间同时存在的张量存储地址空间 | 2,527,762,986（2.354 GiB） |

文件总长842,609,210字节＝8字节长度字段＋21,536字节JSON头＋payload。实际CPU RSS还含依赖导入、Python对象、非持久RoPE buffer、分配器和推理激活；本次没有测量它们。最大单张量是词嵌入51,576,832元素。

固定源码位置：

- `laya/common.py:693-710`：`build_model(..., pretrained=False)`读取本地encoder config，使用`no_init_weights()`加`AutoModel.from_config`；config明确`dtype=float32`，仍分配真实CPU参数空间。
- `common.py:577-606`：head先在meta构造，再`to_empty(device="cpu")`。它已经避免随机初始化及head初始化临时开销，但不省最终FP32存储。源码明确encoder包含不应被`to_empty`清除的非持久RoPE buffer。
- `laya/agent.py:676-684`：先构造模型，再`weights=load_file(...)`，然后`load_state_dict(weights, strict=True)`。兼容性检查只看keys/shapes，未找到整份额外float32 clone。
- 安装的`safetensors/torch.py:328-360`：默认`device="cpu", backend="mmap"`，读取张量映射。Laya已经走该路径。映射页可能被系统回收，因此2.354 GiB是同时存活的张量存储地址空间估计，**不是实际RSS峰值的硬下界**；之前`LAYA_READINESS.json`的`load_peak_lower_bound_bytes`应按这个限定理解，旧记录保持不改。
- 安装的`torch/nn/modules/module.py:2476-2487,2522-2549`：`assign=False`默认保留目标FP32属性，用`param.copy_(input_param)`复制/转换；Laya没有传`assign=True`。改成assign会保留F16源dtype，属于修改官方加载实现和计算精度，不能作为已验证的等价优化。
- `agent.py:312-319,787-790`：`LAYA_CPU_AMP=bf16`只设前向autocast；`agent.py:796-809`的`.to(self.device)`也没有dtype转换。设置该变量不减少上述加载存储量。

官方源码固定为[Agent加载实现](https://github.com/NandhaKishorM/laya/blob/c73908d96ff73c0062c340a11bf7c54ec7d8bc4d/laya/agent.py#L485)及[模型构造](https://github.com/NandhaKishorM/laya/blob/c73908d96ff73c0062c340a11bf7c54ec7d8bc4d/laya/common.py#L693)。安装文件与已审查SDK签收的哈希再次匹配。

## 可执行备选及其限制

本次保留原SDK、checkpoint与默认5 GiB guard。没有发现可以直接开启且保持相同默认FP32计算的官方“低内存加载”选项。ONNX分支需要当前没有的导出权重，且导出仍需先载入模型，不能当成已有可用替代；未为此安装或导出。

主执行器已明确授权：只有正常资源交接、embedding/R3退出、22调用上下文压力诊断退出GPU服务后，才执行原命令。fresh可用RAM≥5 GiB沿用原路径；4.5–5 GiB且没有其他本项目模型常驻时，允许**同一默认SDK在独立CPU进程中做一次有界加载＋既定3题dev warmup预检**；低于4.5 GiB不尝试。备选不改变dtype、权重、输入、候选顺序、算法或超参组合。

```powershell
. ./scripts/runtime_env.ps1
& 'D:/RAGScope_Runtime/envs/models/Scripts/python.exe' scripts/run_laya_v2.py --home D:/RAGScope_Runtime --campaign 20261014_v2 --run
```

这不是已证实的安全内存值，也不是SDK公开优化开关：4.5－2.354－1.25仅留下约0.896 GiB给依赖、分配器和激活。静态张量地址空间不等于RSS。构造函数内部没有合作式暂停点，所以监控不能保证任何两个采样之间的瞬时内存余量。`laya_supervision.py`每0.25秒采样host/RSS，加载阶段上限180秒，运行余量1.25 GiB。

父子进程通过不可变request、随机token、PID、creation time、完整argv、cwd及祖先关系相互验证。子进程必须先取得LOAD许可，加载完再取得WARMUP许可。只有**从未授权任何预热/正式推理**且精确身份仍匹配时，父进程才能有界终止自己的无结果加载进程；从发出WARMUP许可起，永久禁止强停，低内存只写STOP，子进程先保存当前结果再退出。3个既定dev预热全部真实成功后，父进程才发CONTINUE执行1080条。不存在批量终止、原任务停止或自动恢复行为。

`LOAD_FALLBACK_STATE.json`用独占创建消费唯一尝试。失败保留request、child identity、逐次资源采样、加载/错误日志和MONITOR_RESULT，拒绝第二次条件式尝试；不循环下调门槛。Windows同一调用的venv启动器按完整argv与cwd识别，不误当第二个模型；跨进程状态的短暂读写冲突最多重试5次、总等待0.30秒，持续失败仍退出。

功能测试使用微小CPU子进程和显式模拟RAM，不加载任何真实权重，不能作为Laya可行性证据。覆盖两次许可、低内存加载精确停止、预热开始后保存结果再退出、伪身份拒绝、资源门槛、唯一尝试、Windows启动器与有界状态写入。真实Laya依然PENDING，签收凭据为`D:/RAGScope_Runtime/runs/20261014_v2_routing/LAYA_SUPERVISED_FALLBACK_DECISION.json`。
