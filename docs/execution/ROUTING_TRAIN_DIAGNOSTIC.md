# 固定路由模型 train/dev 诊断补充

这是 ROUTING_REAL_RESULTS.md 写入后的补充：使用同一已冻结权重、合法 RoutingRequest 特征、seed42 和 CPU4，对三个模型各实际预测固定 train200。没有重训、修改超参、新增LLM调用或根据test调整模型；dev沿用原冻结60条，并与原最佳checkpoint报告的分类accuracy核对一致。600条输出、全部标签和失败分母保留。

|模型|train accuracy|dev accuracy|train−dev|train预测 BM25/Hybrid/Rerank|dev预测|
|---|---:|---:|---:|---|---|
|linear|0.675000|0.516667|0.158333|191/9/0|60/0/0|
|MLP|0.640000|0.516667|0.123333|200/0/0|60/0/0|
|LSTM|0.640000|0.516667|0.123333|200/0/0|60/0/0|

分类accuracy指匹配“文字质量与成本代理标签”的比例，不是回答F1。train的多数类比例为128/200=64%，dev为31/60=51.667%；MLP/LSTM的差距恰等于该类别占比变化，不能仅凭gap宣称过拟合。线性模型训练集比多数类基线高3.5个百分点，但dev未超过多数类基线，提示有限训练收益未泛化；不将这种小样本观察推广为统计结论。MLP/LSTM原dev损失恶化和早停轨迹仍见原训练报告。

三个实际进程全部exit0并确认退出；峰值RSS约167MB/169MB/214MB，最小host可用内存分别3.736GB/3.749GB/3.700GB。预载总门槛3GiB、运行余量1.25GiB未降低，未操作GPU。初次线性预检误用了带默认1GiB reserve的helper，要求总4GiB而提前退出1；原日志保留，随后显式reserve0恢复既定总3GiB门槛，首轮没有模型加载或预测。

运行目录：`D:/RAGScope_Runtime/runs/20261014_v2_routing/train_diagnostics`。三个 `*_train200.jsonl` 可重建表中train结果；`*_summary.json`绑定原权重、标签、协议、dev决策、真实进程及资源；`ACCEPTANCE.json`独立核600行hash/计数/accuracy和原来源SHA。
