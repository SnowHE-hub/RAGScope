# 比较与总结：独立来源关键点补充诊断

## 两条总结有改善线索，一条比较退步；不足以训练或发布

对开发集6篇论文发出6次4B整篇教师请求，严格逐字引用只接受3个任务：2总结、1比较。原17失败分支及零生成成本段落兼容后的9个失败任务均保留，兼容没有增加合格数。没有填充虚构gold，也没有读取H_QA/H_tasks封存参考。

合格3题以同2B Q4/G0、2048参考预算、512输出预算生成B0与任务动作各3条，共6正式及1独立预热。总结动作选原文完整段落并覆盖至多4个目标章节；比较动作从公开问题提取两方法，各做实际hybrid子检索。引用仍指向原文坐标。

## 关键点的证据与回答必须分开

|任务|条件|文字F1|关键点完整进入上下文比例|回答蕴含关键点proxy下界–上界|
|---|---|---:|---:|---|
|summarize / c11aec|baseline|0.469453|33.3333%|0.0000–0.0000；UNKNOWN 0/6|
|summarize / c11aec|task-action|0.474453|50.0000%|0.0000–0.3333；UNKNOWN 2/6|
|summarize / 479fb5|baseline|0.348148|0.0000%|0.0000–0.1667；UNKNOWN 1/6|
|summarize / 479fb5|task-action|0.669091|100.0000%|0.1667–0.5000；UNKNOWN 2/6|
|compare / 5be7f3|baseline|0.336000|60.0000%|0.0000–0.4000；UNKNOWN 2/5|
|compare / 5be7f3|task-action|0.307692|0.0000%|0.0000–0.2000；UNKNOWN 1/5|

总结中一题文字F1由约0.348升至0.669，关键点上下文覆盖由0升至1；另一题只小幅变化。比较题文字F1约0.336降至0.308，关键点进入上下文比例由0.6降至0。不能仅凭三题均值推断通用工作流收益，不能在看结果后修改选择策略再冒充独立验证。

关键点是原文完整句/段，不是人工审核的原子事实gold；长引用中的多个细节未必都应出现在简洁回答中。回答蕴含proxy采用固定NLI阈值0.8，未知保留上下界；它不是人评，也不把NOT_ENTAILED称事实错误。原参考改写的事实性仍NOT_HUMAN_REVIEWED。

## 实际评分、失败与兼容边界

3组实际4B切分保持全部原答案字符，切分失败0；独立NLI实际完成132关系、61395 encoder tokens。同固定已工程校准的模型/FP32/512窗口/阈值，不更换裁判追分。

首次逐关系守卫重复预留模型加载2GiB，在已加载裁判后受阻；原日志保留。新小队列只做一次resident兼容：加载前≥3GiB，关系有界输入64MiB额外预留和1GiB余量。原全量旧评分仍为有界受阻，没有借此重启旧操作。

既有210构造校准只证明工程链路，不证明自然科学问答裁判准确率。所有UNKNOWN、引用的answer-level关联限制及合法但未覆盖参考的额外陈述保留。

## 阶段门槛与后续

分类训练继续NOT_RUN_GATE：此前60题映射没超过最佳固定策略；本补充仅2/1类样本且主要指标存在不确定性。类别覆盖未达足以推广的规模，列为源标签质量/覆盖限制，不用无效或未审标签扩训练集。

本报告补充真实来源任务与动作验证，不替代封存H_tasks、H_scope和H_QA终验。发布保持v2默认；human_review_status=NOT_REVIEWED。

## 复核来源

- `D:\RAGScope_Runtime\runs\v3_20261015_014506\S4_DEVELOPMENT_TASK_METRICS_COMPLETION_resident_v2.json` SHA256 `8dfd3df6c711b4ce4db785efa4edaba5268cfb07aff555f6a014e33bc0a42a8d`
- `D:\RAGScope_Runtime\runs\v3_20261015_014506\S4_DEVELOPMENT_TASK_SCORES_resident_v2.jsonl` SHA256 `eacde215ade2ca8bc96b041401e08427b610a00713c9cd05b9f75ac0ddeeecaa`
- `D:\RAGScope_Runtime\runs\v3_20261015_014506\S4_DEVELOPMENT_TASK_NLI_RELATIONS_resident_v2.jsonl` SHA256 `25bb668deb1a9c21613fb0a2c1229048286c06391669780e4b76ed9eadd90bbc`

