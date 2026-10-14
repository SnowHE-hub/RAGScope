"""Build the requested local Markdown LoRA report from verified journals and receipts."""
import argparse
import json
from pathlib import Path
import statistics
import sys
PROJECT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(PROJECT))
from ragscope.experiments.ledger import read_records
from ragscope.indexing.identity import artifact,atomic_json


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign',required=True);a=p.parse_args();run=Path(a.campaign).resolve()
    if run.parent!=Path('D:/RAGScope_Runtime/runs').resolve():raise ValueError('Wrong campaign')
    old=json.loads((run/'LORA_OLD_RECOUNT.json').read_text('utf8'))
    runtime=json.loads((run/'S5_RUNTIME_INDEPENDENT_ANALYSIS.json').read_text('utf8'))
    supervision=json.loads((run/'S5_SUPERVISION_DISTRIBUTION_DIAGNOSIS.json').read_text('utf8'))
    merge=json.loads((run/'S5_MERGE_LOGIT_GATE.json').read_text('utf8'))
    fixed={};sources=[artifact(run/n) for n in ('LORA_OLD_RECOUNT.json','S5_RUNTIME_INDEPENDENT_ANALYSIS.json',
        'S5_SUPERVISION_DISTRIBUTION_DIAGNOSIS.json','S5_MERGE_LOGIT_GATE.json')]
    for c in ('base','unmerged'):
        path=run/('S5_FIXED64_'+c+'.jsonl');rows=[r for r in read_records(path) if r['purpose']=='formal']
        if len(rows)!=3 or any(r['error'] for r in rows):raise ValueError('Fixed token benchmark incomplete/failed')
        fixed[c]={'n':3,'first_token_p50_seconds':statistics.median(r['first_token_seconds'] for r in rows),
            'decode_tps_p50':statistics.median(r['tokens_per_second_after_first'] for r in rows)};sources.append(artifact(path))
    smoke_path=run/'S5_SUPERVISION_SMOKE_v2/COMPLETION.json'
    smoke=json.loads(smoke_path.read_text('utf8')) if smoke_path.exists() else None
    if smoke:sources.append(artifact(smoke_path))
    g=runtime['groups'];b,c=g['base'],g['unmerged'];ob,oc=old['conditions']['baseline'],old['conditions']['adapted']
    lines=['# LoRA 退步与生成成本诊断','',
        '## 已验证的结论','',
        '旧100题对照的答案文字F1下降，不能启用旧adapter。新的24题机制对照确认了输出更长和逐token生成更慢；这批样本没有显示EOS无限续写。合并副本未通过事先固定的数值容差，不能声称合并修复了延迟。人工审核保持 NOT_REVIEWED。','',
        '本报告按任务书要求交付本地Markdown/JSON；用户指定格式及不上传要求覆盖报告技能的默认MCP/HTML/Sites呈现方式。少量精确实验值采用表格，避免把不同样本集拼成趋势图。','',
        '## 旧100题：拒答增加是主要负向贡献之一','',
        f'旧同BF16配对共100题，其中92题可回答。文字F1由 {ob["f1"]:.6f} 降至 {oc["f1"]:.6f}，差值 {old["delta_f1"]:+.6f}。23次新增可答题拒答贡献 −0.097927；28题改善贡献 +0.123935，其余退步和失败共同形成净下降。不能把23次拒答全部认定为错误拒答，必须另核材料充分性。',
        '', '旧独立上下文proxy核验得到充分YES8、不足NO12、不确定UNKNOWN3；这是AI代理判断，不是人工审核或正式事实真值。旧停止原因与逐token时间未完整记录，不能从旧汇总补造。','',
        '## 新24题：输出更长，正常停止，并保留引用失败','',
        'D_confirm按query_id排序预先固定前24题；两分支同BF16权重、上下文、eager后端、greedy及512输出预算。22题可回答。成本只覆盖实际冻结上下文生成，不含在线检索；预热单列，每分支1次。','',
        '|指标|基础模型|未合并adapter|','|---|---:|---:|',
        f'|正式题数|{b["formal_n"]}|{c["formal_n"]}|',
        f'|生成P50秒|{b["natural_generation_p50_seconds"]:.4f}|{c["natural_generation_p50_seconds"]:.4f}|',
        f'|实际输出token中位数|{b["output_tokens_p50"]}|{c["output_tokens_p50"]}|',
        f'|格式/引用失败|{b["failures_n"]}|{c["failures_n"]}|',
        f'|可答题拒答|{b["answerable_refusals_n"]}|{c["answerable_refusals_n"]}|',
        f'|含失败零分的文字F1|{b["word_f1"]:.6f}|{c["word_f1"]:.6f}|','',
        '新的文字F1变化与旧100题不同，基础模型存在8条不可见引用失败；不能据此宣称adapter事实质量更好，也不能删除失败后挑选有利分数。该实验为prompt-only BF16协议，不能直接替代schema约束Q4部署成绩。','',
        '两分支24/24均以实际generation_config的EOS结束；tokenizer EOS为248046、模型EOS为248044，但实际在首个im_end后均仅多2个token。字段差异存在；本次结果反驳其造成大量续写的解释，不能推广为所有输入都无停止问题。','',
        '## 固定64-token：adapter每token开销单独可见','',
        '固定同3个dev输入，每分支3次正式请求加1次预热。强制64-token输出只用于微基准，不能评答案质量。同步streamer排除prompt回调，首个生成token时间包含后端准备；解码速度取后续63个token，包含统一的同步回调开销。','',
        '|指标|基础模型|未合并adapter|','|---|---:|---:|',
        f'|首token P50秒|{fixed["base"]["first_token_p50_seconds"]:.4f}|{fixed["unmerged"]["first_token_p50_seconds"]:.4f}|',
        f'|首token后token/秒中位数|{fixed["base"]["decode_tps_p50"]:.4f}|{fixed["unmerged"]["decode_tps_p50"]:.4f}|','',
        '这支持“未合并adapter具有逐token执行开销”的机制解释；样本仅3个，不是在线P95/SLA，也不将其比例直接套用旧100题。输出长度与逐token耗时共同影响自然停止体验。','',
        '## 监督与分布：没有发现全量mask失效，分布迁移仍是描述性证据','',
        f'972条训练样本的数值mask/EOS/可见引用检查未失败；实际解码20条目标均含完整assistant内容和结束标记。训练prompt中位 {supervision["training_prompt_tokens"]["p50"]} token，旧配对评测中位 {supervision["paired_evaluation_actual_prompt_tokens"]["p50"]}；841条以干净参考证据配对，131条使用真实首段负样本。长度和材料差异不能单独证明退步因果。','',
        '首次8夹具自检仅一步，继承的warmup使首步学习率为0，梯度有效但参数未更新；失败日志和checkpoint保留。兼容分支仍是同8个官方配对夹具，重复两次形成2个optimizer steps，仅用于验证监督链路。',
        ('兼容结果尚未完成。' if smoke is None else f'兼容结果：{smoke["status"]}；实际{smoke["optimizer_steps"]}步，变化张量{smoke["changed_tensor_n"]}，平均监督loss变化{smoke["mean_loss_change"]:+.6f}。完整adapter/optimizer/scheduler/RNG/data位置恢复逐项核验；不是泛化提升。'),'',
        '## 合并容差失败，正式重训未触发','',
        '三条探针使用相同输入比较合并前后最后位置logits；预先固定max_abs≤0.08、relative_L2≤0.01。实际最大绝对差0.50–0.625、相对差0.0332–0.0501；argmax一致但容差失败。未保存合并快照，合并24题和合并微基准保留NOT_RUN_GATE/前置阻断，不放宽阈值追求通过。', '',
        '数值误差可能涉及BF16合并舍入及残差计算路径，但尚未隔离原因，不能称已确认PEFT或模型bug。旧模型与adapter文件未覆盖。','',
        '正式新LoRA为 NOT_RUN_GATE：完整主事实评分仍有资源阻断；材料充分条件下的可训练缺陷及300条部署分布训练样本尚未同时满足。不能根据下降的loss追加训练轮数。','',
        '## 后续验收与未解问题','',
        '继续评估真正不同的工作流，保存每题完整效用向量。最终质量、支持性、引用及在线成本需按冻结候选和holdout协议验证；本报告不能替代S6。主要未解项是自然问答事实评分的可靠性、合并函数数值差异来源以及噪声分布迁移的因果大小。默认保留v2；人工 NOT_REVIEWED。','',
        '## 可复核来源与命令','',
        '重算：`scripts/v3/audit_lora_runtime_diagnostics.py --campaign <campaign>`；本报告：`scripts/v3/report_lora_diagnostics.py --campaign <campaign>`。所有模型调用由预算账本计数，失败和预热保留。','']
    for source in sources:lines.append(f'- `{source["path"]}` SHA256 `{source["sha256"]}`')
    destination=PROJECT/'docs/execution/v3/LORA_DIAGNOSIS.md';destination.write_text('\n'.join(lines)+'\n',encoding='utf8')
    atomic_json(run/'S5_LORA_REPORT_SOURCE_NOTES.json',{'report':artifact(destination),'sources':sources,'fixed_token_summary':fixed,
        'delivery':'USER_REQUIRED_MARKDOWN','audience':'technical','visual_omission_reason':'Exact small-cohort tables; no misleading mixed-cohort trend',
        'formal_retraining':'NOT_RUN_GATE','human_review_status':'NOT_REVIEWED','smoke_compatibility_complete':smoke is not None})
    print(str(destination));return 0


if __name__=='__main__':raise SystemExit(main())
