"""Requested protocol report from the completed864-row controlled study and42 supplements."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics
import sys
PROJECT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(PROJECT))
from ragscope.indexing.identity import artifact,atomic_json


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign',required=True);a=p.parse_args();run=Path(a.campaign).resolve()
    if run.parent!=Path('D:/RAGScope_Runtime/runs').resolve():raise ValueError('Wrong campaign')
    study=json.loads((run/'S2_PROTOCOL_ANALYSIS.json').read_text('utf8'))
    extra=json.loads((run/'S2_SUPPLEMENT_ANALYSIS.json').read_text('utf8'))
    if study['status']!='PASS_FROZEN864_WORD_PROTOCOL_AUDIT' or extra['status']!='PASS_INDEPENDENT_SUPPLEMENT_AUDIT':
        raise ValueError('Protocol evidence incomplete')
    lines=['# 生成协议与模型规模诊断','',
        '## 没有发现直接调用与pipeline的通道或token错配；协议影响依模型而异','',
        '已完成72次direct/pipeline冒烟、864条固定问题协议记录和42条补充记录。更简洁的提示对4B/9B文字代理有帮助，对2B没有一致帮助；去除schema增加了输出/引用失败。结果不支持统一更换协议，也不证明9B事实能力必然弱于2B。主要事实评分仍有资源阻断。','',
        '## 样本、分母与控制条件','',
        f'固定24题，每类不可答/yes-no/抽取/自由回答各6题；每题使用干净Oracle和冻结真实检索两种材料。G0复用{study["reused_G0_n"]}条真实结果，新主矩阵{study["new_matrix_n"]}条，总计{study["matrix_n"]}条。G3有3个种子，因此组n=72是24题的重复测量，不能当作72个独立问题。',
        '', 'G0保留原schema与greedy；G1仅加强最小充分回答；G2在G1基础上取消schema；G3在G1基础上用固定官方模型卡文本配置及三个预先种子。全部保留失败分母；可答题文字F1、不可答拒答分开，不能把文字相似度称事实正确率。','',
        f'逐条审计的metric mismatch={study["metric_mismatch_n"]}、实际prompt token mismatch={study["actual_prompt_token_mismatch_n"]}。旧已见题仅作诊断，不能作为新holdout发布证明。','',
        '## 协议对照的完整组结果','',
        '|模型|材料|协议|记录数|可答F1|失败|可答拒答|输出token中位数|',
        '|---|---|---|---:|---:|---:|---:|---:|']
    for g in study['groups']:
        lines.append(f'|{g["model_id"]}|{g["context_kind"]}|{g["protocol"]}|{g["n"]}|{g["legacy_word_f1"]:.6f}|{g["failed_n"]}|{g["answerable_abstained_n"]}|{g["output_tokens_p50"]}|')
    lines += ['', '同一材料下，提示词改变是受控实验变量；不同材料下Oracle和检索同时有证据范围、长度和噪声差异，不能只归因于召回。主协议结果只说明本机固定Q4后端及模板下的表现，不是跨架构能力排名。','',
        '## 干净Oracle与等长干扰仍需分开解释','',
        f'补充实际{extra["padding_calls"]}条Oracle padded生成，覆盖12个可答问题×3模型；证据逐字保留，干扰来自同论文的非标注证据段。最大长度相对差{extra["length_max_relative_difference"]:.4%}；{extra["retrieval_evidence_position_undefined_n"]}个原检索不含证据的位置无法复刻，使用预先声明的中间位置并保留标记。','',
        '|模型|padded−clean可答文字F1|题数|','|---|---:|---:|']
    pairs=defaultdict(list)
    for row in extra['paired_results']:pairs[row['model_id']].append(row['oracle_padded']-row['oracle_audited'])
    for name,values in pairs.items():lines.append(f'|{name}|{statistics.mean(values):+.6f}|{len(values)}|')
    lines += ['', '这些结果说明上下文组织是混杂因素；只有12题，标注之外的段落可能仍有相关信息，未经过操作者人工相关性审核。不能由padded变化推断一般噪声鲁棒性。','',
        '## No Context的任务限制与已有知识探针','',
        '原strict-grounded任务要求只依据材料回答，空材料下拒答不能直接解释为“模型没有知识”。schema允许空引用，但不强制abstain=true；限制主要来自任务提示。补充允许已有知识的H2O/一周天数探针，共6次三模型实际回答均命中字面答案，只证明这两个简单问题，不代表广泛知识成绩。','',
        '## 尚未解决的根因和选型结果','',
        '直接/pipeline的72次冒烟中36对输入和输出一致，未观察到token或thinking通道错配。该反证只覆盖固定夹具及模板；没有足够依据声称所有9B场景都不存在包装问题。',
        '', 'G2的失败保留，不以“能解析出JSON外壳”冒充合法引用或可靠事实答案。官方采样配置不保证本机更好，不能以哪个种子最高来挑选。未新增9B BF16全矩阵，也不将未运行的精度桥接写成已验证。',
        '', 'D_confirm上的core-first配对100题已另完成；文字F1增量约+0.01235、论文聚类95%区间跨零，主要claim/support/citation代理未知。当前没有可据此发布的统一新生成协议，保留v2默认。后续仍需任务工作流和一次冻结终验，人工 NOT_REVIEWED。','',
        '## 可复核来源','',
        '报告只读取已完成审计，无新模型调用。若重新运行报告，不重跑主矩阵或补充模型请求。','']
    sources=[artifact(run/'S2_PROTOCOL_ANALYSIS.json'),artifact(run/'S2_SUPPLEMENT_ANALYSIS.json')]
    for source in sources:lines.append(f'- `{source["path"]}` SHA256 `{source["sha256"]}`')
    target=PROJECT/'docs/execution/v3/GENERATION_PROTOCOL_DIAGNOSIS.md';target.write_text('\n'.join(lines)+'\n',encoding='utf8')
    atomic_json(run/'S2_PROTOCOL_REPORT_SOURCE_NOTES.json',{'report':artifact(target),'sources':sources,'delivery':'USER_REQUIRED_MARKDOWN',
        'audience':'technical','human_review_status':'NOT_REVIEWED','visual_omission_reason':'Exact model/context/protocol audit table, no mixed-cohort trend chart'})
    print(str(target));return 0


if __name__=='__main__':raise SystemExit(main())
