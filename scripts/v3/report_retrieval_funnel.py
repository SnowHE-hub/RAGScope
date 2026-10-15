"""Requested retrieval report from actual window, natural rerank and scope mechanism audits."""
import argparse
import json
from pathlib import Path
import sys
PROJECT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(PROJECT))
from ragscope.indexing.identity import artifact,atomic_json


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign',required=True);a=p.parse_args();run=Path(a.campaign).resolve()
    paths=[run/'S3_NATURAL_RERANK_D_search_ANALYSIS.json',run/'S3_NATURAL_RERANK_D_confirm_ANALYSIS.json',run/'S3_HIERARCHY_REAL_LOOKUPS.json',run/'S3_SCOPED_GENERATION_D_confirm_ANALYSIS.json']
    screen,confirm,hierarchy,packing=[json.loads(path.read_text('utf8')) for path in paths]
    lines=['# 检索漏斗与范围处理诊断','',
        '## 自然重排在开发确认集退步，拒绝部署','',
        '同40题、同实际候选池完成BGE max/top2及Qwen官方yes/no排序。BGE在首批打包完整证据覆盖提升6.25pp，优于Qwen的3.125pp，因此只对BGE扩大到D_confirm100。确认集max下降14.13pp、top2下降13.04pp，拒绝该候选；不根据确认结果改聚合公式追分，不追加生成矩阵。','',
        '## 各阶段分母与实际证据存活','',
        '均为已指定论文的scoped hybrid候选，最多50个实际唯一块，不复制凑数。D_search40中32题有适用标注证据；D_confirm100中92题适用。原文坐标计算连续覆盖、完整段落召回及完整参考集合覆盖，gold只在离线审计读取。','',
        '|集合|条件|候选完整集合覆盖|top10完整集合覆盖|2048预算打包覆盖|',
        '|---|---|---:|---:|---:|']
    for name,data in (('D_search40',screen),('D_confirm100',confirm)):
        for condition,g in data['groups'].items():lines.append(f'|{name}|{condition}|{g["candidate_coverage"]["full_reference_coverage"]:.4%}|{g["rank_top10_coverage"]["full_reference_coverage"]:.4%}|{g["packed_coverage"]["full_reference_coverage"]:.4%}|')
    lines += ['', '候选阶段均包含全部适用参考证据，所以这些scoped样本的主要损失在排序位置及预算打包之后。确认集BGE top10只有小幅损失，打包却明显退步，说明前10有证据不等于模型实际看到证据。这不证明全库召回没有问题，也不直接证明某个具体邻块bug。',
        '', 'BGE固定FP32/512窗口、同GPU比较max与top2；Qwen固定BF16、官方yes/no最后token logits及完整输入上限8192。两模型输入容量不同，跨模型结果不能用于隔离架构因果。实际计算BGE首批727窗口对、Qwen463完整对、BGE确认1823窗口对，全部有逐条预算和模型文件hash。','',
        '## 原窗口与core-first：反证和有限改善均保留','',
        '旧500记录/9519窗口审计未发现query-inclusive预算、字符覆盖、降序或ID映射违例。不能声称旧BGE只看首段。旧max/top2复用相同窗口分数没有所选覆盖增益；新的自然集确有个别排序变化，但确认后的打包收益仍为负。',
        '', f'无重排core-first的100题配对文字F1增量 {packing["word_delta"]:+.6f}，论文聚类95%区间 {packing["word_delta_paper_cluster_ci95"]}，包含零。证据覆盖改善不等于事实/引用改善；主要claim指标另行测量，未通过发布门槛。','',
        '## 两级候选与全库旁路已经运行真实索引机制检查','',
        '文档索引包含1585篇真实题名/摘要/章节的BM25和Dense。块索引为实际完整稀疏矩阵；全库块Dense在三个有界尝试后受阻，部分零矩阵不采用。已指定论文跳过文档定位；可识别对象分支按文档BM25/Dense秩融合取top5，各论文最多10块，并保留全库块BM25旁路及唯一ID融合。缺对象返回候选文档和澄清。',
        '', f'已对20个开发问题执行 {hierarchy["lookup_n"]} 次实际完成索引的机制检索，核验scope不泄漏、澄清和旁路。可识别分支使用显式机制探针，不把该布尔条件当原问题真实可识别性，不报告产品准确率。查询向量是先前真实编码的冻结向量，计时不是在线E2E。',
        '', '原始全库E8没有论文对象的输入与追加可见题名的输入是不同任务；题名条件的高文档召回不能冒充原问题提升。最终H_scope专项尚待冻结后执行。','',
        '## 资源替代、发布和未解项','',
        '本轮scoped结果复用了141篇已指定论文的1647个实际向量，是独立的范围内替代，不声称完成全库块Dense。保留所有旧索引、原失败、partial矩阵和缓存，不再重启该受阻操作。',
        '', '自动默认保持v2。自然重排筛选失败；core-first还需主要事实指标、一次holdout和真实在线成本。缺对象澄清和全库旁路是已验证的工程机制，不等于已发布或生产准确率通过。人工 NOT_REVIEWED。','',
        '## 可复核来源','',
        '重排原记录、窗口预算、范围模式和离线覆盖各有独立文件。报告自身不发模型请求，不重跑成功结果。','']
    for path in paths:lines.append(f'- `{path}` SHA256 `{artifact(path)["sha256"]}`')
    target=PROJECT/'docs/execution/v3/RETRIEVAL_FUNNEL.md';target.write_text('\n'.join(lines)+'\n',encoding='utf8')
    atomic_json(run/'S3_RETRIEVAL_REPORT_SOURCE_NOTES.json',{'report':artifact(target),'sources':[artifact(path) for path in paths],
        'audience':'technical','delivery':'USER_REQUIRED_MARKDOWN','human_review_status':'NOT_REVIEWED',
        'visual_omission_reason':'Exact stage/condition lookup table; no mixing of scoped and unresolved-global cohorts'})
    print(str(target));return 0


if __name__=='__main__':raise SystemExit(main())
