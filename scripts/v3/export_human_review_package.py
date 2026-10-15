"""Anonymized real paired development answers; export never changes NOT_REVIEWED."""
import argparse
import json
from pathlib import Path
import sys
PROJECT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(PROJECT))
from ragscope.experiments.ledger import read_records
from ragscope.indexing.identity import artifact,atomic_json,canonical_json,signature
from ragscope.v3.audit import reference_texts
from ragscope.v3.splits import read_split


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign',required=True);a=p.parse_args();run=Path(a.campaign).resolve()
    if run.parent!=Path('D:/RAGScope_Runtime/runs').resolve():raise ValueError('Wrong campaign')
    complete=json.loads((run/'S3_SCOPED_GENERATION_D_confirm_COMPLETION.json').read_text('utf8'))
    if artifact(complete['records']['path'])!=complete['records']:raise ValueError('Paired original answers changed')
    rows=read_records(Path(complete['records']['path']));labels={q['query_id']:q for q in read_split(run,'D_confirm',purpose='development')}
    manifest=json.loads((run/'S3_SCOPED_CONTEXT_MANIFEST.json').read_text('utf8'))
    inputs={e['input_id']:e for e in map(json.loads,Path(manifest['inputs']['path']).read_text('utf8').splitlines()) if e['dataset_role']=='D_confirm'}
    grouped={qid:sorted([r for r in rows if r['question_id']==qid],key=lambda r:signature([42,'blind_review',qid,r['condition']])) for qid in labels}
    package=[];mapping=[];markdown=['# 人工审核包：真实配对答案，配置名称隐藏','',
        '状态：NOT_REVIEWED。100个开发确认题，非新holdout；每题A/B顺序在固定seed下匿名化。导出和任何AI判分均不是人工审核。',
        '', '请核对关键事实、数字/否定/条件、实际材料是否充分、引用是否支持答案，并记录UNKNOWN。参考答案可能不完整或有标注分歧；原文支持的额外正确内容不要直接判错。',
        '', '同一科学方法名称若出现在问题、答案或原文中予以保留；隐藏的是被比较的生成配置/模型身份。原文片段保持源坐标，不改写答案。','']
    for i,qid in enumerate(sorted(grouped),1):
        pair=grouped[qid]
        if len(pair)!=2:raise ValueError('Missing paired answer')
        q=labels[qid];case_id=f'REVIEW_{i:04d}';variants=[];private={}
        markdown.extend([f'## {case_id}','',q['question'],'','参考：'+json.dumps(reference_texts(q),ensure_ascii=False),''])
        for alias,r in zip(('A','B'),pair):
            context=inputs[r['input_id']]['context_bundle']['pieces'];parsed=r['parsed_response']
            variants.append({'alias':alias,'answer':parsed['answer'] if parsed else None,'citations':parsed['citations'] if parsed else None,
                'abstain':parsed['abstain'] if parsed else None,'failure':r['error_type'],
                'context':[{'citation_id':p['chunk_id'],'doc_id':p['doc_id'],'canonical_start':p['canonical_start'],
                    'canonical_end':p['canonical_end'],'text':p['text']} for p in context]})
            private[alias]={'condition':r['condition'],'source_record_hash':r['record_hash']}
            markdown.extend([f'### 答案{alias}','',parsed['answer'] if parsed else 'UNKNOWN：无合法解析答案。',
                '',f'拒答：{parsed["abstain"] if parsed else "UNKNOWN"}；引用：{parsed["citations"] if parsed else "UNKNOWN"}','',f'实际提供材料{alias}：',''])
            for piece in context:markdown.extend([f'`{piece["chunk_id"]}` / {piece["canonical_start"]}–{piece["canonical_end"]}','',piece['text'],''])
        package.append({'case_id':case_id,'question':q['question'],'user_scope':q['scope'],'reference_answers':reference_texts(q),
            'reference_disagreement':q.get('annotation_disagreements'),'variants':variants,'dataset_role':'DEVELOPMENT_D_CONFIRM',
            'human_review_status':'NOT_REVIEWED','reviewer_verdict':None,'reviewer_notes':None})
        mapping.append({'case_id':case_id,'query_id':qid,'private_mapping':private})
    outputs=[]
    for path,body in ((run/'HUMAN_REVIEW_PACKAGE.jsonl',''.join(canonical_json(x)+'\n' for x in package)),
        (run/'HUMAN_REVIEW_PRIVATE_MAPPING.jsonl',''.join(canonical_json(x)+'\n' for x in mapping)),
        (PROJECT/'docs/execution/v3/HUMAN_REVIEW_PACKAGE.md','\n'.join(markdown)+'\n')):
        if path.exists() and path.read_text('utf8')!=body:raise ValueError('Preserve previous review package')
        if not path.exists():path.write_text(body,encoding='utf8')
        outputs.append(artifact(path))
    result={'status':'PASS_ANONYMIZED_100_REAL_PAIRS_EXPORTED','case_n':100,'answers_n':200,'outputs':outputs,
        'human_review_status':'NOT_REVIEWED','reviewer_blinding':'configuration/model labels hidden; original scientific entity names retained',
        'source_generation':complete['records'],'holdout_gold_read':False,'new_model_requests':0}
    atomic_json(run/'HUMAN_REVIEW_PACKAGE_RECEIPT.json',result);print(canonical_json(result));return 0


if __name__=='__main__':raise SystemExit(main())
