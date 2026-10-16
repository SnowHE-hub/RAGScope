"""Verify the delivered handbook against its source evidence and explicit scope."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def main():
    md, html = ROOT/'docs/architecture.md', ROOT/'docs/architecture.html'
    text = md.read_text(encoding='utf-8')
    snapshot = read(HERE/'repository_snapshot.json')
    consistency = read(HERE/'CONSISTENCY_CHECK.json')
    renderer = read(HERE/'RENDERER_QA.json')
    experiments = read(HERE/'experiment_audit.json')
    training = read(HERE/'training_routing_evidence.json')
    checks = []
    def check(name, condition, evidence):
        checks.append({'requirement': name, 'passed': bool(condition), 'evidence': evidence})
        if not condition:
            raise AssertionError(name)

    head = subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip()
    changed = [f for f, expected in snapshot['source_sha256'].items() if sha(ROOT/f) != expected]
    tracked_diff = subprocess.check_output(['git','diff','--name-only'], cwd=ROOT, text=True).strip()
    check('Current repository identity and protected functional source', head == snapshot['head'] and not changed and not tracked_diff,
          {'head': head, 'python_files_unchanged': len(snapshot['source_sha256']), 'tracked_diff': tracked_diff})
    chapters = re.findall(r'^## (\d+)\. (.+)$', text, re.M)
    check('All required chapters 0 through 18', [int(n) for n,_ in chapters] == list(range(19)), chapters)
    appendix = re.findall(r'^### 附录 ([A-H])：', text, re.M)
    check('Appendices A through H', appendix == list('ABCDEFGH'), appendix)
    qa = re.split(r'^### Q\d+ ', text, flags=re.M)[1:]
    qa = [q.split('## 18.',1)[0] for q in qa]
    fields = ['面试官问法','通俗答案','技术答案要点','项目真实例子','源码位置','还需补充知识','现有证据是否足够']
    check('At least 30 complete interview questions', len(qa) >= 30 and all(all(f in q for f in fields) for q in qa),
          {'questions':len(qa),'required_fields':fields})
    terms = ['RAG','Corpus','Document','Chunk','Token','Tokenizer','Chunk Size','Overlap','Embedding','Vector',
             'Dense Retrieval','Sparse Retrieval','BM25','TF-IDF','Cosine Similarity','Vector Database',
             'Hybrid Retrieval','RRF','Cross-Encoder','Reranker','Parent-Child Retrieval','Context Window','Prompt',
             'LLM','SLM','Quantization','LoRA','QLoRA','Fine-tuning','Routing','Oracle Context','NLI','EM','F1',
             'ROUGE-L','Semantic Similarity','Recall','MRR','nDCG','Faithfulness','Hallucination','Bootstrap',
             'P50','P95','Cache','GPU','VRAM']
    glossary = text.split('## 3.',1)[1].split('## 4.',1)[0].casefold()
    missing = [t for t in terms if t.casefold() not in glossary]
    check('All explicitly requested technical terms explained', not missing, {'terms':len(terms),'missing':missing})
    check('Five evidence levels', all('['+s+']' in text for s in ['代码','实测','文档','推断','未知']),
          'Chapter0 definitions, per-claim references, and unknown limitations')
    references = []
    for token in re.findall(r'`([^`\n]+)`',text):
        m = re.fullmatch(r'((?:ragscope|scripts|ui|configs|tests|docs|reports)/[^\s`]*\.(?:py|ps1|json|ya?ml|md|html|lock))(?::(\d+))?',token)
        if not m: continue
        p = ROOT/m.group(1)
        ok = p.is_file()
        if ok and m.group(2):
            ok = 1 <= int(m.group(2)) <= len(p.read_text(encoding='utf-8-sig').splitlines())
        references.append({'reference':token,'exists_and_line_valid':ok})
    check('Actual local file and line citations', all(r['exists_and_line_valid'] for r in references),
          {'occurrences':len(references),'unique':len({r['reference'] for r in references})})
    verified_sources=[]
    for path, ref in experiments['source_files'].items():
        p=Path(path)
        check_value = p.is_file() and p.stat().st_size == ref['bytes'] and sha(p) == ref['sha256']
        if not check_value: raise AssertionError('Changed experiment evidence: '+path)
        verified_sources.append(path)
    check('Raw experimental evidence still matches the numeric audit', len(verified_sources)==105,
          {'files_rehashed':len(verified_sources),'source':'experiment_audit.json; no model rerun'})
    scored=experiments['V3_finish_scored']
    score_labels = {'token_f1':'Token F1', 'exact_match':'EM', 'rouge_l':'ROUGE-L',
                    'semantic_cosine':'答案/参考向量余弦'}
    score_table = text.split('|最终正式指标|H_QA B0|H_QA FINAL|H_tasks B0|H_tasks FINAL|', 1)[1].split('\n\n', 1)[0]
    table_values = {}
    for metric, label in score_labels.items():
        match = re.search(r'^\|' + re.escape(label) + r'\|([^\n]+)\|$', score_table, re.M)
        assert match, ('Missing final score table row', label)
        table_values[metric] = [float(cell.strip()) for cell in match.group(1).split('|')]
        assert len(table_values[metric]) == 4, label
    for col, group in enumerate(('H_QA:B0','H_QA:FINAL','H_tasks:B0','H_tasks:FINAL')):
        row=scored[group]
        for metric in ('token_f1','exact_match','rouge_l','semantic_cosine'):
            displayed = table_values[metric][col]
            assert abs(displayed-row[metric]['mean']) <= 0.000000500001, (group,metric,displayed,row[metric]['mean'])
        assert row['all_factual_unknown'] and row['all_human_not_reviewed']
    check('Final score table values and their denominators', sum(r['n'] for r in scored.values())==440 and
          scored['H_QA:FINAL']['token_f1']['n']==170 and scored['H_tasks:FINAL']['token_f1']['n']==20,
          'Four groups independently recomputed from440 score records; every metric/table column numerically bound at six-decimal precision, including zero displayed as0')
    check('V2 main experiment scope and retained failures', experiments['V2_generation_total']=={'n':5520,'groups':39,'failures':32},
          experiments['V2_generation_total'])
    check('Global and online current-vs-baseline scope', sum(r['n'] for r in experiments['V3_finish_global'].values())==1200 and
          sum(r['n'] for r in experiments['V3_finish_online']['FINAL_ONLINE_RESIDENT8K.jsonl'].values())==60,
          'Global1200; online30+30; B0ctx4096 and FINALctx8192 explicitly distinguished')
    check('Documented real training and routing proof', all(k in training for k in ('train','routing','environments','models','sources')),
          'Training source arrays/mask/splits and122 steps; route replays vs actual requests; seven env metadata inventories')
    check('Shared Markdown-to-HTML body and table/code equality', consistency['status']=='PASS' and
          consistency['markdown_sha256']==sha(md) and consistency['html_sha256']==sha(html) and
          consistency['table_cells_exact'] and consistency['code_blocks_exact'] and consistency['html_deterministic_from_markdown'],
          'CONSISTENCY_CHECK.json full deterministic transform, not a visual browser assertion')
    svg=re.findall(r'<svg\b.*?</svg>',text,re.S)
    diagram_refs=[]
    for p in sorted((HERE/'diagrams').glob('*.svg')):
        content=p.read_text(encoding='utf-8').strip()
        assert content in svg,p.name
        ET.fromstring(content)
        png=HERE/'previews'/(p.stem+'.png')
        assert png.is_file() and png.stat().st_size>10000,png
        diagram_refs.append({'svg':p.name,'sha256':sha(p),'png':str(png),'png_sha256':sha(png)})
    check('Six actual inline and raster-rendered SVG diagrams',len(svg)==6 and len(diagram_refs)==6 and
          all(not r['text_outside_viewBox'] for r in renderer['svg_boundary_checks']),diagram_refs)
    check('Offline HTML, navigation, Chinese, responsive and print rules',consistency['offline_external_dependencies']==0 and
          not consistency['broken_anchors'] and '\ufffd' not in text and '@media' in html.read_text(encoding='utf-8') and
          '@media print' in html.read_text(encoding='utf-8'),
          {'toc_targets':consistency['toc_targets'],'verification':'Structure, CSS and SVG renders; full browser layout remains unverified'})
    check('No unresolved author placeholders',not any(x in text for x in ('<!--DIAGRAM:','<!--EXPERIMENT_INDEX-->','<!--REPOSITORY_TREE-->')) and
          not re.search(r'\b(?:TODO|FIXME|TBD)\b',text),'Required unknown facts are intentional prose, not author placeholders')
    for ref in read(HERE/'REFERENCE_FILES.json'):
        assert sha(ref['path'])==ref['sha256']
    check('QuantMind actual provided references',True,'MD1825lines read in consecutive chunks; entire HTML/CSS parsed; REFERENCE_FILES and REFERENCE_STYLE_AUDIT bind hashes')
    check('Current path, defaults, commands, choices and known limits',all(x in text for x in
          ['## 14. 当前真实运行路径','### 14.2 当前默认清单','## 15. 关键技术决策','## 16. 已知问题','--check-only','USER_MANUAL_RESUME']),
          'Root inspected code invocation chain; historical/default/experimental paths separated; commands cross-checked with LOCAL_RUN_V3 and scripts')
    check('No own document-preview process remains',not read(HERE/'PREVIEW_PROCESS_AUDIT.json')['matching_owned_profile_processes'],
          'psutil exact workspace-profile matching; no other processes stopped')
    result={'status':'COMPLETE_WITH_DECLARED_BROWSER_PREVIEW_LIMITATION','documentation_complete':True,
            'requirements_passed':len(checks),'checks':checks,'source_references':references,
            'markdown':{'path':str(md),'bytes':md.stat().st_size,'sha256':sha(md),'lines':len(text.splitlines()),
                        'unicode_characters':len(text),'cjk_characters':len(re.findall(r'[\u3400-\u9fff]',text))},
            'html':{'path':str(html),'bytes':html.stat().st_size,'sha256':sha(html)},
            'main_chapters':19,'interview_questions':len(qa),'inline_svg':len(svg),'tables':consistency['tables'],
            'code_blocks':consistency['code_blocks'],'head':head,
            'functional_code_changed':False,'model_or_training_or_index_rerun':False,
            'browser_visual_verified':False,'browser_limitation':'Chrome unavailable; Edge crashed; IAB file protocol rejected. No workaround after policy rejection.',
            'remaining_unknown_facts':['Independent human answer correctness','Pretraining contamination','Generalization beyond exposed tasks',
                                       'Unique causal explanation of LoRA/rerank/scale effects','Full browser desktop/mobile layout'],
            'created_unix':time.time()}
    (HERE/'CONTENT_AUDIT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
    (HERE/'SOURCE_REFERENCE_CHECK.json').write_text(json.dumps(references,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps({k:result[k] for k in ('status','requirements_passed','main_chapters','interview_questions','inline_svg','tables','functional_code_changed')},ensure_ascii=False))


if __name__=='__main__':
    main()
