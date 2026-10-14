"""Preserve the v2 scene and initialize a separately identified v3 campaign."""
from __future__ import annotations
import argparse,hashlib,json,os,subprocess,sys,time,zipfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

PROJECT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(PROJECT))
from ragscope.indexing.identity import artifact,atomic_json
from ragscope.experiments.runtime import configure,require_ram

def git(*args, binary=False):
    result=subprocess.run(['git',*args],cwd=PROJECT,capture_output=True,check=True)
    return result.stdout if binary else result.stdout.decode('utf-8').strip()

def preserve_write(path,content,history):
    if path.exists():
        old=path.read_bytes()
        if old==content:return
        history.mkdir(parents=True,exist_ok=True)
        archive=history/(path.name+'.'+hashlib.sha256(old).hexdigest())
        if not archive.exists():archive.write_bytes(old)
        elif archive.read_bytes()!=old:raise ValueError('Historical hash path collision')
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes(content)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign');parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();paths=configure(cpu_only=True);require_ram(2*1024**3)
    home=Path(paths['home']);docs=PROJECT/'docs/execution/v3'
    pointer=docs/'STATUS.json'
    if args.resume and pointer.exists():
        state=json.loads(pointer.read_text('utf-8'));run=Path(state['campaign_path'])
        if not run.resolve().is_relative_to(home/'runs'):raise ValueError('Campaign pointer outside runtime')
        print(json.dumps({'status':'EXISTING_CAMPAIGN_REUSED','campaign':str(run)}));return 0
    local=datetime.now(ZoneInfo('Australia/Sydney'))
    name=args.campaign or 'v3_'+local.strftime('%Y%m%d_%H%M%S')
    if not name.startswith('v3_') or not name.replace('_','').isalnum():raise ValueError('Unsafe campaign name')
    run=home/'runs'/name
    if run.exists():raise FileExistsError('Existing campaign is never overwritten; use --resume')
    if git('rev-parse','--show-toplevel').replace('\\','/')!=str(PROJECT).replace('\\','/'):
        raise ValueError('Project identity mismatch')
    for path in ['ragscope/schema.py','scripts/build_indexes.py','reports/v2/FINAL_REPORT.md']:
        if not (PROJECT/path).is_file():raise FileNotFoundError(path)
    original_head=git('rev-parse','HEAD');original_branch=git('branch','--show-current')
    status=git('-c','core.quotepath=false','status','--porcelain=v1','-z',binary=True)
    dirty=git('diff','--binary',binary=True);staged=git('diff','--cached','--binary',binary=True)
    plan=PROJECT/'RAGScope_v3_Goal_Plan.md';diagnosis=PROJECT/'RAGScope_v2_Diagnosis.md'
    pack=PROJECT/'RAGScope_v3_Improvement_Pack.zip'
    prefix='RAGScope_v3_Improvement_Pack/'
    with zipfile.ZipFile(pack) as source:
        checks=json.loads(source.read(prefix+'PACK_CHECKSUMS.json'))
        entries={}
        for item in checks:
            filename=item['name']
            if Path(filename).name!=filename:raise ValueError('Archive paths must be simple approved filenames')
            data=source.read(prefix+filename)
            if len(data)!=item['bytes'] or hashlib.sha256(data).hexdigest()!=item['sha256']:
                raise ValueError('User package checksum mismatch:'+filename)
            entries[filename]=data
        if entries[plan.name]!=plan.read_bytes() or entries[diagnosis.name]!=diagnosis.read_bytes():
            raise ValueError('Loose plan/diagnosis differs from package; preserve and resolve first')
        entries['PACK_CHECKSUMS.json']=source.read(prefix+'PACK_CHECKSUMS.json')
    import yaml
    budget=yaml.safe_load(entries['EXPERIMENT_BUDGET.yaml'])
    assert budget['budget']['new_llm_requests_total']==6000
    assert budget['budget']['new_llm_output_tokens_total']==4000000
    assert sum(budget['budget']['stage_requests'].values())==6000
    assert budget['budget']['stage_requests']['S6_final_validation']==1800
    run.mkdir(parents=True);(run/'source_package').mkdir();(run/'baseline').mkdir()
    for filename,data in entries.items():(run/'source_package'/filename).write_bytes(data)
    preserve_write(docs/'MASTER_PLAN.md',entries[plan.name],docs/'history')
    preserve_write(docs/'DIAGNOSIS_INPUT.md',entries[diagnosis.name],docs/'history')
    preserve_write(docs/'EXPERIMENT_BUDGET.yaml',entries['EXPERIMENT_BUDGET.yaml'],docs/'history')
    (run/'baseline/git_status_before.bin').write_bytes(status)
    (run/'baseline/user_dirty_before.patch').write_bytes(dirty)
    (run/'baseline/user_staged_before.patch').write_bytes(staged)
    tracked=git('ls-files','-z',binary=True).decode('utf-8').strip('\0').split('\0')
    source_paths=[p for p in tracked if p.startswith(('ragscope/','scripts/','ui/','tests/','configs/','docs/','reports/')) or p.startswith(('requirements','pyproject','README','AGENTS'))]
    refs=[]
    with zipfile.ZipFile(run/'baseline/v2_small_tracked_sources.zip','x',zipfile.ZIP_DEFLATED) as archive:
        for relative in source_paths:
            path=PROJECT/relative
            if not path.is_file() or path.is_symlink():continue
            refs.append(artifact(path))
            if path.stat().st_size<=2*1024**2:archive.write(path,relative)
    atomic_json(run/'BASELINE_PRESERVATION.json',{'original_head':original_head,'original_branch':original_branch,
        'original_status_file':artifact(run/'baseline/git_status_before.bin'),'dirty_patch':artifact(run/'baseline/user_dirty_before.patch'),
        'staged_patch':artifact(run/'baseline/user_staged_before.patch'),'tracked_sources':refs,
        'source_archive':artifact(run/'baseline/v2_small_tracked_sources.zip'),
        'own_new_file_before_capture':'scripts/v3/bootstrap_campaign.py','user_untracked_files_unchanged_and_not_archived':True})
    target='codex/ragscope-v3-diagnostics'
    exists=subprocess.run(['git','show-ref','--verify','--quiet','refs/heads/'+target],cwd=PROJECT).returncode==0
    if exists:
        if git('rev-parse',target)!=original_head:raise ValueError('Existing v3 branch diverges; preserved scene, no reset')
        git('switch',target)
    else:git('switch','-c',target)
    import psutil,shutil
    gpu=subprocess.run(['nvidia-smi','--query-gpu=index,name,memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True)
    atomic_json(run/'ENVIRONMENT_AT_START.json',{'utc':datetime.now(ZoneInfo('UTC')).isoformat(),'local':local.isoformat(),
        'timezone':'Australia/Sydney','python':sys.executable,'python_version':sys.version,'paths':paths,
        'cuda_mask':os.environ.get('CUDA_VISIBLE_DEVICES'),'available_host_bytes':psutil.virtual_memory().available,
        'host_boot_unix':psutil.boot_time(),'disk':shutil.disk_usage(home)._asdict(),'nvidia_smi_readonly':gpu.stdout.strip(),
        'gpu_models_loaded_by_v3':0,'original_jobs_policy':'USER_MANUAL_RESUME','system_or_wsl_restart':False})
    atomic_json(run/'CAMPAIGN_IDENTITY.json',{'campaign_id':name,'project':str(PROJECT),'runtime':str(home),'branch':target,
        'source_head':original_head,'plan':artifact(docs/'MASTER_PLAN.md'),'budget_spec':artifact(docs/'EXPERIMENT_BUDGET.yaml'),
        'diagnosis':artifact(docs/'DIAGNOSIS_INPUT.md'),'package':artifact(pack),'bootstrap_source':artifact(__file__),
        'created_local':local.isoformat(),'human_review_status':'NOT_REVIEWED'})
    event={'event':'INITIALIZED','unix':time.time(),'requests_reserved':0,'output_tokens_charged':0,'output_tokens_reserved':0,
        'request_limit':6000,'output_token_limit':4000000,'rerank_pair_limit':200000,
        'stage_request_limits':budget['budget']['stage_requests'],'protected_final_requests':1800,'model_calls':0}
    (run/'BUDGET.jsonl').write_text(json.dumps(event,sort_keys=True)+'\n',encoding='utf-8')
    status_value={'campaign_id':name,'campaign_path':str(run),'updated_local':local.isoformat(),'current_stage':'S0',
        'workflow_finished':False,'execution_complete':False,'performance_status':'INCONCLUSIVE','release_status':'KEEP_V2_DEFAULT',
        'human_review_status':'NOT_REVIEWED','stages':{s:('RUNNING' if s=='S0' else 'PENDING') for s in ['S0','S1','S2','S3','S4','S5','S6']},
        'new_llm_requests':0,'next_command':'CPU-only scripts/v3/audit_sources.py --campaign '+str(run)+' --resume',
        'next_command_status':'IMPLEMENT_NEXT; not yet an executable completion claim','active_model_processes':[]}
    atomic_json(run/'STATUS.json',status_value);atomic_json(pointer,status_value)
    decision='# v3 决策记录\n\n'+local.isoformat()+'：完整任务书/预算/诊断已核SHA并保存。纯审计CPU-only四线程，新campaign独立；旧v2与用户文件只读保护。先回原journal重算，后评分校准和有界实验；H01–H12均是待验证假设。S6保留1800请求，所有失败/预热/裁判计账。\n'
    recovery='# v3 恢复记录\n\n当前campaign：`'+str(run)+'`。优先读取STATUS.json、CAMPAIGN_IDENTITY.json和BUDGET.jsonl。此时没有模型/服务活进程，没有新LLM请求，不重复bootstrap；准确下一动作是实现并运行audit_sources的CPU-only只读原journal审计。原训练USER_MANUAL_RESUME，不自动恢复或启动WSL/定时任务。v2回退HEAD：`'+original_head+'`，未执行reset。\n'
    preserve_write(docs/'DECISIONS.md',decision.encode(),docs/'history')
    preserve_write(docs/'RECOVERY.md',recovery.encode(),docs/'history')
    (run/'DECISIONS.md').write_text(decision,encoding='utf-8');(run/'RECOVERY.md').write_text(recovery,encoding='utf-8')
    print(json.dumps({'status':'INITIALIZED_NO_MODEL_CALLS','campaign':str(run),'branch':target,'tracked_sources_preserved':len(refs),'plan':artifact(docs/'MASTER_PLAN.md')},ensure_ascii=False))
    return 0

if __name__=='__main__':raise SystemExit(main())
