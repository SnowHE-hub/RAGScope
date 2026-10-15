"""Record actual reused environments/packages; install or upgrade nothing."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
PROJECT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(PROJECT))
from ragscope.indexing.identity import artifact,atomic_json,canonical_json


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign',required=True);a=p.parse_args();run=Path(a.campaign).resolve()
    runtime=Path('D:/RAGScope_Runtime').resolve()
    if run.parent!=runtime/'runs':raise ValueError('Wrong campaign')
    packages=['numpy','psutil','requests','filelock','tokenizers','transformers','torch','streamlit','peft','trl','bert-score']
    code='import importlib.metadata as m,json,sys; wanted='+repr(packages)+'; versions={d.metadata["Name"].lower():d.version for d in m.distributions()}; print(json.dumps({"python":sys.version,"executable":sys.executable,"packages":{n:versions.get(n.lower()) for n in wanted}}))'
    rows=[]
    for directory in sorted((runtime/'envs').iterdir()):
        python=directory/'Scripts/python.exe'
        if not python.is_file():continue
        result=subprocess.run([str(python),'-c',code],capture_output=True,text=True,encoding='utf8',timeout=30)
        if result.returncode:raise RuntimeError('Environment metadata failed: '+directory.name)
        rows.append({'environment':directory.name,'python_launcher':artifact(python),'observed':json.loads(result.stdout),'actual_exit_code':result.returncode})
    lock={'status':'PASS_OBSERVED_REUSED_ENVIRONMENT_LOCK','environments':rows,'packages_installed_or_upgraded_by_this_command':0,
        'runner':artifact(__file__),'note':'Separate environments intentionally retain different versions. No combined pip-install recipe or v2 upgrade implied.'}
    atomic_json(PROJECT/'configs/v3/dependencies.lock.json',lock);atomic_json(run/'DEPENDENCIES_V3_LOCK.json',lock)
    print(canonical_json({'status':lock['status'],'environments':[r['environment'] for r in rows]}));return 0


if __name__=='__main__':raise SystemExit(main())
