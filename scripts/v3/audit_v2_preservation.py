"""Check every file in the frozen v2 commit without modifying its checkout."""
import argparse
from datetime import datetime,timezone
import os
from pathlib import Path
import subprocess
import sys
PROJECT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(PROJECT))
from ragscope.indexing.identity import atomic_json,canonical_json


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign',required=True);a=p.parse_args();run=Path(a.campaign).resolve()
    if run.parent!=Path('D:/RAGScope_Runtime/runs').resolve():raise ValueError('Wrong campaign')
    base='d431f646a91502c2c75f1e991521f4ec0c519a1d'
    items=subprocess.check_output(['git','ls-tree','-r','-z',base],cwd=PROJECT).split(b'\0')
    count=0;line_endings=[];differences=[]
    for item in items:
        if not item:continue
        meta,name=item.split(b'\t',1);relative=os.fsdecode(name)
        original=subprocess.check_output(['git','cat-file','blob',meta.split()[2].decode()],cwd=PROJECT)
        path=PROJECT/relative;current=path.read_bytes() if path.is_file() else None;count+=1
        if current!=original:
            if current is not None and current.replace(b'\r\n',b'\n')==original.replace(b'\r\n',b'\n'):line_endings.append(relative)
            else:differences.append(relative)
    result={'status':'PASS_V2_SOURCE_PRESERVED' if not differences else 'FAIL_V2_SOURCE_CHANGED',
        'baseline_commit':base,'tracked_files':count,'content_differences':differences,'line_ending_only_files':line_endings,
        'observed_utc':datetime.now(timezone.utc).isoformat(),'verification_scope':'All baseline tracked files; model and environment artifacts have separate S0 identity audits'}
    destination=run/('V2_PRESERVATION_RECHECK_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')+'.json')
    atomic_json(destination,result);print(canonical_json(result));return 0 if not differences else 2


if __name__=='__main__':raise SystemExit(main())
