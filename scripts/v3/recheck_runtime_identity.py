"""Final read-only checksum recheck of every original model and five environments."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sys
PROJECT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(PROJECT));sys.path.insert(0,str(Path(__file__).parent))
from audit_identities import artifacts_in
from ragscope.indexing.identity import artifact,atomic_json,canonical_json


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign',required=True);a=p.parse_args();run=Path(a.campaign).resolve()
    runtime=Path('D:/RAGScope_Runtime').resolve()
    if run.parent!=runtime/'runs':raise ValueError('Wrong campaign')
    source=run/'S0_IDENTITY_E8_AUDIT.json';old=json.loads(source.read_text('utf8'))
    refs=[{k:r[k] for k in ('path','bytes','sha256')} for r in old['verified_artifacts'] if r['status']=='FOUND']
    refs.extend(artifacts_in(old['environments']));cache={};errors=[];checked=0;total=0
    for expected in refs:
        path=Path(expected['path']).resolve()
        if not path.is_relative_to(runtime) and not path.is_relative_to(PROJECT):raise ValueError('Protected artifact outside known project/runtime')
        if not path.is_file():errors.append({'path':str(path),'status':'MISSING'});continue
        stat=path.stat();key=(stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns)
        if key not in cache:
            digest=hashlib.sha256()
            with path.open('rb') as f:
                for chunk in iter(lambda:f.read(1024*1024),b''):digest.update(chunk)
            cache[key]=digest.hexdigest();total+=stat.st_size
        checked+=1
        if stat.st_size!=expected['bytes'] or cache[key]!=expected['sha256']:errors.append({'path':str(path),'status':'HASH_MISMATCH'})
        if checked%20==0:print(canonical_json({'event':'IDENTITY_RECHECK','checked':checked,'unique_bytes_hashed':total}),flush=True)
    result={'status':'PASS_ORIGINAL_RUNTIME_IDENTITIES_PRESERVED' if not errors else 'FAIL_ORIGINAL_RUNTIME_IDENTITY',
        'expected_artifacts':len(refs),'checked_artifacts':checked,'original_model_refs':len(old['verified_artifacts']),
        'original_environments':[e['name'] for e in old['environments']], 'unique_bytes_hashed':total,'errors':errors,
        'source':artifact(source),'runner':artifact(__file__),'observed_utc':datetime.now(timezone.utc).isoformat(),
        'models_loaded':False,'new_llm_requests':0,'files_modified_by_recheck':False}
    atomic_json(run/'FINAL_RUNTIME_IDENTITY.json',result);print(canonical_json(result));return 0 if not errors else 3


if __name__=='__main__':raise SystemExit(main())
