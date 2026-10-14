"""Build four bounded real sparse variants from three completed chunkers."""
import argparse
import json
from pathlib import Path
import sys
PROJECT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(PROJECT))
from filelock import FileLock
from ragscope.experiments.ledger import read_records
from ragscope.experiments.runtime import configure,require_ram,require_disk
from ragscope.indexing.identity import artifact,atomic_json,canonical_json,signature
from ragscope.v3.compressed_index import CompressedBM25

VARIANTS={'R512_body':('R512','text'),'R512_metadata':('R512','retrieval_text'),
          'F512_O0_metadata':('F512_O0','retrieval_text'),'P256_1024_metadata':('P256_1024','retrieval_text')}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign',required=True);p.add_argument('--resume',action='store_true');a=p.parse_args()
    runtime=Path(configure(cpu_only=True)['home']).resolve();run=Path(a.campaign).resolve()
    if run.parent!=runtime/'runs':raise ValueError('Wrong campaign')
    require_ram(2*1024**3);require_disk(600*1024**2)
    source_path=run/'S3_RETRIEVAL_SOURCE_PREPARATION.json';source=json.loads(source_path.read_text('utf8'))
    if source['status']!='PASS_SOURCE_PREPARATION':raise ValueError('Source preparation incomplete')
    journal=source['journal']
    if artifact(journal['path'])!=journal:raise ValueError('Source build journal changed')
    records=read_records(Path(journal['path']))
    with FileLock(str(run/'chunk_sparse_build.lock'),timeout=0):
        outputs={}
        for variant,(group,field) in VARIANTS.items():
            require_ram(2*1024**3)
            selected=[x for x in records if x['group']==group]
            identity={'source':artifact(source_path),'group':group,'retrieval_text_field':field,
                'chunk_artifacts':[x['output'] for x in selected],
                'sparse_parameters':{'k1':1.5,'b':.75,'epsilon':.25,'tokenizer':'alnum-cjk-v1'},
                'compressed_implementation':artifact(PROJECT/'ragscope/v3/compressed_index.py'),'runner':artifact(__file__)}
            directory=runtime/'indexes'/run.name/'chunk_indices'/variant/signature(identity);directory.mkdir(parents=True,exist_ok=True)
            manifest_path=directory/'SPARSE_MANIFEST.json'
            if manifest_path.exists():
                saved=json.loads(manifest_path.read_text('utf8'))
                if not a.resume or saved['identity']!=identity or any(artifact(x['path'])!=x for x in saved['artifacts']):raise ValueError('Completed sparse index identity changed')
                outputs[variant]=artifact(manifest_path);continue
            chunk_path=directory/'chunks.jsonl'
            if chunk_path.exists():raise ValueError('Unreceipted corpus retained; reconcile instead of overwriting')
            texts=[];count=0
            with chunk_path.open('x',encoding='utf8',newline='\n') as f:
                for r in selected:
                    if artifact(r['output']['path'])!=r['output']:raise ValueError('Canonical source artifact changed')
                    doc=json.loads(Path(r['output']['path']).read_text('utf8'))
                    for chunk in doc['chunks']:
                        texts.append(chunk[field]);f.write(canonical_json(chunk)+'\n');count+=1
            index=CompressedBM25.build(texts)
            index_outputs=index.save(directory)+[artifact(chunk_path)]
            del index,texts
            result={'status':'sparse_complete','identity':identity,'signature':signature(identity),'count':count,
                'paper_n':len(selected),'artifacts':index_outputs,'dense_status':'NOT_BUILT','new_generation_requests':0}
            atomic_json(manifest_path,result);outputs[variant]=artifact(manifest_path)
            print(canonical_json({'event':'CHUNK_SPARSE_BUILT','variant':variant,'chunk_n':count,'papers':len(selected)}),flush=True)
        atomic_json(run/'S3_CHUNK_SPARSE_INDICES.json',{'status':'PASS_FOUR_REAL_SPARSE_VARIANTS','variants':outputs,
            'candidate_design':'one no-prefix R512 control; metadata prefixes on three reused chunkers; no full Cartesian experiment grid',
            'dense_status':'NOT_BUILT','new_generation_requests':0})
    return 0


if __name__=='__main__':raise SystemExit(main())
