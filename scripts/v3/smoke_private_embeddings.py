"""Real fixed embedding relevance probes and backend GPU identity audit."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
PROJECT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(PROJECT))
from ragscope.experiments.runtime import configure
from ragscope.indexing.identity import artifact,atomic_json,canonical_json
from ragscope.indexing.local_embedding import QUERY_INSTRUCTION
from ragscope.v3.local_runtime import ManagedOllama
from ragscope.v3.private_embeddings import PrivateEmbeddingClient


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign',required=True);a=p.parse_args()
    runtime=Path(configure(cpu_only=True)['home']).resolve();run=Path(a.campaign).resolve()
    if run.parent!=runtime/'runs':raise ValueError('Wrong campaign')
    fixtures=[('Which planet is known as the Red Planet?','Mars is known as the Red Planet.','Water is H2O.'),
        ('What dataset was used?','The experiment used dataset Delta.','The conference was held in May.'),
        ('What is the learning rate?','The learning rate is 0.001.','The paper describes English data.'),
        ('How many samples were used?','The training set had 128 samples.','The authors reported accuracy.')]
    with ManagedOllama(run,runtime/'tools/ollama-0.34.4/ollama.exe',num_ctx=4096) as service:
        client=PrivateEmbeddingClient(run,service)
        texts=[text for q,pos,neg in fixtures for text in (QUERY_INSTRUCTION+q,pos,neg)]
        vectors=client.embed(texts)
        cases=[{'query':q,'positive':pos,'negative':neg,'positive_similarity':float(vectors[i*3]@vectors[i*3+1]),
                'negative_similarity':float(vectors[i*3]@vectors[i*3+2]),
                'correct':bool(vectors[i*3]@vectors[i*3+1]>vectors[i*3]@vectors[i*3+2])} for i,(q,pos,neg) in enumerate(fixtures)]
        repeat=client.embed(texts)
        ps=service.session.get(service.base_url+'/api/ps',timeout=10).json()
        models=ps.get('models',[])
        gpu_resident=bool(models) and all(x.get('size_vram',0)>0 for x in models)
        result={'status':'PASS_CONSTRUCTED_EMBEDDING_SMOKE' if all(c['correct'] for c in cases) and gpu_resident else 'FAIL_EMBEDDING_SMOKE',
            'cases':cases,'shape':list(vectors.shape),'repeat_cache_identical':bool(np.array_equal(vectors,repeat)),
            'model_identity':client.identity,'backend_loaded_models':ps,'gpu_resident_observed':gpu_resident,
            'human_review_status':'NOT_REVIEWED','new_generation_requests':0,'source':artifact(__file__),
            'natural_retrieval_quality':'NOT_YET_EVALUATED','backend_comparability':'new GPU Ollama identity; not the old CPU llama backend'}
        atomic_json(run/'S3_PRIVATE_EMBEDDING_SMOKE.json',result)
        print(canonical_json({k:result[k] for k in ('status','cases','shape','repeat_cache_identical','gpu_resident_observed')}))
    return 0 if result['status'].startswith('PASS') else 2


if __name__=='__main__':raise SystemExit(main())
