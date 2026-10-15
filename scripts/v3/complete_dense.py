"""Complete full-corpus vectors, reusing verified cache; flush before checkpoint."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
from ragscope.experiments.runtime import configure
from ragscope.indexing.identity import artifact, atomic_json, signature


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--batch', type=int, default=16)
    args = parser.parse_args()
    runtime = Path(configure(cpu_only=True)['home'])
    import numpy as np
    import psutil
    from filelock import FileLock
    from ragscope.indexing.array_index import ArrayCache
    from ragscope.indexing.local_embedding import HFJsonTokenizer
    from ragscope.v3.local_runtime import ManagedOllama
    from ragscope.v3.embedding_service_guard import EmbeddingServiceGuard
    from ragscope.v3.private_embeddings import register_existing_weight, TAG

    source, output = Path(args.source).resolve(), Path(args.output).resolve()
    if output.parent != runtime / 'runs' or source.parent != runtime / 'runs':
        raise ValueError('Campaign directories must remain on runtime drive')
    output.mkdir(parents=True, exist_ok=True)
    pointer = json.loads((source/'S3_CHUNK_SPARSE_INDICES.json').read_text('utf8'))['variants']['R512_body']
    if artifact(pointer['path']) != pointer:
        raise ValueError('Sparse source changed')
    sparse = json.loads(Path(pointer['path']).read_text('utf8'))
    directory = Path(pointer['path']).parent
    with (directory/'chunks.jsonl').open(encoding='utf8') as handle:
        chunks = [json.loads(line) for line in handle]
    identity = json.loads((source/'S3_PRIVATE_EMBEDDING_SMOKE.json').read_text('utf8'))['model_identity']
    cache = ArrayCache(source/'embedding_cache', identity)
    field = sparse['identity']['retrieval_text_field']
    config = {'source': pointer, 'embedding': identity, 'count': len(chunks),
              'row_ids_sha256': signature([c['chunk_id'] for c in chunks]),
              'batch': args.batch, 'implementation': artifact(__file__),
              'resident_ram_floor_bytes': 512*1024**2, 'source_campaign': str(source)}
    if (output/'DENSE_CONFIG.json').exists():
        previous = json.loads((output/'DENSE_CONFIG.json').read_text('utf8'))
        if {k:v for k,v in previous.items() if k!='implementation'} != {k:v for k,v in config.items() if k!='implementation'}:
            raise ValueError('Resume configuration changed')
        if previous != config:
            atomic_json(output/('DENSE_RESUME_'+signature(config)[:12]+'.json'), config)
    else:
        atomic_json(output/'DENSE_CONFIG.json', config)
    with FileLock(str(output/'dense_writer.lock'), timeout=0):
        path = output/'vectors.npy'
        matrix = np.lib.format.open_memmap(path, mode='r+' if path.exists() else 'w+',
                                         dtype=np.float32, shape=(len(chunks), 1024))
        bitmap_path = output/'dense_complete.npy'
        completed = np.load(bitmap_path, allow_pickle=False) if bitmap_path.exists() else np.zeros(len(chunks), dtype=bool)
        reused = 0
        for i, chunk in enumerate(chunks):
            if completed[i]:
                if not np.isfinite(matrix[i]).all() or abs(float(np.linalg.norm(matrix[i]))-1)>1e-4:
                    raise ValueError('Checkpoint vector corrupt')
                continue
            value = cache.get(chunk[field])
            if value is not None:
                matrix[i] = value
                completed[i] = True
                reused += 1
        matrix.flush()
        np.save(bitmap_path, completed, allow_pickle=False)
        pending = np.flatnonzero(~completed).tolist()
        print(json.dumps({'event':'CACHE_REUSE', 'total':len(chunks), 'verified_complete':int(completed.sum()),
                          'reused_this_process':reused, 'remaining':len(pending),
                          'papers':len({c['doc_id'] for c in chunks})}), flush=True)
        if pending:
            # Model load is guarded by ManagedOllama. Once resident, reserve only
            # batch working space; requesting another 3GiB each batch double-counts it.
            with ManagedOllama(output, runtime/'tools/ollama-0.34.4/ollama.exe', num_ctx=4096, startup_ram_bytes=512*1024**2) as owned:
                service = EmbeddingServiceGuard(owned)
                receipt = register_existing_weight(output, service)
                if receipt['weight']['sha256'] != identity['weight_sha256'] or receipt['tag_manifest']['sha256'] != identity['tag_manifest_sha256'] or receipt['backend'] != identity['backend_version']:
                    raise ValueError('Existing embedding identity mismatch')
                tokenizer = HFJsonTokenizer(receipt['tokenizer']['path'], 'Qwen/Qwen3-Embedding-0.6B', identity['tokenizer_revision'])
                started = time.monotonic()
                for begin in range(0, len(pending), args.batch):
                    ids = pending[begin:begin+args.batch]
                    texts = [chunks[i][field] for i in ids]
                    if any(len(tokenizer.encode(t))+1>4096 for t in texts):
                        raise ValueError('No silent embedding truncation')
                    if (begin and begin % 512 == 0) or psutil.virtual_memory().available < 1024**3:
                        # This backend retains host allocations across embed batches.
                        # Unload only our exact model; persisted vectors are unaffected.
                        unloaded = service.session.post(service.base_url+'/api/generate',
                            json={'model':TAG,'keep_alive':0},timeout=30)
                        unloaded.raise_for_status()
                        deadline=time.monotonic()+20
                        while psutil.virtual_memory().available<1024**3 and time.monotonic()<deadline:
                            time.sleep(.5)
                        print(json.dumps({'event':'OWNED_ENCODER_RECYCLED','complete':int(completed.sum()),
                            'available_ram_gib':psutil.virtual_memory().available/1024**3}),flush=True)
                    if psutil.virtual_memory().available < 512*1024**2:
                        raise RuntimeError('Less than 512MiB available RAM; checkpoint preserved')
                    service.verify()
                    rid = signature([identity, ids, [signature(t) for t in texts]])
                    journal = {'request_id':rid, 'offsets':ids, 'status':'RESERVED', 'time':time.time()}
                    with (output/'DENSE_REQUESTS.jsonl').open('a', encoding='utf8') as handle:
                        handle.write(json.dumps(journal)+'\n'); handle.flush(); os.fsync(handle.fileno())
                    tick = time.monotonic()
                    response = service.session.post(service.base_url+'/api/embed', json={
                        'model':TAG, 'input':texts, 'truncate':False, 'keep_alive':'10m',
                        'options':{'num_ctx':4096, 'num_gpu':999, 'num_thread':8}}, timeout=180)
                    response.raise_for_status()
                    raw = response.json()
                    values = np.asarray(raw.pop('embeddings'), dtype=np.float32)
                    norms = np.linalg.norm(values, axis=1, keepdims=True)
                    if values.shape!=(len(ids),1024) or not np.isfinite(values).all() or np.any(norms<=0):
                        raise ValueError('Invalid actual backend vectors')
                    values /= norms
                    matrix[ids] = values
                    matrix.flush()
                    for text, value in zip(texts, values):
                        cache.put(text, value)
                    completed[ids] = True
                    temporary = output/'dense_complete.tmp.npy'
                    np.save(temporary, completed, allow_pickle=False)
                    os.replace(temporary, bitmap_path)
                    atomic_json(output/'embedding_batches'/f'{rid}.json', {
                        'request_id':rid, 'offsets':ids, 'backend_metadata':raw,
                        'elapsed_s':time.monotonic()-tick, 'vector_sha256':signature(values.tolist())})
                    print(json.dumps({'event':'ENCODED', 'complete':int(completed.sum()), 'total':len(chunks),
                        'new_vectors':begin+len(ids), 'elapsed_s':time.monotonic()-started,
                        'available_ram_gib':psutil.virtual_memory().available/1024**3}), flush=True)
        for begin in range(0, len(chunks), 512):
            block = matrix[begin:begin+512]
            if not np.isfinite(block).all() or not np.allclose(np.linalg.norm(block,axis=1),1,atol=1e-4):
                raise ValueError('Full matrix validation failed')
        if not completed.all():
            raise ValueError('Incomplete bitmap')
        atomic_json(output/'FULL_DENSE_INDEX.json', {'status':'COMPLETE', 'papers':len({c['doc_id'] for c in chunks}),
            'chunks':len(chunks), 'vectors':artifact(path), 'bitmap':artifact(bitmap_path),
            'sparse_manifest':pointer, 'embedding':identity, 'config':artifact(output/'DENSE_CONFIG.json'),
            'source_campaign':str(source), 'new_generation_requests':0})
        print('FULL_DENSE_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
