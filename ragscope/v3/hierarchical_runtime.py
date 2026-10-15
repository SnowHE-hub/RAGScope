"""Load only completed real indices, never the incomplete full chunk dense matrix."""
import json
from pathlib import Path
import numpy as np
from ragscope.experiments.ledger import read_records
from ragscope.indexing.identity import artifact,verify_artifacts
from ragscope.indexing.sparse_index import BM25Index
from ragscope.v3.compressed_index import CompressedBM25
from ragscope.v3.hierarchical_retrieval import HierarchicalRetriever


class DocumentSparse:
    def __init__(self,index):self.index=index
    def scores(self,question):return np.asarray(self.index.bm25.get_scores(self.index._tokenize(question)))


def load_hierarchy(run):
    run=Path(run);runtime=run.parents[1]
    sparse_document=json.loads((run/'S3_DOCUMENT_BM25_INDEX.json').read_text('utf8'))
    dense_document=json.loads((run/'S3_DOCUMENT_DENSE_INDEX.json').read_text('utf8'))
    if sparse_document['status']!='PASS_REAL_DOCUMENT_BM25_BUILD' or dense_document['status']!='PASS_REAL_DOCUMENT_DENSE_INDEX':
        raise ValueError('Completed real document indices required')
    if not verify_artifacts(sparse_document['outputs']+dense_document['outputs']):raise ValueError('Document index checksum mismatch')
    doc_sparse=BM25Index.load(sparse_document['outputs'][0]['path'])
    matrix=np.load(Path(dense_document['directory'])/'vectors.npy',mmap_mode='r',allow_pickle=False)
    ids=dense_document['config']['paper_ids']
    if ids!=[c['doc_id'] for c in doc_sparse.chunks]:raise ValueError('Document sparse/dense order differs')
    titles={}
    with (runtime/'corpora/20261014_v2/all_documents.jsonl').open(encoding='utf8') as f:
        for line in f:
            doc=json.loads(line);titles[doc['doc_id']]=doc['title']
    documents=[{'doc_id':pid,'title':titles[pid]} for pid in ids]
    sparse_meta=json.loads((run/'S3_CHUNK_SPARSE_INDICES.json').read_text('utf8'))['variants']['R512_body']
    if artifact(sparse_meta['path'])!=sparse_meta:raise ValueError('Chunk sparse manifest changed')
    manifest=json.loads(Path(sparse_meta['path']).read_text('utf8'))
    if not verify_artifacts(manifest['artifacts']):raise ValueError('Chunk sparse index changed')
    directory=Path(sparse_meta['path']).parent;chunks=[json.loads(l) for l in (directory/'chunks.jsonl').read_text('utf8').splitlines()]
    chunk_sparse=CompressedBM25.load(directory);indexed={}
    for c in chunks:indexed.setdefault(c['doc_id'],[]).append(c['chunk_id'])
    scoped={}
    for r in read_records(run/'S3_SCOPED_VECTOR_PAPERS.jsonl'):
        if artifact(r['vectors']['path'])!=r['vectors']:raise ValueError('Compact completed vector changed')
        array=np.load(r['vectors']['path'],allow_pickle=False)
        if array['chunk_ids'].tolist()!=indexed[r['doc_id']]:raise ValueError('Compact/global row order differs')
        scoped[r['doc_id']]=array['vectors']
    return HierarchicalRetriever(documents=documents,document_sparse=DocumentSparse(doc_sparse),document_vectors=matrix,
        chunks=chunks,chunk_sparse=chunk_sparse,scoped_vectors=scoped,embedding_identity=dense_document['config']['embedding'])
