"""Complete corpus retrieval with document priors and explicit multi-paper scope."""
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
from ragscope.indexing.identity import artifact,verify_artifacts,signature
from ragscope.v3.compressed_index import CompressedBM25
from ragscope.v3.hierarchical_runtime import load_hierarchy


class FullCorpusRetriever:
    def __init__(self, source, completion):
        source, completion = Path(source), Path(completion)
        manifest = json.loads((completion/'FULL_DENSE_INDEX.json').read_text('utf8'))
        if manifest['status'] != 'COMPLETE' or artifact(manifest['vectors']['path']) != manifest['vectors']:
            raise ValueError('Complete verified full dense index required')
        old = load_hierarchy(source)
        self.__dict__.update(old.__dict__)
        self.vectors = np.load(manifest['vectors']['path'], mmap_mode='r', allow_pickle=False)
        if self.vectors.shape != (len(self.chunks), 1024) or manifest['embedding'] != self.embedding_identity:
            raise ValueError('Full corpus alignment or embedding space differs')
        config=json.loads(Path(manifest['config']['path']).read_text('utf8'))
        if signature([c['chunk_id'] for c in self.chunks])!=config['row_ids_sha256']:
            raise ValueError('Full dense row order differs from canonical chunks')
        sparse_pointer = json.loads((source/'S3_CHUNK_SPARSE_INDICES.json').read_text('utf8'))['variants']['R512_metadata']
        metadata_manifest=json.loads(Path(sparse_pointer['path']).read_text('utf8'))
        if artifact(sparse_pointer['path'])!=sparse_pointer or not verify_artifacts(metadata_manifest['artifacts']):
            raise ValueError('Metadata sparse index checksum mismatch')
        self.metadata_sparse = CompressedBM25.load(Path(sparse_pointer['path']).parent)
        if self.metadata_sparse.weights.shape[0]!=len(self.chunks):
            raise ValueError('Metadata sparse row count differs')
        with (Path(sparse_pointer['path']).parent/'chunks.jsonl').open(encoding='utf8') as handle:
            metadata_ids=[json.loads(line)['chunk_id'] for line in handle]
        if metadata_ids!=[c['chunk_id'] for c in self.chunks]:
            raise ValueError('Metadata sparse row order differs')
        self.doc_rows = {d['doc_id']: i for i,d in enumerate(self.documents)}
        self.manifest = manifest
        self.scoped_vectors = {pid: True for pid in self.by_doc}
        self.chunk_rows = {c['chunk_id']: i for i,c in enumerate(self.chunks)}

    def search(self, question, *, scope=None, query_vector=None, embedding_identity=None,
               method='hybrid', alpha=.5, metadata_weight=0., document_weight=0.,
               k=50, candidate_depth=200, balanced=False, **unused):
        if method not in {'bm25','dense','hybrid'} or not 0<=alpha<=1:
            raise ValueError('Invalid retrieval method or fusion')
        specified = (scope or {}).get('doc_ids', []) if isinstance(scope,dict) else ([scope] if scope else [])
        if any(pid not in self.by_doc for pid in specified):
            raise ValueError('Specified paper is not indexed')
        ids = np.asarray([i for pid in specified for i in self.by_doc[pid]] if specified else range(len(self.chunks)), dtype=np.int32)
        if not len(ids):
            return {'action':'RETRIEVE','candidates':[], 'document_candidates':[]}
        vector = None
        if method != 'bm25':
            if embedding_identity != self.embedding_identity:
                raise ValueError('Query embedding space differs')
            vector = np.asarray(query_vector,dtype=np.float32)
            if vector.shape!=(1024,) or not np.isfinite(vector).all():
                raise ValueError('Invalid query vector')
        depth = min(candidate_depth, len(ids))
        def ranking(values):
            return ids[np.argsort(-values[ids],kind='stable')[:depth]].tolist()
        ranks=[];weights=[]
        if method in {'bm25','hybrid'}:
            ranks.append(ranking(self.chunk_sparse.scores(question)))
            weights.append(1 if method=='bm25' else 1-alpha)
        if method in {'dense','hybrid'}:
            # A scoped query multiplies only its document rows.
            dense = np.empty(len(self.chunks), dtype=np.float32)
            dense[ids] = self.vectors[ids]@vector
            ranks.append(ranking(dense));weights.append(1 if method=='dense' else alpha)
        if metadata_weight and method!='dense':
            ranks.append(ranking(self.metadata_sparse.scores(question))); weights.append(metadata_weight)
        scores=defaultdict(float)
        for weight, rank in zip(weights,ranks):
            for pos,i in enumerate(rank,1):
                scores[i] += weight/(60+pos)
        docs=[]
        if not specified:
            ds=self.document_sparse.scores(question)
            doc_ranks=[] if method=='dense' else [np.argsort(-ds,kind='stable')[:20]]
            if vector is not None:
                doc_ranks.append(np.argsort(-(self.document_vectors@vector),kind='stable')[:20])
            doc_scores=defaultdict(float)
            for rank in doc_ranks:
                for pos,i in enumerate(rank,1):doc_scores[int(i)]+=1/len(doc_ranks)/(60+pos)
            docs=[{**self.documents[i],'score':score} for i,score in sorted(doc_scores.items(),key=lambda p:-p[1])[:10]]
            for i in scores:
                scores[i] += document_weight*doc_scores.get(self.doc_rows[self.chunks[i]['doc_id']],0)
        order=sorted(scores,key=lambda i:(-scores[i],self.chunks[i]['chunk_id']))
        if balanced and len(specified)>1:
            groups=[[i for i in order if self.chunks[i]['doc_id']==pid] for pid in specified]
            order=[group[r] for r in range(max(map(len,groups))) for group in groups if r<len(group)]
        return {'action':'RETRIEVE','candidates':[{**self.chunks[i],'score':float(scores[i])} for i in order[:k]],
                'document_candidates':docs,'document_lookup_performed':not bool(specified),
                'scope_policy':'USER_SPECIFIED_DOCUMENTS' if specified else 'FULL_CORPUS',
                'full_chunk_dense_status':'COMPLETE','papers_indexed':len(self.by_doc),
                'global_bypass_used':not bool(specified),'retrieval_method':method}
