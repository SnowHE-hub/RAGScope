"""Compressed BM25 and mmap dense search, preserving canonical chunk bodies."""
from collections import Counter,defaultdict
import json
from pathlib import Path
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer
from ragscope.indexing.array_index import tokenize
from ragscope.indexing.identity import artifact,atomic_json,signature


class CompressedBM25:
    def __init__(self,weights,vocabulary):self.weights,self.vocabulary=weights,vocabulary

    @classmethod
    def build(cls,texts):
        if not texts:raise ValueError('Empty corpus')
        analyzer=lambda text:tokenize(text) or ['__empty__']
        vectorizer=CountVectorizer(analyzer=analyzer,dtype=np.int32)
        counts=vectorizer.fit_transform(texts).tocsr()
        lengths=np.asarray(counts.sum(axis=1)).ravel().astype(np.float64)
        mean_length=float(lengths.mean());n=len(texts)
        df=np.asarray((counts>0).sum(axis=0)).ravel()
        idf=np.log(n-df+.5)-np.log(df+.5)
        idf[idf<0]=.25*float(idf.mean())
        weights=counts.astype(np.float64)
        rows=np.repeat(np.arange(n),np.diff(weights.indptr))
        denom=weights.data+1.5*(1-.75+.75*lengths[rows]/mean_length)
        weights.data=weights.data*2.5/denom*idf[weights.indices]
        return cls(weights.tocsc(),vectorizer.vocabulary_)

    def scores(self,query):
        result=np.zeros(self.weights.shape[0],dtype=np.float64)
        for token,count in Counter(tokenize(query)).items():
            column=self.vocabulary.get(token)
            if column is None:continue
            start,end=self.weights.indptr[column:column+2]
            result[self.weights.indices[start:end]]+=count*self.weights.data[start:end]
        return result

    def save(self,directory):
        directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
        if (directory/'bm25_weights.npz').exists() or (directory/'vocabulary.json').exists():raise ValueError('Do not overwrite sparse index')
        sparse.save_npz(directory/'bm25_weights.npz',self.weights,compressed=True)
        atomic_json(directory/'vocabulary.json',self.vocabulary)
        return [artifact(directory/'bm25_weights.npz'),artifact(directory/'vocabulary.json')]

    @classmethod
    def load(cls,directory):
        directory=Path(directory)
        return cls(sparse.load_npz(directory/'bm25_weights.npz').tocsc(),json.loads((directory/'vocabulary.json').read_text('utf8')))


class CanonicalIndex:
    def __init__(self,chunks,bm25,vectors,identity):
        self.chunks,self.bm25,self.vectors,self.identity=chunks,bm25,vectors,identity
        if bm25.weights.shape[0]!=len(chunks) or vectors.shape!=(len(chunks),1024):raise ValueError('Index alignment mismatch')
        if len({x['chunk_id'] for x in chunks})!=len(chunks):raise ValueError('Duplicate chunk IDs')
        self.by_doc=defaultdict(list)
        for i,x in enumerate(chunks):self.by_doc[x['doc_id']].append(i)

    def search(self,question,*,method='hybrid',query_vector=None,embedding_identity=None,scope=None,k=10,candidate_depth=50,alpha=.5):
        if k<1 or candidate_depth<k or alpha not in {.25,.5,.75}:raise ValueError('Unapproved retrieval bounds')
        ids=self.by_doc.get(scope,[]) if scope is not None else range(len(self.chunks))
        ids=list(ids)
        if not ids:return []
        sparse_scores=self.bm25.scores(question) if method in {'bm25','hybrid'} else None
        sparse_rank=sorted(ids,key=lambda i:(-sparse_scores[i],self.chunks[i]['chunk_id']))[:candidate_depth] if sparse_scores is not None else []
        if method in {'dense','hybrid'}:
            if embedding_identity!=self.identity['embedding'] or query_vector is None:raise ValueError('Query embedding space mismatch')
            if np.asarray(query_vector).shape!=(1024,):raise ValueError('Query vector shape mismatch')
            dense_scores=self.vectors@np.asarray(query_vector,dtype=np.float32)
            dense_rank=sorted(ids,key=lambda i:(-dense_scores[i],self.chunks[i]['chunk_id']))[:candidate_depth]
        if method=='bm25':ranked=sparse_rank;scores={i:float(sparse_scores[i]) for i in ranked}
        elif method=='dense':ranked=dense_rank;scores={i:float(dense_scores[i]) for i in ranked}
        elif method=='hybrid':
            scores=defaultdict(float)
            for weight,ranking in ((1-alpha,sparse_rank),(alpha,dense_rank)):
                for rank,i in enumerate(ranking,1):scores[i]+=weight/(60+rank)
            ranked=sorted(scores,key=lambda i:(-scores[i],self.chunks[i]['chunk_id']))
        else:raise ValueError('Unknown retrieval method')
        # Canonical body remains text. Metadata prefixes were used only to build index inputs.
        return [{**self.chunks[i],'score':scores[i]} for i in ranked[:k]]

    @classmethod
    def load(cls,manifest_path):
        manifest=json.loads(Path(manifest_path).read_text('utf8'))
        if manifest['status']!='complete' or signature(manifest['identity'])!=manifest['signature']:raise ValueError('Incomplete or corrupt index manifest')
        if any(artifact(x['path'])!=x for x in manifest['artifacts']):raise ValueError('Index artifact changed')
        directory=Path(manifest_path).parent
        chunks=[json.loads(l) for l in (directory/'chunks.jsonl').read_text('utf8').splitlines()]
        return cls(chunks,CompressedBM25.load(directory),np.load(directory/'vectors.npy',mmap_mode='r',allow_pickle=False),manifest['identity'])
