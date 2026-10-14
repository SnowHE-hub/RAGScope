"""CPU-only, resumable three-chunker source preparation without question gold."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
PROJECT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(PROJECT))
from filelock import FileLock
from tokenizers import Tokenizer
from ragscope.experiments.ledger import RecordStore
from ragscope.experiments.runtime import configure,require_ram,require_disk
from ragscope.ingestion.chunkers import TokenizerAdapter,StructureRecursiveChunker,FixedSizeChunker,ParentChildChunker
from ragscope.indexing.identity import artifact,atomic_json,canonical_json,signature


class RawTokenizer:
    def __init__(self,path):self.tokenizer=Tokenizer.from_file(str(path))
    def encode(self,text,add_special_tokens=False):return self.tokenizer.encode(text,add_special_tokens=add_special_tokens).ids
    def decode(self,ids,**kwargs):return self.tokenizer.decode(ids,skip_special_tokens=False)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign',required=True);p.add_argument('--resume',action='store_true');a=p.parse_args()
    runtime=Path(configure(cpu_only=True)['home']).resolve();run=Path(a.campaign).resolve()
    if run.parent!=runtime/'runs':raise ValueError('Wrong campaign')
    require_ram(2*1024**3);require_disk(400*1024**2)
    with FileLock(str(run/'retrieval_source_preparation.lock'),timeout=0):
        source=runtime/'corpora/20261014_v2/all_documents.jsonl'
        tok_path=runtime/'models/qwen3-embedding-0.6b/tokenizer.json'
        tok_art=artifact(tok_path)
        if tok_art['sha256']!='def76fb086971c7867b829c23a26261e38d9d74e02139253b38aeb9df8b4b50a':raise ValueError('Original chunk tokenizer changed')
        tokenizer=TokenizerAdapter(RawTokenizer(tok_path),'Qwen/Qwen3-Embedding-0.6B','97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3')
        groups={'R512':StructureRecursiveChunker(512,tokenizer=tokenizer),
            'F512_O0':FixedSizeChunker(512,0,tokenizer=tokenizer),
            'P256_1024':ParentChildChunker(256,32,1024,tokenizer=tokenizer)}
        config={'source_documents':artifact(source),'tokenizer':tok_art,'runner':artifact(__file__),
            'chunker_implementation':artifact(PROJECT/'ragscope/ingestion/chunkers.py'),
            'groups':{k:v.chunking_signature for k,v in groups.items()},
            'retrieval_text_template':'Title: {original_title}\nSection: {original_section_titles}\n\n{canonical_body}',
            'question_or_answer_gold_read':False,'generation_uses_only_canonical_body':True,'cpu_threads':4}
        signature_id=signature(config);directory=runtime/'indexes'/run.name/'source_chunks'/signature_id;directory.mkdir(parents=True,exist_ok=True)
        store_path=directory/'DOCUMENT_BUILD_RECORDS.jsonl'
        if store_path.exists() and not a.resume:raise ValueError('Existing source build requires --resume')
        store=RecordStore(store_path,config,max_records=10000)
        expected=0
        with source.open(encoding='utf8') as handle:
            for line in handle:
                doc=json.loads(line);unit_by_id={u['unit_id']:u for u in doc['evidence_units']}
                headings={s['section_id']:s['title'] for s in doc['sections']}
                abstract='\n'.join(u['text'] for u in doc['evidence_units'] if u.get('section_id')=='abstract' and u['kind']=='paragraph')
                document_text='Title: '+doc['title']+'\nAbstract: '+abstract+'\nSections: '+' | '.join(s['title'] for s in doc['sections'])
                for group,chunker in groups.items():
                    expected+=1;rid=group+':'+doc['doc_id']
                    if rid in store.ids:
                        previous=next(x for x in store.records if x['record_id']==rid)
                        if artifact(previous['output']['path'])!=previous['output']:raise ValueError('Completed chunk artifact changed')
                        continue
                    require_ram(2*1024**3)
                    chunks,parents=chunker.chunk_with_parents(doc) if group=='P256_1024' else (chunker.chunk(doc),[])
                    rows=[]
                    for chunk in chunks:
                        row=asdict(chunk)
                        if doc['text'][row['canonical_start']:row['canonical_end']]!=row['text']:raise ValueError('Chunk does not match canonical source')
                        section_titles=list(dict.fromkeys(headings.get(unit_by_id[s['unit_id']]['section_id'],'Abstract') for s in row['unit_spans']))
                        row['retrieval_text']='Title: '+doc['title']+'\nSection: '+' | '.join(section_titles)+'\n\n'+row['text']
                        row['retrieval_text_body_coordinates']='text/canonical_start/canonical_end retain original body; prefixes are metadata only'
                        row['retrieval_input_tokens_without_model_wrapper']=tokenizer.count(row['retrieval_text'])
                        rows.append(row)
                    destination=directory/group/(signature(doc['doc_id'])[:24]+'.json')
                    payload={'doc_id':doc['doc_id'],'source_revision':doc['source_revision'],'source_checksum':doc['canonical_checksum'],
                        'document_retrieval_text':document_text,'document_retrieval_tokens':tokenizer.count(document_text),
                        'chunks':rows,'parents':[asdict(x) for x in parents]}
                    if destination.exists():
                        if json.loads(destination.read_text('utf8'))!=payload:raise ValueError('Unjournaled artifact differs; never overwrite')
                    else:atomic_json(destination,payload)
                    store.add(rid,{'group':group,'doc_id':doc['doc_id'],'chunk_n':len(rows),'parent_n':len(parents),
                        'boundary_events':chunker.last_events,'output':artifact(destination),
                        'retrieval_input_token_max':max((x['retrieval_input_tokens_without_model_wrapper'] for x in rows),default=0)})
                    if len(store.ids)%100==0:print(canonical_json({'event':'SOURCE_CHUNKS_PROGRESS','document_group_records':len(store.ids)}),flush=True)
        summary={'status':'PASS_SOURCE_PREPARATION' if len(store.ids)==expected else 'INCOMPLETE','expected_document_groups':expected,
            'completed_document_groups':len(store.ids),'directory':str(directory),'config':config,'journal':artifact(store.path),
            'groups':{g:{'paper_n':sum(x['group']==g for x in store.records),'chunk_n':sum(x['chunk_n'] for x in store.records if x['group']==g),
                'max_retrieval_input_tokens_without_wrapper':max(x['retrieval_input_token_max'] for x in store.records if x['group']==g)} for g in groups},
            'dense_index_status':'NOT_BUILT','bm25_index_status':'NOT_BUILT','question_gold_read':False,'new_llm_requests':0}
        atomic_json(run/'S3_RETRIEVAL_SOURCE_PREPARATION.json',summary)
        print(canonical_json({k:summary[k] for k in ('status','expected_document_groups','completed_document_groups','groups')}));return 0 if len(store.ids)==expected else 2


if __name__=='__main__':raise SystemExit(main())
