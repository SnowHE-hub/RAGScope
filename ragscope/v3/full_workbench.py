"""Usable full-corpus workbench; manual requests remain separate from benchmarks."""
import json
from pathlib import Path
import time
import uuid
from ragscope.experiments.ledger import RecordStore
from ragscope.generation.pipeline import RAGPipeline
from ragscope.generation.prompts import visible_output_schema
from ragscope.indexing.identity import artifact, signature
from ragscope.v3.finish_runtime import answer_messages
from ragscope.v3.splits import assert_inference_safe
from ragscope.v3.deployment import deployment_context,DeploymentClient


def validate_full_request(payload):
    if not isinstance(payload,dict):raise ValueError('JSON object required')
    assert_inference_safe(payload)
    if set(payload)-{'question','doc_ids','task','method','answer_cache'}:
        raise ValueError('Unsupported request fields')
    if not isinstance(payload.get('question'),str) or not payload['question'].strip():
        return {'action':'EMPTY_QUESTION','message':'请输入问题。'}
    if len(payload['question'])>8000:raise ValueError('问题超过本地输入预算。')
    ids=payload.get('doc_ids',[])
    if not isinstance(ids,list) or len(ids)>10 or any(not isinstance(p,str) for p in ids) or len(set(ids))!=len(ids):
        raise ValueError('论文范围必须是最多10个不同的doc_id。')
    if payload.get('task','auto') not in {'auto','fact','explanation','comparison','summary'}:
        raise ValueError('未知任务类型。')
    if payload.get('method','hybrid') not in {'bm25','dense','hybrid'}:
        raise ValueError('未知检索方式。')
    if not isinstance(payload.get('answer_cache',False),bool):raise ValueError('缓存设置必须为布尔值。')
    return None


class FullWorkbench:
    def __init__(self,run,engine,service,encoder,original):
        self.run,self.engine,self.service=Path(run),engine,service
        self.encoder,self.original=encoder,original
        selection=json.loads((self.run/'RETRIEVAL_SELECTION.json').read_text('utf8'))
        self.selection=selection
        self.selected=selection['selected']['options']
        self.global_selected=selection['global_selected']['options']
        self.qa_policy=json.loads((self.run/'GENERATION_SELECTION.json').read_text('utf8')).get('qa_policy','FINAL')
        self.cache={}
        config={'kind':'MANUAL_APPLICATION_NOT_FROZEN_TEST','implementation':artifact(__file__),
            'configuration':artifact(self.run/'FINAL_CONFIGURATION.json'),'deployment':artifact(self.run/('DEPLOYMENT_CONFIGURATION_RESIDENT8K.json' if (self.run/'DEPLOYMENT_CONFIGURATION_RESIDENT8K.json').exists() else 'DEPLOYMENT_CONFIGURATION.json')),
            'application_revision':artifact(self.run/'APPLICATION_REVISION.json') if (self.run/'APPLICATION_REVISION.json').exists() else None,
            'model':original.identity,'index':artifact(self.run/'FULL_DENSE_INDEX.json')}
        self.store=RecordStore(self.run/('APPLICATION_'+signature(config)[:12]+'.jsonl'),config,max_records=100000)

    def answer(self,payload):
        invalid=validate_full_request(payload)
        if invalid:return invalid
        ids=payload.get('doc_ids',[]);question=payload['question'].strip()
        if any(pid not in self.engine.by_doc for pid in ids):
            return {'action':'INVALID_DOCUMENT','message':'指定论文不在完整索引中。'}
        if not ids and any(phrase in question.lower() for phrase in ('this paper','the paper','this study')):
            return {'action':'CLARIFY_DOCUMENT','message':'这个问题提到“这篇论文”，请在论文范围中选择具体论文。'}
        key=signature([self.store.config_hash,payload])
        if payload.get('answer_cache') and key in self.cache:
            return {**self.cache[key],'answer_cache_hit':True}
        rid='application:'+uuid.uuid4().hex;started=time.perf_counter()
        scope={'kind':'document','doc_ids':ids} if ids else {'kind':'global'}
        client=DeploymentClient(self.original,self.service)
        context=None;parsed=None;error=None;raw=None;retrieval=None;kind=None
        try:
            self.service.verify()
            vector,encoding=self.encoder.query(question,rid,cached=True)
            explicit=payload.get('task')
            context,retrieval,kind,ctx,limit,effective=deployment_context(self.engine,question,scope,vector,self.original,self.selection,explicit,payload.get('method','hybrid'))
            client.num_ctx=ctx;client.identity['num_ctx']=ctx
            if explicit in {'fact','explanation','comparison','summary'}:kind=explicit
            messages=answer_messages(question,context,condition=effective,kind=kind)
            schema=visible_output_schema(context['pieces'])
            if len({piece['doc_id'] for piece in context['pieces']})>1:
                messages[0]['content']+=' Identify the papers clearly. Use at most three concise sentences per paper and at most four distinct citations.'
                schema['properties']['citations']['maxItems']=4
                schema['properties']['answer']['maxLength']=1600
            raw=client.chat(messages,max_tokens=limit,temperature=0,output_schema=schema)
            output=RAGPipeline.parse_output(raw);parsed={k:output[k] for k in ('answer','citations','abstain')}
            if client.last_metadata.get('token_count_match') is not True:raise ValueError('实际输入token检查失败。')
            if any(cid not in {p['chunk_id'] for p in context['pieces']} for cid in parsed['citations']):
                raise ValueError('引用没有对应可见证据。')
        except Exception as exc:error={'kind':type(exc).__name__,'message':str(exc)}
        elapsed=time.perf_counter()-started
        self.store.add(rid,{'question':question,'scope':scope,'context':context,'parsed':parsed,'raw_text':raw,
            'raw_backend_response':getattr(client,'raw_response',None),'actual_sent_payload':client.sent_payload,
            'backend':client.last_metadata,'elapsed_s':elapsed,'success':error is None,'error':error,'task':kind,
            'answer_cache':False,'query_embedding_cache_allowed':True,'model_identity':client.identity})
        if error:return {'action':'EXECUTION_ERROR','message':error['message'],'record_id':rid}
        result={'action':'ANSWER',**parsed,'context_bundle':context,'record_id':rid,'application_seconds':elapsed,
            'answer_cache_hit':False,'task':kind,'document_candidates':retrieval['document_candidates'],
            'source_character_coverage':retrieval.get('source_character_coverage',{}),
            'model':self.original.model,'indexed_papers':len(self.engine.by_doc)}
        if payload.get('answer_cache'):self.cache[key]=result
        return result
