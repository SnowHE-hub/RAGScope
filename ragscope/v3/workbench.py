"""Local diagnostic workbench. Only explicit launch loads existing models."""
from dataclasses import asdict
import json
from pathlib import Path
import time
import uuid
import psutil
from ragscope.experiments.ledger import RecordStore, append_jsonl, read_records
from ragscope.experiments.runtime import require_ram
from ragscope.generation.context import pack_context
from ragscope.generation.pipeline import RAGPipeline
from ragscope.indexing.identity import artifact, canonical_json, signature
from ragscope.v3.budget import Budget
from ragscope.v3.core_packing import pack_core_first
from ragscope.v3.generation_protocols import ProtocolClient, protocol_messages, protocol_schema
from ragscope.v3.records import canonical_payload
from ragscope.v3.splits import assert_inference_safe


def validate_request(payload):
    assert_inference_safe(payload)
    if set(payload) - {'question', 'doc_id', 'condition', 'answer_cache'}:
        raise ValueError('Unsupported request fields')
    if not isinstance(payload.get('question'), str) or not payload['question'].strip():
        return {'action': 'EMPTY_QUESTION', 'message': '请输入问题。'}
    if len(payload['question']) > 8000:
        raise ValueError('问题超过本地输入预算。')
    if payload.get('condition', 'B0') not in {'B0', 'C1_DIAGNOSTIC'}:
        raise ValueError('未批准的实验条件。')
    if not isinstance(payload.get('doc_id', ''), str) or not payload.get('doc_id', '').strip():
        return {'action': 'CLARIFY_DOCUMENT', 'message': '请先指定要分析的论文或文档。'}
    if not isinstance(payload.get('answer_cache', False), bool):
        raise ValueError('缓存设置必须为布尔值。')
    return None


class Workbench:
    def __init__(self, run, engine, service, encoder, original):
        self.run, self.engine, self.service = Path(run), engine, service
        self.encoder, self.original = encoder, original
        self.chunks = {c['chunk_id']: c for c in engine.chunks}
        self.cache = {}; self.budget = Budget(run)
        self.config = {'kind': 'manual_application_supplement_not_frozen_quality_test',
            'implementation': artifact(__file__), 'model': original.identity,
            'embedding': encoder.identity, 'reference_budget': 2048, 'output_budget': 512,
            'default': 'B0', 'C1_admitted': False, 'human_review_status': 'NOT_REVIEWED'}
        path=self.run/('S6_APPLICATION_REQUESTS_'+signature(self.config)[:12]+'.jsonl')
        other_n=sum(len(read_records(p)) for p in self.run.glob('S6_APPLICATION_REQUESTS*.jsonl') if p!=path)
        self.store = RecordStore(path, self.config, max_records=max(0,30-other_n))

    def answer(self, payload):
        validation = validate_request(payload)
        if validation: return validation
        question, pid = payload['question'].strip(), payload['doc_id'].strip()
        if pid not in self.engine.by_doc:
            return {'action': 'INVALID_DOCUMENT', 'message': '该论文不在已核验的索引中。'}
        if pid not in self.engine.scoped_vectors:
            return {'action': 'HYBRID_UNAVAILABLE',
                    'message': '该论文的Hybrid向量索引未完成，不能用稀疏检索冒充保留的Hybrid问答。可查看已保存的范围诊断。',
                    'new_llm_requests': 0}
        condition = payload.get('condition', 'B0')
        key = signature([self.store.config_hash, question, pid, condition])
        if payload.get('answer_cache') and key in self.cache:
            started = time.perf_counter(); result = dict(self.cache[key], answer_cache_hit=True)
            result['cached_application_seconds'] = time.perf_counter()-started
            result['uncached_application_seconds'] = result['application_seconds']
            result['application_seconds'] = result['cached_application_seconds']
            append_jsonl(self.run/'S6_APPLICATION_CACHE_EVENTS.jsonl', {
                'event': 'ANSWER_CACHE_HIT', 'cache_key': key,
                'source_record_id': result['record_id'], 'wall_seconds': result['cached_application_seconds'],
                'new_llm_requests': 0, 'kind': 'application_only_not_primary_online_cost'})
            return result
        # Never return irrelevant zero-score top-k chunks as evidence.
        sparse = self.engine.chunk_sparse.scores(question)
        if max((sparse[i] for i in self.engine.by_doc[pid]), default=0) <= 0:
            return {'action': 'NO_EVIDENCE', 'answer': '当前材料没有检索到相关证据。',
                    'abstain': True, 'citations': [], 'new_llm_requests': 0}
        # A new user input is a new task, unlike the already-frozen online cohort.
        require_ram(2*1024**3)
        self.service.verify(); rid = 'S6:application:'+uuid.uuid4().hex
        client = ProtocolClient(self.original, self.service, protocol='G0')
        self.budget.reserve(request_id=rid, stage='S6_final_validation', max_output_tokens=512,
                            purpose='explicit_manual_UI_application', model_identity=client.identity)
        start = time.perf_counter(); output = None; error = None; issued = False; raw = {}; context = None
        messages = None; ranked = []
        try:
            vector, encoding = self.encoder.query(question, rid)
            result = self.engine.search(question, scope={'kind':'document','doc_ids':[pid]},
                query_vector=vector, embedding_identity=self.encoder.identity)
            ranked = result['candidates']
            if condition == 'B0':
                bundle = pack_context(ranked[:10], max_tokens=2048, scope=pid,
                                      expansion='neighbor', chunk_lookup=self.chunks)
            else:
                bundle, _ = pack_core_first(ranked[:10], max_tokens=2048, scope=pid, chunk_lookup=self.chunks)
            context = asdict(bundle); messages = protocol_messages(question, context, 'G0')
            if self.original.prompt_token_counter(messages)+512 > 4096:
                raise ValueError('实际输入超过共同模型预算。')
            issued = True
            text = client.chat(messages, max_tokens=512, temperature=0, output_schema=protocol_schema(context,'G0'))
            parsed = RAGPipeline.parse_output(text)
            output = {k: parsed[k] for k in ('answer','citations','abstain')}
            if any(c not in {x['chunk_id'] for x in context['pieces']} for c in output['citations']):
                raise ValueError('模型引用不在实际可见材料中。')
            if client.last_metadata.get('token_count_match') is not True:
                raise ValueError('实际输入token与预检不一致。')
            raw = {'content': text, 'raw_backend_response': client.raw_response}
        except Exception as exc:
            error = {'kind': type(exc).__name__, 'message': str(exc)}
            raw = {'raw_backend_response': client.raw_response,
                   'backend_metadata': client.last_metadata}
        elapsed = time.perf_counter()-start
        self.budget.finalize(rid, output_tokens=client.last_metadata.get('eval_count') if issued else 0,
                             outcome='FAILED' if error else 'BACKEND_RESPONSE')
        row = canonical_payload({'question_id': rid, 'dataset_role':'APPLICATION_SUPPLEMENT',
            'exposure_status':'NOT_FROZEN_TEST', 'question': question, 'condition':condition,
            'user_scope':{'kind':'document','doc_ids':[pid]}, 'model_identity': client.identity,
            'context_bundle':context, 'raw_response':raw, 'parsed_response':output,
            'success':error is None, 'error_type':error, 'budget_ticket':rid,
            'wall_time_ms':elapsed*1000, 'answer_cache':False, 'embedding_cache':False,
            'actual_prompt_tokens':client.last_metadata.get('prompt_eval_count'),
            'output_tokens':client.last_metadata.get('eval_count'), 'measured_resource':client.last_metadata.get('resources'),
            'actual_sent_payload':client.sent_payload, 'stop_reason':client.last_metadata.get('done_reason'),
            'prompt_hash':signature(messages) if messages else None,
            'context_intervals':[{'doc_id':x['doc_id'],'start':x['canonical_start'],'end':x['canonical_end']} for x in context['pieces']] if context else [],
            'retriever_candidates':ranked,'rank_scores':[{'chunk_id':x['chunk_id'],'score':x.get('score')} for x in ranked],
            'packing_drops':context['truncations'] if context else []},
            campaign_id=self.run.name, task_id='S6_application')
        self.store.add(rid,row)
        if error: return {'action':'EXECUTION_ERROR','message':error['message'],'record_id':rid}
        answer = {'action':'ANSWER','record_id':rid, **output, 'context_bundle':context,
                  'application_seconds':elapsed,'answer_cache_hit':False,'condition':condition,
                  'release_status':'KEEP_V2_DEFAULT','human_review_status':'NOT_REVIEWED'}
        if payload.get('answer_cache'): self.cache[key] = answer
        return answer
