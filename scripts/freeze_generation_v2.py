"""Freeze mandatory generation groups only after pilot/selection/input validation.

No model generation, service startup or downloads occur here. Pending prerequisites
produce FREEZE_READINESS.json; explicit evidence-backed blocks preserve denominators.
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import copy
from dataclasses import asdict
import json
import random

from ragscope.experiments.generation_campaign import (freeze_campaign, oracle_context, position_contexts,
    validate_core_matrix, iter_record_sources, load_campaign)
from ragscope.experiments.ledger import read_records
from ragscope.experiments.runtime import runtime_home
from ragscope.generation.context import pack_context, render_context, validate_bundle
from ragscope.generation.llm_client import LocalLLMClient
from ragscope.generation.prompts import (RAG_SYSTEM_PROMPT, RAG_USER_TEMPLATE, PROMPT_VERSION,
    OUTPUT_CONSTRAINT_VERSION, visible_output_schema)
from ragscope.indexing.identity import artifact, atomic_json, file_hash, signature, verify_artifacts

RUNTIME_POLICY={'prompt_ram_cache_mib':0,'prompt_cache_policy':'backend_history_disabled'}


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def read_jsonl(path):
    with Path(path).open(encoding='utf-8-sig') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def messages(question, context):
    return [{'role': 'system', 'content': RAG_SYSTEM_PROMPT},
            {'role': 'user', 'content': RAG_USER_TEMPLATE.format(question=question, context=context)}]


def seeded_subset(rows, n):
    rows = sorted(rows, key=lambda row: row['query_id'])
    random.Random(42).shuffle(rows)
    return rows[:n]


def verified_corpus(directory):
    directory = Path(directory)
    manifest = read_json(directory / 'SPLIT_MANIFEST.json')
    names = ['documents.jsonl'] + [f'queries_{split}.jsonl' for split in ('train', 'dev', 'test')
                                  if f'queries_{split}.jsonl' in manifest['files']]
    for name in names:
        if file_hash(directory/name) != manifest['files'][name]['sha256']:
            raise ValueError('Frozen corpus changed: '+name)
    queries = {split: read_jsonl(directory/f'queries_{split}.jsonl') for split in ('train', 'dev', 'test')
               if (directory/f'queries_{split}.jsonl').exists()}
    for split, rows in queries.items():
        expected_ids = (manifest['query_ids'][split] if 'query_ids' in manifest else
                        manifest['selected_query_ids'] if split == 'test' else [])
        if set(row['query_id'] for row in rows) != set(expected_ids):
            raise ValueError('Frozen query membership changed')
    documents = read_jsonl(directory/'documents.jsonl')
    units = {unit['unit_id']: unit for doc in documents for unit in doc['evidence_units']}
    return queries, units, artifact(directory/'SPLIT_MANIFEST.json')


def document_lengths(directory):
    """Copy non-label canonical-document metadata into frozen analysis inputs."""
    result = {}
    for doc in read_jsonl(Path(directory)/'documents.jsonl'):
        meta = doc.get('metadata', {})
        count = meta.get('reference_n_tokens')
        group = 'short' if isinstance(count,int) and count<=512 else 'medium' if isinstance(count,int) and count<=4096 else 'long' if isinstance(count,int) else None
        if count is not None and (not isinstance(count,int) or isinstance(count,bool) or count<0):
            raise ValueError('Invalid canonical document reference token count')
        if group is not None and meta.get('length_group',group)!=group:
            raise ValueError('Canonical document length group disagrees with frozen boundaries')
        result[doc['doc_id']]={'doc_id':doc['doc_id'],'reference_n_tokens':count,'length_group':group,
            'reference_tokenizer':meta.get('reference_tokenizer'), 'definition':'canonical document before chunking; short<=512, medium<=4096, long>4096'}
    return result


def verify_pilot(client, counter, summary_path, records_path, dev_ids):
    summary = read_json(summary_path)
    if summary.get('status') not in {'PASS','PASS_WITH_OUTPUT_FAILURES'} or summary.get('expected_n') != 20 or summary.get('unique_queries') != 20:
        raise ValueError('Twenty distinct dev pilots have not passed')
    if summary.get('model_identity') != client.identity or summary.get('counter_identity') != counter.identity:
        raise ValueError('Pilot model/tokenizer identity differs from requested formal client')
    rows = read_records(Path(records_path))
    latest = {}
    for row in rows:
        result = row.get('result', {})
        if result.get('model_identity') == client.identity and row.get('purpose') == 'pilot':
            if row.get('split') != 'dev' or row['query_id'] not in dev_ids or not row.get('ticket'):
                raise ValueError('Pilot lacks legitimate dev membership or global budget ticket')
            latest[row['query_id']] = row
    if len(latest) != 20:
        raise ValueError('Pilot raw rows do not establish twenty distinct verified dev queries')
    output_failures=[]
    for row in latest.values():
        result = row['result']
        observed = result.get('backend_metadata', {}).get('prompt_eval_count')
        if observed is None:
            observed = (result.get('backend_metadata', {}).get('usage') or {}).get('prompt_tokens')
        error=result.get('error')
        bounded_output_failure=bool(error and error.get('kind')=='parse_error'
            and result.get('backend_metadata',{}).get('done_reason')=='length')
        if (error and not bounded_output_failure) or result.get('real_model') is not True or observed != counter(result['prompt']):
            raise ValueError('Pilot failure or actual backend/template token mismatch')
        if bounded_output_failure:
            output_failures.append({'query_id':row['query_id'],'error':error,'done_reason':'length'})
        if result['prompt'] != messages(result['question'], render_context(result['context_bundle'])):
            raise ValueError('Pilot prompt differs from the current frozen prompt version')
        if (result.get('prompt_version')!=PROMPT_VERSION or not result.get('citation_constraint',{}).get('applied')
                or result.get('output_schema')!=visible_output_schema(result['context_bundle']['pieces'])):
            raise ValueError('Pilot has not verified the final per-context visible citation schema')
    return {'summary': artifact(Path(summary_path)), 'records': artifact(Path(records_path)),
            'query_ids': sorted(latest), 'verified_n': len(latest), 'counter_identity': counter.identity,
            'output_failures':output_failures,'valid_output_n':len(latest)-len(output_failures),
            'interpretation':'Completed length-limited parse failures retained as model outcomes; no output repair or denominator removal'}


def model_client_factory(spec, dev_ids, verify_weights=True):
    """Recreate a pinned local client without loading weights or calling its service."""
    if read_json(spec['pilot_summary']).get('status') not in {'PASS','PASS_WITH_OUTPUT_FAILURES'}:
        raise ValueError('Twenty-dev pilot not ready; no tokenizer or weight inspection started')
    source = read_json(spec['source_manifest'])
    tokenizer_manifest = read_json(Path(spec['tokenizer_dir'])/'manifest.json')
    if not verify_artifacts(tokenizer_manifest['artifacts']):
        raise ValueError('Tokenizer artifacts changed')
    if verify_weights and not verify_artifacts(source['artifacts'] + [source['manifest']]):
        raise ValueError('Pinned generation artifacts changed')
    if spec.get('counter_kind', 'qwen35') == 'qwen35':
        from ragscope.generation.token_budget import Qwen35PromptCounter
        counter = Qwen35PromptCounter(spec['tokenizer_dir'], spec['renderer_source'])
    elif spec['counter_kind'] == 'lfm25':
        from ragscope.generation.token_budget import LFMPromptCounter
        counter = LFMPromptCounter(spec['tokenizer_dir'], spec['renderer_source'])
    elif spec['counter_kind'] == 'hf_chat':
        # This path is enabled only after actual backend token counts agree on
        # all 20 pilots; a merely available HF template is insufficient.
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(spec['tokenizer_dir'],local_files_only=True,trust_remote_code=False)
        template_options=spec.get('template_options',{'enable_thinking':False})
        class HFChatCounter:
            identity={'tokenizer_manifest_sha256':file_hash(Path(spec['tokenizer_dir'])/'manifest.json'),
                      'chat_template_sha256':signature(tokenizer.chat_template),'options':template_options,
                      'implementation':'official_hf_chat_template_local_v1'}
            template_hash=signature(identity)
            def __call__(self,messages):
                return len(tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,**{**template_options,'return_dict':False}))
        counter=HFChatCounter()
    else:
        raise ValueError('No verified token counter factory for requested model')
    weight = next(row for row in source['artifacts'] if row['media_type'] == 'application/vnd.ollama.image.model')
    options = dict(spec.get('client_options', {}))
    client = LocalLLMClient(source['model'], revision=source['manifest_sha256'], weight_sha256=weight['sha256'],
        quantization=source['config']['file_type'], backend_version=spec['backend_version'],
        chat_template_hash=counter.template_hash, prompt_token_counter=counter,
        **{'num_gpu': 999, 'num_ctx': 4096, 'think': False, 'service_device_verified': 'gpu', 'timeout': 180, **options})
    pilot = verify_pilot(client, counter, spec['pilot_summary'], spec['pilot_records'], dev_ids)
    return client, pilot


def explicit_block(block):
    if not block or not block.get('reason') or not block.get('evidence'):
        return None
    paths = [Path(path) for path in block['evidence']]
    if not all(path.is_file() for path in paths):
        raise ValueError('Block evidence must exist')
    return {'status': 'BLOCKED', 'reason': block['reason'], 'evidence': [artifact(path) for path in paths]}


def verified_context_rows(path, queries):
    path = Path(path)
    rows = read_records(path)
    config = read_json(path.with_suffix('.config.json'))
    mapping = {row['query_id']: row for row in rows}
    if len(mapping) != len(rows):
        raise ValueError('Duplicate retrieval query rows')
    if not set(q['query_id'] for q in queries) <= set(mapping):
        raise ValueError('Retrieval rows omit frozen queries')
    result = []
    for query in queries:
        row = mapping[query['query_id']]
        if row.get('config_hash') != signature(config) or not row.get('record_hash'):
            raise ValueError('Retrieval config/record identity missing')
        if row['question'] != query['question'] or row['scope'] != query['scope'] or row['split'] != query['split']:
            raise ValueError('Retrieval query/scope/split differs from frozen population')
        if row.get('error'):
            raise ValueError('Retrieval has failed rows; preserve and explicitly block this condition until resolved')
        validate_bundle(row['context_bundle'], scope=query['scope'])
        result.append({'query': query, 'context_bundle': row['context_bundle'], 'raw_retrieval': row['raw_retrieval'],
            'selected_retrieval': row.get('selected_retrieval'),
            'retrieval_timings': row.get('timings', {}), 'retrieval_source': str(path),
            'context_config': {'source_config_hash': row['config_hash'], 'index_signature': row.get('index_signature'),
                               'retrieval_record_id': row['record_id']}})
    return result, {'records': artifact(path), 'config': artifact(path.with_suffix('.config.json'))}


def shared_context(source, clients):
    """Repack once against every applicable true template; send identical result."""
    query = source['query']
    checks = [lambda text, client=client: client.prompt_token_counter(messages(query['question'], text))+512 <= client.num_ctx
              for client in clients]
    if not all(check('') for check in checks):
        raise ValueError('Question plus prompt/output budget exceeds a model context before evidence')
    old = source['context_bundle']
    bundle = pack_context(old['pieces'], max_tokens=2048, scope=query.get('scope'), model_budget_checks=checks)
    return {**source, 'retrieval_context_bundle': old, 'context_bundle': asdict(bundle),
            'full_prompt_tokens_by_model': {client.model: client.prompt_token_counter(messages(query['question'], render_context(bundle)))
                                           for client in clients}}


def unit_piece(unit, prefix):
    return {'chunk_id':prefix+unit['unit_id'],'doc_id':unit['doc_id'],'text':unit['text'],
        'canonical_start':unit['canonical_start'],'canonical_end':unit['canonical_end'],
        'unit_spans':[{'unit_id':unit['unit_id'],'doc_id':unit['doc_id'],
                       'start':unit['canonical_start'],'end':unit['canonical_end']}]}


def make_position_fixtures(queries, units, clients, n=40):
    """Preselect by source/budget feasibility only; never read generated answers.

    Distractors are same-paper paragraphs outside every annotated evidence set.
    Their semantic irrelevance is not claimed beyond the annotation exclusion.
    """
    accepted, rejected = [], []
    by_doc = {}
    for unit in units.values():
        by_doc.setdefault(unit['doc_id'],[]).append(unit)
    for q in seeded_subset([q for q in queries if q['answerability']=='answerable'],len(queries)):
        try:
            reference = q['evidence_reference_sets'][0]
            excluded = {uid for ref in q['evidence_reference_sets'] for uid in ref}
            evidence = [unit_piece(units[uid],'evidence:') for uid in dict.fromkeys(reference)]
            noise_units = [u for u in sorted(by_doc[q['doc_id']],key=lambda u:u['canonical_start'])
                           if u['unit_id'] not in excluded and u.get('kind')=='paragraph' and 100<=len(u['text'])<=1200]
            if len(noise_units)<2:
                raise ValueError('fewer than two bounded unannotated same-paper paragraphs')
            noise = [unit_piece(u,'noise:') for u in noise_units[:2]]
            checks = [lambda text,c=c,q=q:c.prompt_token_counter(messages(q['question'],text))+512<=c.num_ctx for c in clients]
            position_contexts(evidence,noise,model_budget_checks=checks)
            accepted.append({'query_id':q['query_id'],'evidence_pieces':evidence,'noise_pieces':noise})
            if len(accepted)>=n: break
        except (KeyError,IndexError,ValueError) as exc:
            rejected.append({'query_id':q['query_id'],'reason':str(exc)})
    return {'fixtures':accepted,'excluded_before_generation':rejected,'target_n':n,'actual_n':len(accepted),
            'rule':'seed42 source/budget feasibility; two first eligible same-paper paragraphs outside all reference sets',
            'noise_limitation':'unannotated distractor paragraphs; semantic irrelevance not independently human-verified'}


def default_spec(home, campaign):
    home, run = Path(home), Path(home)/'runs'/campaign
    spec = {'campaign': campaign, 'output_dir': str(home/'runs'/(campaign+'_generation')),
        'service_receipt':str(run/'generation_service.json'),
        'corpus_dir': str(home/'corpora'/campaign), 'retrieval_run': str(home/'runs'/(campaign+'_retrieval')),
        'main_models': ['qwen_2b', 'qwen_4b', 'qwen_9b'], 'small_models': ['qwen_2b', 'qwen_4b'],
        'smallest_model': 'qwen_2b', 'heterogeneous_model': 'lfm',
        'models': {f'qwen_{size}': {'source_manifest': str(home/'models/generation/text_only'/f'qwen3.5_{size}-q4_K_M.json'),
            'tokenizer_dir': str(home/'models/tokenizers'/f'Qwen3.5-{size.upper()}'),
            'renderer_source': str(home/'downloads/ollama_renderer_0.34.4/qwen35.go'),
            'backend_version': '0.34.4', 'pilot_summary': str(run/f'pilot_{size}_summary.json'),
            'pilot_records': str(run/'generation_pilots.jsonl')} for size in ['2b','4b','9b']},
        'context_sources': {}, 'blocks': {}, 'model_blocks': {}}
    spec['models']['lfm']={'counter_kind':'lfm25','source_manifest':str(home/'models/generation/lfm2.5_q4_k_m.json'),
        'tokenizer_dir':str(home/'models/tokenizers/LFM2.5-1.2B-Instruct'),
        'renderer_source':str(home/'models/ollama/blobs/sha256-58e51bf3ff10a5f7ed85bdbcb131428218f367ffd9fd30915cd73e31d2edb904'),
        'backend_version':'0.34.4','pilot_summary':str(run/'pilot_lfm_summary.json'),
        'pilot_records':str(run/'generation_pilots.jsonl')}
    shard=home/'runs'/(campaign+'_generation_e3')/'EXPERIMENT_MANIFEST.json'
    if shard.exists():
        spec['external_campaigns']=[{'manifest':artifact(shard)}]
    return spec


def freeze_all(spec, *, client_factory=model_client_factory):
    spec=copy.deepcopy(spec)
    corpus = Path(spec['corpus_dir'])
    queries, units, corpus_artifact = verified_corpus(corpus)
    lengths = document_lengths(corpus)
    retrieval = Path(spec['retrieval_run'])
    readiness = {'status': 'PENDING', 'pending': [], 'blocks': {}, 'model_pilots': {}, 'generation_calls': 0}
    service_receipt=None
    try:
        service_receipt=read_json(spec['service_receipt'])
        if service_receipt.get('environment',{}).get('LLAMA_ARG_CACHE_RAM')!='0':
            raise ValueError('Formal runtime requires LLAMA_ARG_CACHE_RAM=0, independently of answer-cache policy')
        readiness['backend_runtime_config']=RUNTIME_POLICY
        for model_spec in spec['models'].values():
            options=model_spec.setdefault('client_options',{})
            if options.get('backend_runtime_config',RUNTIME_POLICY)!=RUNTIME_POLICY:
                raise ValueError('Requested model runtime policy differs from verified service')
            options['backend_runtime_config']=RUNTIME_POLICY
    except (OSError,ValueError,KeyError) as exc:
        readiness['pending'].append({'item':'service_runtime_policy','reason':str(exc)})
    clients, models = {}, {}
    all_model_ids = list(dict.fromkeys(spec['main_models']+[spec['heterogeneous_model']]))
    for model_id in all_model_ids:
        block = explicit_block(spec.get('model_blocks', {}).get(model_id))
        if block:
            readiness['blocks']['model:'+model_id] = block
            continue
        if model_id not in spec['models']:
            readiness['pending'].append({'item': 'model:'+model_id, 'reason': 'Verified source, backend, tokenizer and pilot configuration missing'})
            continue
        try:
            client, audit = client_factory(spec['models'][model_id], {row['query_id'] for row in queries['dev']})
            clients[model_id], models[model_id] = client, client.identity
            readiness['model_pilots'][model_id] = audit
        except (OSError, ValueError, KeyError, StopIteration) as exc:
            readiness['pending'].append({'item': 'model:'+model_id, 'reason': str(exc)})
    selection_path = retrieval/'CANDIDATE_SELECTION.json'
    selection = None
    try:
        selection = read_json(selection_path)
        if selection.get('split') != 'dev' or selection.get('method') != 'hybrid':
            raise ValueError('Final selection must come from the frozen Hybrid dev matrix')
    except (OSError, ValueError) as exc:
        readiness['pending'].append({'item': 'retrieval_selection', 'reason': str(exc)})
    context_selection_path = retrieval/'CONTEXT_SELECTION.json'
    context_selection = None
    try:
        context_selection = read_json(context_selection_path)
        if (not selection or context_selection.get('split')!='dev'
                or context_selection.get('candidate_selection_sha256')!=file_hash(selection_path)):
            raise ValueError('Context strategy must match frozen dev candidate selection')
        if any(context_selection['selected'].get(group) not in {'none','neighbor','parent'}
               for group in selection['selected'].values()):
            raise ValueError('Every selected group requires a declared dev context strategy')
    except (OSError,ValueError,KeyError) as exc:
        readiness['pending'].append({'item':'context_selection','reason':str(exc)})
    subset = seeded_subset(queries['test'], 100)
    route = {split: seeded_subset(queries[split], n) for split, n in [('train',200),('dev',60)]}
    groups, inputs, sources = [], {}, {}
    main, small = spec['main_models'], spec['small_models']
    selected = selection['selected'] if selection else {'simple': 'PENDING', 'structured': 'PENDING'}
    def path_for(prefix, structure, method):
        group = selected[structure]
        expansion = context_selection['selected'].get(group,'PENDING') if context_selection else 'PENDING'
        return str(retrieval/'records'/f'{prefix}_{group}_{method}_{expansion}.jsonl')
    requests = []
    for method, backend in [('R0','bm25'),('R1','dense'),('R2','hybrid'),('R3','hybrid_rerank')]:
        requests.append(('E2', method, main, queries['test'], path_for('TEST_CONTEXT','structured',backend)))
    for method in ['oracle','no_context']:
        requests.append(('E3', method, main, queries['test'], None))
    squad_queries = []
    try:
        sq, squad_units, squad_artifact = verified_corpus(corpus/'squad_e4')
        if any(units[uid]!=squad_units[uid] for uid in set(units)&set(squad_units)):
            raise ValueError('Conflicting evidence unit IDs across datasets')
        squad_queries = sq['test']; units.update(squad_units); sources['squad_split'] = squad_artifact
        lengths.update(document_lengths(corpus/'squad_e4'))
    except (OSError, ValueError, KeyError) as exc:
        readiness['pending'].append({'item':'E4_population','reason':str(exc)})
    requests.append(('E4','hybrid',main,squad_queries,path_for('E4_CONTEXT','structured','hybrid')))
    for method in ['bm25','hybrid']:
        requests.append(('E5',method,[spec['heterogeneous_model']],subset,path_for('TEST_CONTEXT','structured',method)))
    for structure in ['simple','structured']:
        requests.append(('E6',structure,small,subset,path_for('TEST_CONTEXT',structure,'hybrid')))
    for split, population in route.items():
        for method in ['bm25','hybrid','hybrid_rerank']:
            requests.append(('ROUTE_'+split.upper(),method,[spec['smallest_model']],population,
                             path_for('ROUTE_CONTEXT_'+split.upper(),'structured',method)))
    for experiment, method, requested_models, population, default_path in requests:
        key = experiment+':'+method
        base = {'experiment_id':experiment,'method':method,'model_ids':requested_models,'expected_n':len(population) or (100 if experiment=='E4' else 0)}
        block = explicit_block(spec.get('blocks',{}).get(key))
        if block:
            groups.append({**base,**block}); readiness['blocks'][key]=block; continue
        # A missing model creates its own full-N blocked condition; runnable models remain.
        active = [mid for mid in requested_models if mid in clients]
        for mid in requested_models:
            if 'model:'+mid in readiness['blocks']:
                groups.append({**base,'model_ids':[mid],**readiness['blocks']['model:'+mid]})
        if not active:
            if not all('model:'+mid in readiness['blocks'] for mid in requested_models):
                groups.append({**base,'status':'PENDING','reason':'Model pilot/identity prerequisites not ready'})
            continue
        try:
            if method == 'oracle' and experiment == 'E3':
                source_rows = [{'query':q,'context_bundle':asdict(oracle_context(q,units))} for q in population]
            elif method == 'no_context' and experiment == 'E3':
                source_rows = [{'query':q,'context_bundle':asdict(pack_context([]))} for q in population]
            else:
                path = spec.get('context_sources',{}).get(key,default_path)
                source_rows, audit = verified_context_rows(path,population); sources[key]=audit
            ids = []
            # Every main-model input uses all available main tokenizers, even an
            # extension currently executed on only one or two of those models.
            common_clients = [clients[mid] for mid in main if mid in clients]
            common_clients += [clients[mid] for mid in active if mid not in main]
            for source in source_rows:
                input_id = key+':'+source['query']['query_id']
                inputs[input_id] = shared_context(source,common_clients)
                ids.append(input_id)
            groups.append({**base,'model_ids':active,'input_ids':ids,'status':'READY'})
        except (OSError,ValueError,KeyError) as exc:
            readiness['pending'].append({'item':key,'reason':str(exc)})
            groups.append({**base,'model_ids':active,'status':'PENDING','reason':str(exc)})
    # E7 is synthetic-position diagnosis: same exact source/noise pieces, chosen
    # from a deterministic answerable population before observing model results.
    e7_block = explicit_block(spec.get('blocks',{}).get('E7'))
    if e7_block:
        for position in ['start','middle','end']:
            groups.append({'experiment_id':'E7','method':position,'model_ids':small,'expected_n':40,**e7_block})
        readiness['blocks']['E7']=e7_block
    else:
        ready_small = [mid for mid in small if mid in clients]
        if ready_small:
            if spec.get('e7_inputs'):
                fixture_audit = read_json(spec['e7_inputs'])
            else:
                fixture_audit = make_position_fixtures(queries['test'],units,[clients[mid] for mid in ready_small])
            fixtures = fixture_audit['fixtures']
            test = {q['query_id']:q for q in queries['test']}
            if not 0<len(fixtures)<=40 or len({row['query_id'] for row in fixtures})!=len(fixtures):
                raise ValueError('E7 must declare up to forty unique answerable queries before model results')
            variant_ids = {position:[] for position in ['start','middle','end']}
            for row in fixtures:
                q = test[row['query_id']]
                if q['answerability']!='answerable': raise ValueError('E7 requires answerable queries')
                for piece in row['evidence_pieces']+row['noise_pieces']:
                    uid=piece['unit_spans'][0]['unit_id']; unit=units[uid]
                    if piece['text']!=unit['text'] or piece['doc_id']!=unit['doc_id'] or piece['canonical_start']!=unit['canonical_start'] or piece['canonical_end']!=unit['canonical_end']:
                        raise ValueError('E7 text/provenance differs from exact canonical evidence unit')
                checks = [lambda text,c=clients[mid],q=q:c.prompt_token_counter(messages(q['question'],text))+512<=c.num_ctx for mid in small if mid in clients]
                variants = position_contexts(row['evidence_pieces'],row['noise_pieces'],model_budget_checks=checks)
                for position,bundle in variants.items():
                    iid = 'E7:'+position+':'+q['query_id']; inputs[iid]={'query':q,'context_bundle':asdict(bundle),'synthetic_diagnosis':True}; variant_ids[position].append(iid)
            for position,ids in variant_ids.items():
                groups.append({'experiment_id':'E7','method':position,'model_ids':ready_small,'expected_n':len(fixtures),'input_ids':ids,'status':'READY'})
                for mid in small:
                    if 'model:'+mid in readiness['blocks']:
                        groups.append({'experiment_id':'E7','method':position,'model_ids':[mid],'expected_n':len(fixtures),**readiness['blocks']['model:'+mid]})
            sources['E7']=fixture_audit
        elif not all('model:'+mid in readiness['blocks'] for mid in small):
            readiness['pending'].append({'item':'E7','reason':'At least one verified small-model token counter is required for position feasibility'})
            for position in ['start','middle','end']:
                groups.append({'experiment_id':'E7','method':position,'model_ids':small,'expected_n':40,
                               'status':'PENDING','reason':'Verified small-model token counter pending'})
        else:
            for position in ['start','middle','end']:
                for mid in small:
                    groups.append({'experiment_id':'E7','method':position,'model_ids':[mid],'expected_n':40,**readiness['blocks']['model:'+mid]})
    for source in inputs.values():
        query=source['query']
        doc_id=query.get('doc_id') or query.get('metadata',{}).get('gold_doc_id_evaluation_only')
        source['document_length']=lengths.get(doc_id,{'doc_id':doc_id,'reference_n_tokens':None,'length_group':None,
            'reason':'Canonical document length metadata unavailable; never substitute retrieved context length'})
    out = Path(spec['output_dir']); out.mkdir(parents=True,exist_ok=True)
    readiness.update(groups=groups, frozen_subset_ids=[q['query_id'] for q in subset],
        routing_query_ids={split:[q['query_id'] for q in rows] for split,rows in route.items()},
        source_artifacts=sources, spec_hash=signature(spec))
    atomic_json(out/'FREEZE_READINESS.json',readiness)
    if readiness['pending']:
        return readiness
    if not selection:
        raise ValueError('No final dev selection')
    snapshot=out/'SERVICE_RUNTIME_SNAPSHOT.json'
    if not snapshot.exists():
        atomic_json(snapshot,service_receipt)
    elif read_json(snapshot).get('environment',{}).get('LLAMA_ARG_CACHE_RAM')!='0':
        raise ValueError('Frozen service runtime receipt differs')
    preview_tasks=[{'experiment_id':group['experiment_id'],'method':group['method'],'model_id':mid,
                    'query_id':inputs[iid]['query']['query_id']} for group in groups if group['status']=='READY'
                    for mid in group['model_ids'] for iid in group['input_ids']]
    validate_core_matrix({'groups':groups,'tasks':preview_tasks},[q['query_id'] for q in queries['test']],main)
    manifest = freeze_campaign(out,models,groups,inputs,provenance={'spec':spec,'readiness':readiness,
        'external_campaigns':spec.get('external_campaigns',[]),
        'corpus_manifest':corpus_artifact,'selection':artifact(selection_path),'selection_result':selection,
        'context_selection':artifact(context_selection_path),'context_selection_result':context_selection,
        'backend_runtime_config':RUNTIME_POLICY,'service_runtime_snapshot':artifact(snapshot),
        'latency_population':'Formal runs only, measured under frozen runtime policy; pilot latencies excluded',
        'prompt_version':PROMPT_VERSION,'prompt_hash':signature([RAG_SYSTEM_PROMPT,RAG_USER_TEMPLATE]),
        'output_constraint_version':OUTPUT_CONSTRAINT_VERSION,
        'citation_visibility_interpretation':'Citation ID visibility constrained by backend schema; not a model faithfulness score',
        'oracle_unanswerable_definition':'Empty context by experimental construction, not detected answerability',
        'subset_rule':'query_id sorted then seeded shuffle42 firstN; never selected by test outputs'})
    frozen_manifest,frozen_inputs=load_campaign(out)
    list(iter_record_sources(out,frozen_manifest,frozen_inputs))
    validate_core_matrix(manifest,[q['query_id'] for q in queries['test']],main)
    readiness['status']='FROZEN'; readiness['manifest_hash']=manifest['manifest_hash']
    atomic_json(out/'FREEZE_READINESS.json',readiness)
    return readiness


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec'); parser.add_argument('--campaign',default='20261014_v2')
    parser.add_argument('--write-default-spec')
    args=parser.parse_args()
    spec=read_json(args.spec) if args.spec else default_spec(runtime_home(),args.campaign)
    if args.write_default_spec:
        atomic_json(Path(args.write_default_spec),spec); return 0
    report=freeze_all(spec)
    print(json.dumps({'status':report['status'],'pending':report['pending'],'groups':len(report['groups'])},ensure_ascii=False),flush=True)
    return 0 if report['status']=='FROZEN' else 2


if __name__=='__main__':
    raise SystemExit(main())
