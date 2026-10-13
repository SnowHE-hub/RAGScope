"""Frozen local generation campaigns: serial model execution and global budget.

This module starts no servers/downloads. Root orchestration supplies a verified
local client after model pilot and lease checks. E2/E3 are reserved first.
"""
from __future__ import annotations
import copy
import json
import random
import re
import time
from dataclasses import asdict
from pathlib import Path

from filelock import FileLock

from ragscope.evaluation.v2 import score_generation, summarize_records
from ragscope.experiments.ledger import RecordStore, append_jsonl, read_records
from ragscope.generation.context import pack_context, validate_bundle, fingerprint
from ragscope.generation.pipeline import RAGPipeline
from ragscope.indexing.identity import atomic_json, signature, verify_artifacts


class BudgetExhausted(RuntimeError):
    kind = "generation_budget_exhausted"


def _safe_id(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", value):
        raise ValueError("Unsafe campaign model/experiment identifier")
    return value


class GenerationBudget:
    """One shared 10,000-call ceiling across pilots, failures, warmups and studies.

    A reserved interrupted call stays counted, never silently refunded. Core
    E2/E3 reservations protect their remaining planned calls from extensions.
    """
    def __init__(self, path, limit=10000, core_reserved=3600):
        if not 0 < limit <= 10000 or not 0 <= core_reserved <= limit:
            raise ValueError("Invalid global call budget")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = FileLock(str(self.path)+".lock", timeout=10)
        with self.lock:
            if self.path.exists():
                state = json.loads(self.path.read_text(encoding="utf-8"))
                if state["limit"] != limit or state["initial_core_reservation"] != core_reserved:
                    raise ValueError("Cannot change a live generation budget")
            else:
                atomic_json(self.path, {"limit": limit, "initial_core_reservation": core_reserved,
                                       "reserved_calls": 0, "core_calls": 0})

    def reserve(self, *, purpose, query_id, model_identity):
        with self.lock:
            state = json.loads(self.path.read_text(encoding="utf-8"))
            core = purpose in {"E2", "E3"}
            remaining_core = max(0, state["initial_core_reservation"]-state["core_calls"])
            effective_limit = state["limit"] if core else state["limit"]-remaining_core
            if state["reserved_calls"] >= effective_limit:
                raise BudgetExhausted("Global generation ceiling or reserved E2/E3 budget reached")
            state["reserved_calls"] += 1
            state["core_calls"] += int(core)
            atomic_json(self.path, state)
            ticket = {"call_number": state["reserved_calls"], "purpose": purpose, "query_id": query_id,
                      "model_identity": model_identity, "reserved_at_unix": time.time()}
            append_jsonl(self.path.with_suffix(".calls.jsonl"), ticket)
            return ticket

    def state(self):
        with self.lock:
            return json.loads(self.path.read_text(encoding="utf-8"))


class CountedClient:
    """Wrap every real request, including parse failures and transport timeouts."""
    def __init__(self, client, budget, purpose, query_id):
        self.client, self.budget, self.purpose, self.query_id = client, budget, purpose, query_id
        self.last_ticket = None

    def __getattr__(self, key):
        return getattr(self.client, key)

    def chat(self, messages, **kwargs):
        self.last_ticket = None
        metadata = getattr(self.client, 'last_metadata', None)
        if isinstance(metadata, dict):
            metadata.clear()
        self.last_ticket = self.budget.reserve(purpose=self.purpose, query_id=self.query_id, model_identity=self.client.identity)
        return self.client.chat(messages, **kwargs)


def oracle_context(query, evidence_units, *, max_tokens=2048, token_counter=None, model_budget_checks=()):
    """Only source evidence text enters Oracle; never answer fields or hints."""
    if query["answerability"] == "unanswerable":
        return pack_context([], max_tokens=max_tokens, token_counter=token_counter,
                            model_budget_checks=model_budget_checks)
    references = query.get("evidence_reference_sets", [])
    if not references or not references[0]:
        raise ValueError("Oracle requires legitimate textual evidence")
    pieces = []
    for uid in dict.fromkeys(references[0]):
        unit = evidence_units[uid]
        a, b = unit["canonical_start"], unit["canonical_end"]
        pieces.append({"chunk_id": "oracle:"+uid, "doc_id": unit["doc_id"], "text": unit["text"],
                       "canonical_start": a, "canonical_end": b,
                       "unit_spans": [{"doc_id": unit["doc_id"], "unit_id": uid, "start": a, "end": b}]})
    bundle = pack_context(pieces, max_tokens=max_tokens, token_counter=token_counter,
                          scope=query.get("scope"), model_budget_checks=model_budget_checks)
    # Oracle budget loss is visible; do not silently advertise a full oracle.
    return bundle


def position_contexts(evidence_pieces, noise_pieces, *, max_tokens=2048, token_counter=None, model_budget_checks=()):
    middle = len(noise_pieces)//2
    orderings = {"start": evidence_pieces+noise_pieces,
                 "middle": noise_pieces[:middle]+evidence_pieces+noise_pieces[middle:],
                 "end": noise_pieces+evidence_pieces}
    expected = {piece["chunk_id"]: piece["text"] for piece in evidence_pieces+noise_pieces}
    if len(expected) != len(evidence_pieces)+len(noise_pieces):
        raise ValueError("Position study requires distinct fixed pieces")
    bundles = {}
    for position, pieces in orderings.items():
        bundle = pack_context(pieces, max_tokens=max_tokens, token_counter=token_counter,
                              model_budget_checks=model_budget_checks)
        if {piece["chunk_id"]: piece["text"] for piece in bundle.pieces} != expected:
            raise ValueError("Position experiment must retain identical evidence/noise; budget/dedup changed content")
        bundles[position] = bundle
    if len({bundle.budget["reference_tokens"] for bundle in bundles.values()}) != 1:
        raise ValueError("Position experiment must preserve identical reference-token length")
    return bundles


def freeze_campaign(output_dir, models, groups, inputs, *, seed=42, generation_config=None, provenance=None):
    """Freeze inputs once; each group lists explicit input_ids and model_ids.

    inputs[id] = {query,context_bundle,raw_retrieval,context_config,...}.
    Models carry the full verified backend/weight identity. All mandatory groups,
    including currently blocked ones, must be listed rather than dropped.
    """
    generation_config = generation_config or {"max_tokens": 512, "temperature": 0.0}
    for model_id, identity in models.items():
        _safe_id(model_id)
        if not all(identity.get(key) for key in ("model", "revision", "weight_sha256", "backend_version")):
            raise ValueError("Frozen model requires pinned weights and backend identity")
    validated = {}
    for input_id, source in inputs.items():
        row = copy.deepcopy(source)
        validate_bundle(row["context_bundle"], scope=row["query"].get("scope"))
        row["input_id"] = input_id
        row["input_hash"] = signature(row)
        validated[input_id] = row
    tasks, ids, logical_ids = [], set(), set()
    for group in groups:
        _safe_id(group["experiment_id"])
        if group.get("status") == "BLOCKED":
            if not group.get("reason") or not group.get("expected_n"):
                raise ValueError("Blocked groups need explicit evidence/reason and expected denominator")
            continue
        for model_id in group["model_ids"]:
            if model_id not in models:
                raise ValueError("Unknown frozen model")
            for input_id in group["input_ids"]:
                if input_id not in validated:
                    raise ValueError("Missing frozen input")
                query = validated[input_id]["query"]
                task = {"experiment_id": group["experiment_id"], "method": group["method"],
                        "model_id": model_id, "input_id": input_id, "query_id": query["query_id"],
                        "input_hash": validated[input_id]["input_hash"]}
                task["task_id"] = signature(task)
                logical = (task["experiment_id"], task["method"], model_id, query["query_id"])
                if task["task_id"] in ids or logical in logical_ids:
                    raise ValueError("Duplicate experiment task")
                ids.add(task["task_id"])
                logical_ids.add(logical)
                tasks.append(task)
    if len(tasks) > 10000:
        raise ValueError("Frozen tasks exceed total real-call ceiling before warmups/retries")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = {"schema_version": "generation-campaign-v2", "models": models, "groups": groups,
              "seed": seed, "generation_config": generation_config, "metric_version": "generation-evidence-v2",
              "provenance": provenance or {},
              "reference_set_rule": "best_fullset_then_fullunit_then_continuous",
              "parse_repair_attempts": 0, "tasks": tasks, "input_hashes": {key: row["input_hash"] for key, row in validated.items()}}
    config["manifest_hash"] = signature(config)
    path = output_dir / "EXPERIMENT_MANIFEST.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != config:
            raise ValueError("Immutable campaign inputs/config changed; create another run")
    else:
        atomic_json(output_dir / "FROZEN_INPUTS.json", validated)
        atomic_json(path, config)
    saved = json.loads((output_dir / "FROZEN_INPUTS.json").read_text(encoding="utf-8"))
    if saved != validated:
        raise ValueError("Saved frozen inputs differ")
    return config


def load_campaign(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "EXPERIMENT_MANIFEST.json").read_text(encoding="utf-8"))
    expected = manifest.pop("manifest_hash")
    if signature(manifest) != expected:
        raise ValueError("Campaign manifest hash mismatch")
    manifest["manifest_hash"] = expected
    inputs = json.loads((directory / "FROZEN_INPUTS.json").read_text(encoding="utf-8"))
    if set(inputs) != set(manifest["input_hashes"]):
        raise ValueError("Frozen input membership changed")
    for input_id, source in inputs.items():
        body = {key: val for key, val in source.items() if key != "input_hash"}
        if signature(body) != manifest["input_hashes"][input_id] or source["input_hash"] != manifest["input_hashes"][input_id]:
            raise ValueError("Frozen context/query content changed")
        validate_bundle(source["context_bundle"], scope=source["query"].get("scope"))
    return manifest, inputs


def iter_record_sources(directory, manifest, inputs):
    """Yield independently owned sources without rewriting a shard's records.

    Every descriptor contains directory, manifest, inputs and owned_task_ids.
    External ownership covers unfinished tasks too: the main runner must never
    silently re-run an incomplete externally assigned task in its own journal.
    """
    directory = Path(directory).resolve()
    main_tasks = {task['task_id']: task for task in manifest['tasks']}
    owned, paths, external = set(), {directory}, []
    for reference in manifest.get('provenance', {}).get('external_campaigns', []):
        receipt = reference['manifest']
        if not verify_artifacts([receipt]):
            raise ValueError('External campaign manifest artifact changed')
        path = Path(receipt['path']).resolve()
        if path.name != 'EXPERIMENT_MANIFEST.json' or path.parent in paths:
            raise ValueError('External campaign cycle or duplicate source')
        paths.add(path.parent)
        source_manifest, source_inputs = load_campaign(path.parent)
        if source_manifest.get('provenance', {}).get('external_campaigns'):
            raise ValueError('Nested external generation campaigns are prohibited')
        if source_manifest['generation_config'] != manifest['generation_config']:
            raise ValueError('External generation configuration differs')
        for field in ('metric_version', 'reference_set_rule', 'parse_repair_attempts', 'seed'):
            if source_manifest.get(field) != manifest.get(field):
                raise ValueError(f'External experiment semantics differ: {field}')
        for field in ('prompt_version', 'prompt_hash', 'output_constraint_version', 'backend_runtime_config'):
            if source_manifest.get('provenance', {}).get(field) != manifest.get('provenance', {}).get(field):
                raise ValueError(f'External generation protocol differs: {field}')
        task_ids = set()
        for task in source_manifest['tasks']:
            task_id = task['task_id']
            if task_id not in main_tasks or task != main_tasks[task_id]:
                raise ValueError('External task semantics differ from the main frozen task')
            if task_id in owned:
                raise ValueError('External task ownership overlaps')
            if source_manifest['models'][task['model_id']] != manifest['models'][task['model_id']]:
                raise ValueError('External model identity differs')
            if source_inputs[task['input_id']] != inputs[task['input_id']]:
                raise ValueError('External input content differs')
            task_ids.add(task_id)
        if not task_ids:
            raise ValueError('Empty external campaign cannot own tasks')
        owned.update(task_ids)
        external.append({'directory': path.parent, 'manifest': source_manifest,
                         'inputs': source_inputs, 'owned_task_ids': task_ids})
    yield {'directory': directory, 'manifest': manifest, 'inputs': inputs,
           'owned_task_ids': set(main_tasks)-owned}
    yield from external


def read_source_records(source):
    """Read hash-verified owned formal rows; missing external rows stay missing."""
    manifest, inputs = source['manifest'], source['inputs']
    tasks = {task['task_id']: task for task in manifest['tasks'] if task['task_id'] in source['owned_task_ids']}
    rows = []
    for experiment, model_id in sorted({(t['experiment_id'], t['model_id']) for t in tasks.values()}):
        path = source['directory']/'records'/f'{experiment}_{model_id}.jsonl'
        if not path.exists():
            continue
        config = {'manifest_hash': manifest['manifest_hash'], 'model_id': model_id, 'experiment_id': experiment}
        if json.loads(path.with_suffix('.config.json').read_text(encoding='utf-8')) != config:
            raise ValueError('External/local record configuration differs')
        seen = set()
        for row in read_records(path):
            task_id = row['task_id']
            if task_id not in tasks or task_id in seen:
                raise ValueError('Duplicate or unowned source task record')
            task = tasks[task_id]
            if (not row.get('record_hash') or row['record_id'] != task_id or row['config_hash'] != signature(config)
                    or any(row.get(key) != value for key, value in task.items())
                    or row.get('real_model') is not True or row.get('model_identity') != manifest['models'][model_id]
                    or row.get('context_bundle') != inputs[task['input_id']]['context_bundle']
                    or not row.get('call_ticket')):
                raise ValueError('Source record identity/context/hash differs')
            seen.add(task_id)
            rows.append(row)
    return rows


def run_model(directory, model_id, client, budget, *, evidence_units, warmups=3, max_tasks=None, token_counter=None):
    """Run one externally loaded model serially. No hidden retries or downloads."""
    manifest, inputs = load_campaign(directory)
    record_sources = list(iter_record_sources(directory, manifest, inputs))
    local_owned = record_sources[0]['owned_task_ids']
    external_rows = [row for source in record_sources[1:] for row in read_source_records(source)
                     if row['model_id'] == model_id]
    external_completed = {row['task_id'] for row in external_rows}
    if warmups != 3:
        raise ValueError("Real timing runs require exactly three separately recorded warmups")
    if max_tasks is not None and max_tasks <= 0:
        raise ValueError("max_tasks must be positive")
    if client.real_model is not True or client.identity != manifest["models"][model_id]:
        raise ValueError("Real loaded model differs from frozen weight/backend identity")
    tasks = [task for task in manifest["tasks"] if task["model_id"] == model_id]
    # Reserve core experiments before extensions, deterministically random within priority.
    random.Random(manifest["seed"]).shuffle(tasks)
    tasks.sort(key=lambda task: 0 if task["experiment_id"] in {"E2", "E3"} else 1)
    directory = Path(directory)
    completed, results, stop_reason, backend_failures = 0, {}, None, 0
    backend_error_kinds = {'transport_error','timeout','oom','device_policy_error','service_not_ready'}
    with FileLock(str(directory / "generation-writer.lock"), timeout=0):
        stores = {}
        for task in tasks:
            tag = task["experiment_id"]
            if tag not in stores:
                stores[tag] = RecordStore(directory / "records" / f"{tag}_{model_id}.jsonl",
                    {"manifest_hash": manifest["manifest_hash"], "model_id": model_id, "experiment_id": tag}, real_only=True)
        pending = [task for task in tasks if task['task_id'] in local_owned
                   and task["task_id"] not in stores[task["experiment_id"]].ids]
        # Validate references before making costly requests. A bad evidence mapping
        # must not lose a real output during scoring or silently alter its denominator.
        for task in pending:
            source = inputs[task["input_id"]]
            score_generation(source["query"], {"error": {"kind": "preflight"},
                             "context_bundle": source["context_bundle"]}, evidence_units=evidence_units)
        if pending:
            warmup_store = RecordStore(directory / "records" / f"warmup_{model_id}.jsonl",
                {"manifest_hash": manifest["manifest_hash"], "model_id": model_id, "purpose": "warmup"}, real_only=True)
            session = str(time.time_ns())
            source = inputs[pending[0]["input_id"]]
            for number in range(3):
                counted = CountedClient(client, budget, "warmup", source["query"]["query_id"])
                pipeline = RAGPipeline(None, counted, use_cache=False, config=manifest["generation_config"], token_counter=token_counter)
                result = pipeline.answer(source["query"]["question"], context_bundle=source["context_bundle"], formal_evaluation=True)
                if counted.last_ticket is None:
                    stop_reason = result.error or {"kind": "request_not_started"}
                    break
                warmup_store.add(f"{session}:{number}", {**asdict(result), "purpose": "warmup", "call_ticket": counted.last_ticket})
                backend_failures = backend_failures+1 if result.error and result.error['kind'] in backend_error_kinds else 0
                if result.error and (result.error['kind']=='oom' or backend_failures>=3):
                    stop_reason={'kind':'backend_circuit_break','last_error':result.error,'consecutive_failures':backend_failures}
                    break
        for task in pending:
            if stop_reason is not None:
                break
            if max_tasks is not None and completed >= max_tasks:
                break
            source = inputs[task["input_id"]]
            query = source["query"]
            counted = CountedClient(client, budget, task["experiment_id"], query["query_id"])
            pipeline = RAGPipeline(None, counted, use_cache=False, config=manifest["generation_config"], token_counter=token_counter)
            result = pipeline.answer(query["question"], context_bundle=source["context_bundle"],
                                     scope=query.get("scope"), formal_evaluation=True)
            if counted.last_ticket is None:
                stop_reason = result.error or {"kind": "request_not_started"}
                break
            scores = score_generation(query, result, evidence_units=evidence_units)
            query_metadata = query.get('metadata', {})
            gold_doc_id = query.get('doc_id') or query_metadata.get('gold_doc_id_evaluation_only')
            cluster_id = query_metadata.get('article_cluster_id') or gold_doc_id
            resources = result.backend_metadata.get('resources', {
                'peak_gpu_allocated_bytes':None,'peak_gpu_reserved_bytes':None,'peak_host_rss_bytes':None,
                'actual_gpu_offload':None,'status':'NOT_MEASURED_BY_CLIENT',
                'reason':'Nonstreaming Ollama/llama.cpp response does not establish per-request GPU/RAM peaks or offload'})
            record = {**asdict(result), **task, "doc_id": gold_doc_id, 'cluster_id':cluster_id,
                      'cluster_definition':'SQuAD article' if query_metadata.get('article_cluster_id') else 'QASPER paper',
                      'query_metadata':query_metadata,'document_length':source.get('document_length',{}),
                      'resources':resources, "split": query["split"],
                      "scope": query.get("scope"), "answerability": query["answerability"], "scores": scores,
                      "raw_retrieval": source.get("raw_retrieval", []), "context_config": source.get("context_config", {}),
                      "selected_retrieval": source.get('selected_retrieval'),
                      "call_ticket": counted.last_ticket, "seed": manifest["seed"], "timing_mode": "generation_with_frozen_context"}
            stores[task["experiment_id"]].add(task["task_id"], record)
            completed += 1
            backend_failures = backend_failures+1 if result.error and result.error['kind'] in backend_error_kinds else 0
            if result.error and (result.error['kind']=='oom' or backend_failures>=3):
                stop_reason={'kind':'backend_circuit_break','last_error':result.error,'consecutive_failures':backend_failures}
            print(json.dumps({"experiment": task["experiment_id"], "model": model_id,
                              "completed_this_session": completed, "error": result.error}), flush=True)
        for tag, store in stores.items():
            if any(row['task_id'] not in local_owned for row in store.records):
                raise ValueError('Main journal contains an externally owned task')
            combined = store.records+[row for row in external_rows if row['experiment_id'] == tag]
            finished = store.ids | external_completed
            methods = sorted({task["method"] for task in tasks if task["experiment_id"] == tag})
            results[tag] = {**summarize_records(combined),
                            "n_expected": sum(task["experiment_id"] == tag for task in tasks),
                            "n_pending": sum(task["experiment_id"] == tag and task["task_id"] not in finished for task in tasks),
                            "stop_reason": stop_reason,
                            "manifest_hash": manifest["manifest_hash"], "records": str(store.path),
                            'external_sources':[str(source['directory']) for source in record_sources[1:]]}
            results[tag]["methods"] = {method: {**summarize_records([row for row in combined if row["method"] == method]),
                "n_expected": sum(task["experiment_id"] == tag and task["method"] == method for task in tasks)} for method in methods}
            atomic_json(directory / "summaries" / f"{tag}_{model_id}.json", results[tag])
        atomic_json(directory / 'summaries' / f'MODEL_{model_id}.json', {
            'status':'COMPLETE' if all(result['n_pending']==0 for result in results.values()) else 'INCOMPLETE',
            'model_id':model_id,'manifest_hash':manifest['manifest_hash'], 'groups':results,
            'n_expected':len(tasks), 'n_pending':sum(result['n_pending'] for result in results.values()),
            'stop_reason':stop_reason})
    return results


def validate_core_matrix(manifest, query_ids, model_ids):
    """Audit all 6 mandatory conditions per model, or explicit group blocks.

    Call before executing a campaign after its dev selection is frozen. A smaller
    frozen query population is allowed only when supplied explicitly here.
    """
    expected = set(query_ids)
    if not expected or len(expected) != len(query_ids) or len(expected) > 200:
        raise ValueError("Core query population must be unique, nonempty and at most 200")
    required = {"E2": {"R0", "R1", "R2", "R3"}, "E3": {"oracle", "no_context"}}
    for experiment, methods in required.items():
        for method in methods:
            for model in model_ids:
                groups = [group for group in manifest["groups"] if group["experiment_id"] == experiment
                          and group["method"] == method and model in group["model_ids"]]
                if len(groups) != 1:
                    raise ValueError(f"Missing/duplicate mandatory condition: {experiment}/{method}/{model}")
                group = groups[0]
                if group.get("status") == "BLOCKED":
                    if not group.get("reason") or group.get("expected_n") != len(expected):
                        raise ValueError("Blocked core condition needs the full frozen denominator")
                    continue
                actual = [task["query_id"] for task in manifest["tasks"] if task["experiment_id"] == experiment
                          and task["method"] == method and task["model_id"] == model]
                if len(actual) != len(expected) or set(actual) != expected:
                    raise ValueError("Mandatory conditions must share every frozen query, including failures")
    return True
