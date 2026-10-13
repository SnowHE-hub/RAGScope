"""Materialize real R3 from immutable Hybrid-50 records, never truncated top10."""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path
import random
import sys
import time

from ragscope.indexing.identity import artifact, atomic_json, file_hash, signature, verify_artifacts
from ragscope.experiments.ledger import Campaign, RecordStore, read_records


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def jsonl(path):
    with Path(path).open(encoding="utf-8") as reader:
        return [json.loads(line) for line in reader if line.strip()]


def frozen_jobs(home, campaign):
    """Use all frozen dev/test IDs and the parent's explicit routing manifest."""
    home = Path(home)
    run = home / "runs" / (campaign + "_retrieval")
    corpus = home / "corpora" / campaign
    split = read(corpus / "SPLIT_MANIFEST.json")
    selection_path, context_path = run / "CANDIDATE_SELECTION.json", run / "CONTEXT_SELECTION.json"
    selection, context = read(selection_path), read(context_path)
    if (selection.get("split") != "dev" or selection.get("method") != "hybrid"
            or context.get("split") != "dev" or context.get("candidate_selection_sha256") != file_hash(selection_path)):
        raise ValueError("Final dev candidate/context choices are not frozen consistently")
    group = selection["selected"]["structured"]
    expansion = context["selected"][group]
    if expansion not in {"none", "neighbor", "parent"}:
        raise ValueError("Unknown frozen context strategy")
    routing_path = home / "runs" / campaign / "ROUTING_QUERY_MANIFEST.json"
    routing = read(routing_path)
    if (routing.get("manifest_hash") != signature({k:v for k,v in routing.items() if k!="manifest_hash"})
            or routing.get("source") != artifact(corpus/"SPLIT_MANIFEST.json") or routing.get("seed") != 42):
        raise ValueError("Routing IDs must come from the hash-verified frozen corpus manifest")
    jobs = []
    populations = [("E1", "dev", 100), ("TEST_CONTEXT", "test", 200), ("E8", "global_test", 100),
                   ("ROUTE_CONTEXT_TRAIN", "train", 200), ("ROUTE_CONTEXT_DEV", "dev", 60)]
    for prefix, split_name, n in populations:
        path = corpus / f"queries_{split_name}.jsonl"
        # Global population is a separately frozen view of the test corpus.
        expected_sha = split["files"].get(path.name, {}).get("sha256")
        if not expected_sha or file_hash(path) != expected_sha:
            raise ValueError("Frozen query file identity missing/changed: " + str(path))
        queries = jsonl(path)
        if prefix.startswith("ROUTE_"):
            ids = routing["query_ids"][split_name]
            if len(ids) != n or len(set(ids)) != n:
                raise ValueError("Routing query manifest denominator mismatch")
            by_id = {q["query_id"]: q for q in queries}
            queries = [by_id[qid] for qid in ids]
        if len(queries) != n:
            raise ValueError("Fixed R3 population denominator mismatch: " + prefix)
        jobs.append({"prefix": prefix, "group_id": group, "expansion": expansion, "queries": queries,
                     "source": run / "records" / f"{prefix}_{group}_hybrid_{expansion}.jsonl",
                     "destination": run / "records" / f"{prefix}_{group}_hybrid_rerank_{expansion}.jsonl",
                     "provenance": [artifact(selection_path), artifact(context_path), artifact(path), artifact(routing_path)]})
    return jobs


def verified_source(job):
    path = job["source"]
    config = read(path.with_suffix(".config.json"))
    if (config["method"] != "hybrid" or config["group"]["id"] != job["group_id"]
            or config["expansion"] != job["expansion"] or config["retrieval"]["candidate_depth"] != 50
            or config["retrieval"]["final_k"] != 10 or config["retrieval"]["context"]["max_tokens"] != 2048):
        raise ValueError("R3 requires final Hybrid-50 source with fixed ten-output/context settings")
    rows = read_records(path)
    wanted = {q["query_id"]: q for q in job["queries"]}
    if len(rows) != len(wanted) or {r["query_id"] for r in rows} != set(wanted):
        raise ValueError("Hybrid source omits or adds frozen population members")
    for row in rows:
        query = wanted[row["query_id"]]
        if (row.get("error") or row.get("real_model") is not True or not row.get("record_hash")
                or row.get("config_hash") != signature(config) or row.get("record_id") != row["query_id"]
                or row["question"] != query["question"] or row["scope"] != query["scope"] or row["split"] != query["split"]
                or row.get("index_signature") != signature(config["index_identity"])):
            raise ValueError("Hybrid source row is failed or mismatches frozen input/identity")
    return config, rows


def find_index_material(home, campaign, identity):
    """Read only an exactly matching manifest; no vector/model/BM25 construction."""
    root = Path(home) / "indexes" / campaign
    matches = []
    for path in root.rglob("manifest.json"):
        info = read(path)
        if info.get("identity") == identity and info.get("signature") == signature(identity):
            matches.append((path, info))
    if len(matches) != 1:
        raise ValueError("Expected one exact Hybrid source index manifest")
    path, manifest = matches[0]
    if (not manifest.get("artifacts") or not verify_artifacts(manifest["artifacts"])
            or any(Path(a["path"]).resolve().parent != path.parent.resolve() for a in manifest["artifacts"])):
        raise ValueError("Source index artifact integrity/path failure")
    chunk_manifest_path = path.parent.parent / "chunk_manifest.json"
    chunk_manifest = read(chunk_manifest_path)
    if not verify_artifacts(chunk_manifest["artifacts"]):
        raise ValueError("Canonical chunks/parents changed")
    if any(identity.get(k) != v for k, v in chunk_manifest["identity"].items()):
        raise ValueError("Index/chunk provenance mismatch")
    chunks = read(path.parent / "chunks.json")
    parents = jsonl(path.parent.parent / "parents.jsonl")
    if len(chunks) != manifest["count"] or len({c["chunk_id"] for c in chunks}) != len(chunks):
        raise ValueError("Index chunk count/uniqueness mismatch")
    return chunks, parents, [artifact(path), artifact(chunk_manifest_path)]


def restore_candidates(row, chunks):
    lookup = {c["chunk_id"]: c for c in chunks}
    scope = row["scope"]
    allowed = None if scope.get("kind") == "global" else set(scope["doc_ids"])
    eligible = sum(allowed is None or c["doc_id"] in allowed for c in chunks)
    raw = row["raw_retrieval"]
    if len(raw) != min(50, eligible) or len({c["chunk_id"] for c in raw}) != len(raw):
        raise ValueError("Raw candidates are not the actual full Hybrid50 (or all available scoped chunks)")
    restored = []
    for rank, hit in enumerate(raw, 1):
        source = lookup[hit["chunk_id"]]
        if (hit.get("rank") != rank or not math.isfinite(hit["score"])
                or any(source[k] != hit[k] for k in ("doc_id", "canonical_start", "canonical_end", "unit_spans"))
                or (allowed is not None and source["doc_id"] not in allowed)):
            raise ValueError("Raw candidate identity/coordinates/scope changed")
        restored.append(dict(source, score=hit["score"], hybrid_rank=rank))
    return restored, eligible


def rerank_record(source, query, chunks, parents, units, reranker, expansion):
    from ragscope.generation.context import pack_context
    from ragscope.evaluation.evidence import context_statistics
    from ragscope.experiments.retrieval_campaign import score_retrieval
    started = time.perf_counter()
    candidates, eligible = restore_candidates(source, chunks)
    selected, bundle, error = [], {}, None
    timings = {}
    try:
        tick = time.perf_counter()
        selected = reranker.rerank(query["question"], candidates, top_k=10)
        timings["rerank_ms"] = (time.perf_counter()-tick)*1000
        tick = time.perf_counter()
        bundle = pack_context(selected, max_tokens=2048, scope=query["scope"], expansion=expansion,
                              chunk_lookup={c["chunk_id"]: c for c in chunks},
                              parent_lookup={p["chunk_id"]: p for p in parents}).to_dict()
        timings["context_ms"] = (time.perf_counter()-tick)*1000
    except Exception as exc:
        error = {"type": type(exc).__name__, "message": str(exc)}
    timings["materialization_ms"] = (time.perf_counter()-started)*1000
    # Composite counterfactual costs are labelled, never called measured online latency.
    pre_rerank = sum(source.get("timings", {}).get(key, 0) or 0 for key in ("query_encoding_ms", "retrieval_ms"))
    timings["total_ms"] = pre_rerank + timings["materialization_ms"]
    selected_raw = [{k: c[k] for k in ("chunk_id", "doc_id", "score", "canonical_start", "canonical_end", "unit_spans",
                                      "rerank_score", "rerank_windows", "hybrid_rank")} | {"rank": rank}
                    for rank, c in enumerate(selected, 1)]
    gold = query.get("doc_id", query.get("metadata", {}).get("gold_doc_id_evaluation_only"))
    scores = score_retrieval({**query, "doc_id": gold}, selected, bundle, units, failed=bool(error))
    fixture = bool(reranker.identity.get("offline_fixture", False))
    return {"query_id":query["query_id"], "question":query["question"], "doc_id":gold, "scope":query["scope"],
            "split":query["split"], "answerability":query["answerability"], "experiment_type":"retrieval",
            "real_model":not fixture, "device":"cpu", "model_identity":reranker.identity, "reranker_identity":reranker.identity,
            "embedding_identity":source["embedding_identity"], "index_signature":source["index_signature"],
            "raw_retrieval":copy.deepcopy(source["raw_retrieval"]), "selected_retrieval":selected_raw,
            "retrieval_record_schema_version":2, "candidate_count":len(candidates), "scoped_chunk_count":eligible,
            "candidate_shortfall_reason":"scope_has_fewer_than_50_chunks" if eligible<50 else None,
            "rerank_windows":copy.deepcopy(reranker.last_window_records), "context_bundle":bundle,
            "context_statistics":context_statistics(bundle) if bundle else {}, "scores":scores, "error":error,
            "timings":timings, "source_retrieval_timings":source.get("timings",{}), "seed":42,
            "latency_interpretation":"source encoding/search cost plus separately measured rerank/context; not online end-to-end",
            "source_record_id":source["record_id"], "source_record_hash":source["record_hash"]}


def run_job(job, chunks, parents, units, reranker, campaign, index_artifacts, *, reserve_bytes=1280*1024**2):
    from ragscope.experiments.retrieval_campaign import summarize_run
    import psutil
    config, rows = verified_source(job)
    run_id = job["destination"].stem
    run_config = {"run_id":run_id, "method":"hybrid_rerank", "source_config":config,
                  "index_identity":config["index_identity"],"retrieval":config["retrieval"],"group":config["group"],
                  "source_artifacts":[artifact(job["source"]),artifact(job["source"].with_suffix(".config.json"))],
                  "index_artifacts":index_artifacts, "selection_provenance":job["provenance"],
                  "reranker_identity":reranker.identity, "window_aggregation":"windowed_max",
                  "candidate_depth":50, "final_k":10, "expansion":job["expansion"], "max_context_tokens":2048,
                  "query_ids":[q["query_id"] for q in job["queries"]]}
    store = RecordStore(job["destination"], run_config, real_only=True, max_records=len(rows))
    summary_path = campaign.path / "summaries" / (run_id+".json")
    if reranker.identity.get("offline_fixture"):
        raise ValueError("A functional fixture cannot enter the formal R3 RecordStore")
    if not campaign.begin(run_id,run_config,[sys.executable,*sys.argv]):return read(summary_path)
    queries = {q["query_id"]:q for q in job["queries"]}
    try:
        warm_ids=sorted(queries)
        random.Random(42).shuffle(warm_ids)
        warm_ids=warm_ids[:3]
        warm=RecordStore(campaign.path/"warmups"/(run_id+".jsonl"),
                         {"formal_config_hash":signature(run_config),"query_ids":warm_ids,
                          "policy":"three real reranker warmups excluded from reported latency"},real_only=True,max_records=3)
        sources={row["query_id"]:row for row in rows}
        if any(row.get("error") for row in warm.records):
            raise ValueError("A recorded warmup failed; preserve it and require explicit bounded diagnosis")
        for qid in warm_ids:
            if qid in warm.ids:continue
            if psutil.virtual_memory().available<reserve_bytes:raise MemoryError("R3 host RAM reserve before warmup")
            candidates,_=restore_candidates(sources[qid],chunks)
            tick=time.perf_counter()
            result,error=[],None
            try:result=reranker.rerank(queries[qid]["question"],candidates,top_k=10)
            except Exception as exc:error={"type":type(exc).__name__,"message":str(exc)}
            warm.add(qid,{"real_model":True,"model_identity":reranker.identity,"purpose":"warmup",
                          "candidate_count":len(candidates),"selected_count":len(result),
                          "seconds":time.perf_counter()-tick,"window_count":len(reranker.last_window_records),"error":error})
            if error:raise RuntimeError("Real reranker warmup failed; durable error preserved")
        consecutive_failures=0
        for previous in reversed(store.records):
            if not previous.get("error"):break
            consecutive_failures+=1
        if consecutive_failures>=3:
            raise RuntimeError("Three consecutive stored R3 failures require bounded diagnosis before continuation")
        for source in rows:
            if source["query_id"] in store.ids:continue
            if psutil.virtual_memory().available<reserve_bytes:raise MemoryError("R3 host RAM reserve")
            row = rerank_record(source,queries[source["query_id"]],chunks,parents,units,reranker,job["expansion"])
            row.update(campaign_id=campaign.campaign_id,run_id=run_id,
                       actual_process_rss=psutil.Process(os.getpid()).memory_info().rss,
                       host_available_bytes=psutil.virtual_memory().available)
            store.add(source["query_id"],row)
            consecutive_failures=consecutive_failures+1 if row["error"] else 0
            print(json.dumps({"run_id":run_id,"completed":len(store.ids),"expected":len(rows),
                              "candidates":row["candidate_count"],"selected":len(row["selected_retrieval"]),
                              "windows":len(row["rerank_windows"]),"error":row["error"]}),flush=True)
            if psutil.virtual_memory().available<reserve_bytes:raise MemoryError("R3 RAM reserve after committed query")
            if consecutive_failures>=3:
                raise RuntimeError("Three consecutive real R3 failures; all rows retained and remaining population pending")
    except MemoryError:
        task=campaign.state["tasks"][run_id]
        task.update(status="PENDING",attempts=max(0,task["attempts"]-1))
        campaign.event("rerank_resource_pause",task_id=run_id,committed=len(store.ids),expected=len(rows))
        campaign.save()
        raise
    summary={**summarize_run(store.records),"run_id":run_id,"method":"hybrid_rerank","expansion":job["expansion"],
             "group":config["group"],"split":job["queries"][0]["split"],"selection_eligible":False,
             "index_signature":signature(config["index_identity"]),
             "reranker_identity":reranker.identity,"actual_raw_candidates":[r["candidate_count"] for r in store.records],
             "latency_interpretation":"composite offline costs; real online latency must be measured separately"}
    atomic_json(summary_path,summary)
    campaign.finish(run_id,"PASS" if not summary["n_failed"] else "FAILED",
                    [store.path,store.path.with_suffix(".config.json"),summary_path,warm.path,warm.path.with_suffix(".config.json")],summary)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--home",default="D:/RAGScope_Runtime")
    parser.add_argument("--campaign",default="20261014_v2")
    parser.add_argument("--run",action="store_true")
    parser.add_argument("--prefix",action="append")
    args=parser.parse_args()
    home=Path(args.home)
    readiness={"status":"PENDING","model_loaded":False,"jobs":[],"pending":[]}
    try:
        jobs=frozen_jobs(home,args.campaign)
        if args.prefix:jobs=[job for job in jobs if job["prefix"] in args.prefix]
        if not jobs:raise ValueError("No requested frozen R3 jobs")
        for job in jobs:
            try:
                config,rows=verified_source(job)
                readiness["jobs"].append({"prefix":job["prefix"],"n":len(rows),"source":str(job["source"]),
                                          "destination":str(job["destination"]),"status":"READY"})
            except (OSError,ValueError,KeyError) as exc:
                readiness["pending"].append({"prefix":job["prefix"],"reason":str(exc)})
    except (OSError,ValueError,KeyError) as exc:
        readiness["pending"].append({"reason":str(exc)})
    readiness["status"]="READY" if not readiness["pending"] else "PENDING"
    destination=home/"runs"/(args.campaign+"_retrieval")/"R3_READINESS.json"
    atomic_json(destination,readiness)
    if readiness["pending"] or not args.run:
        print(json.dumps(readiness,ensure_ascii=False),flush=True)
        return 2 if readiness["pending"] else 0
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="":raise ValueError("CPU runtime environment required")
    import psutil
    if psutil.virtual_memory().available<3072*1024**2:raise MemoryError("R3 requires 3 GiB available before model load")
    from ragscope.retrieval.verified_reranker import load_verified_reranker
    # Owning the existing retrieval campaign lock prevents concurrent writers.
    with Campaign(home,args.campaign+"_retrieval") as campaign:
        reranker=load_verified_reranker(home)
        documents=jsonl(home/"corpora"/args.campaign/"documents.jsonl")
        units={u["unit_id"]:u for doc in documents for u in doc["evidence_units"]}
        del documents
        failures=0
        for job in jobs:
            config,_=verified_source(job)
            chunks,parents,artifacts=find_index_material(home,args.campaign,config["index_identity"])
            result=run_job(job,chunks,parents,units,reranker,campaign,artifacts)
            failures+=result["n_failed"]
            del chunks,parents
    return 1 if failures else 0


if __name__=="__main__":raise SystemExit(main())
