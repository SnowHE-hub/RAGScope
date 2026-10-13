"""Bounded, resumable CPU retrieval experiments over the frozen v2 corpus."""
from __future__ import annotations

import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

from ragscope.evaluation.evidence import context_statistics, evidence_metrics, merge_intervals
from ragscope.evaluation.v2 import percentile
from ragscope.experiments.ledger import Campaign, RecordStore
from ragscope.generation.context import pack_context
from ragscope.indexing.array_index import ArrayIndex
from ragscope.indexing.identity import artifact, atomic_json, file_hash, signature, verify_artifacts
from ragscope.indexing.local_embedding import HFJsonTokenizer, LocalEmbeddingService
from ragscope.ingestion.chunkers import (FixedSizeChunker, ParentChildChunker,
    SentenceAwareChunker, StructureRecursiveChunker, TokenizerAdapter)


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def verify_corpus_files(corpus, names):
    """Reject changed frozen source bytes before any indexing or model calls."""
    corpus = Path(corpus)
    manifest = json.loads((corpus / "SPLIT_MANIFEST.json").read_text(encoding="utf-8"))
    for name in names:
        expected = manifest["files"].get(name)
        path = corpus / name
        if (not expected or not path.is_file() or file_hash(path) != expected["sha256"]
                or ("bytes" in expected and path.stat().st_size != expected["bytes"])):
            raise ValueError("Frozen corpus source hash/size differs: " + name)
    return manifest


def write_jsonl(path, rows):
    path = Path(path)
    temporary = path.with_suffix(".jsonl.part")
    path.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    if path.exists() and file_hash(path) != file_hash(temporary):
        raise ValueError("Immutable chunk output differs: " + str(path))
    if path.exists():
        temporary.unlink()
    else:
        temporary.replace(path)


def chunker_for(group, tokenizer):
    parameters = {k: v for k, v in group.items() if k not in {"id", "method"}}
    classes = {"fixed": FixedSizeChunker, "sentence": SentenceAwareChunker,
               "structure_recursive": StructureRecursiveChunker, "parent_child": ParentChildChunker}
    return classes[group["method"]](**parameters, tokenizer=tokenizer)


def validate_chunks(document, chunks, parents, group, tokenizer):
    """Check actual tokenizer budgets and total canonical coverage before indexing."""
    intervals = []
    parent_map = {p["chunk_id"]: p for p in parents}
    limit = group.get("chunk_size", group.get("max_tokens", group.get("child_tokens")))
    for chunk in chunks:
        assert chunk["doc_id"] == document["doc_id"]
        start, end = chunk["canonical_start"], chunk["canonical_end"]
        assert document["text"][start:end] == chunk["text"]
        actual = tokenizer.count(chunk["text"])
        assert actual == chunk["n_tokens"] and actual <= limit
        intervals.append((start, end))
        if chunk["parent_id"]:
            parent = parent_map[chunk["parent_id"]]
            assert parent["doc_id"] == chunk["doc_id"]
            assert parent["canonical_start"] <= start < end <= parent["canonical_end"]
    assert merge_intervals(intervals) == [(0, len(document["text"]))]
    for parent in parents:
        assert tokenizer.count(parent["text"]) <= group["parent_tokens"]


def build_group(documents, group, tokenizer, root, base_identity):
    chunker = chunker_for(group, tokenizer)
    identity = {**base_identity, "group": group, "chunking_signature": chunker.chunking_signature}
    directory = Path(root) / signature(identity)
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / "chunk_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["identity"] != identity or not verify_artifacts(manifest["artifacts"]):
            raise ValueError("Existing chunk index identity/hash mismatch")
        return list(read_jsonl(directory / "chunks.jsonl")), list(read_jsonl(directory / "parents.jsonl")), manifest, directory
    chunks, parents, doc_checks, events = [], [], [], Counter()
    started = time.perf_counter()
    preflight_start = started
    preflight_seconds = None
    for index, document in enumerate(documents):
        if isinstance(chunker, ParentChildChunker):
            children, parent_rows = chunker.chunk_with_parents(document)
        else:
            children, parent_rows = chunker.chunk(document), []
        children, parent_rows = [c.to_dict() for c in children], [p.to_dict() for p in parent_rows]
        validate_chunks(document, children, parent_rows, group, tokenizer)
        chunks.extend(children)
        parents.extend(parent_rows)
        events.update(event["reason"] for event in chunker.last_events)
        doc_checks.append({"doc_id": document["doc_id"], "source_chars": len(document["text"]),
                           "chunks": len(children), "parents": len(parent_rows), "coverage_verified": True})
        if index == min(2, len(documents)-1):
            preflight_seconds = time.perf_counter() - preflight_start
        if len(documents) > 60 and (index+1) % 25 == 0:
            print(f"{group['id']} canonical coverage verified: {index+1}/{len(documents)} documents", flush=True)
    build_seconds = time.perf_counter() - started
    write_jsonl(directory / "chunks.jsonl", chunks)
    write_jsonl(directory / "parents.jsonl", parents)
    manifest = {"identity": identity, "signature": signature(identity), "count": len(chunks),
                "parent_count": len(parents), "build_seconds": build_seconds,
                "body_token_max": max((c["n_tokens"] for c in chunks), default=0),
                "body_token_mean": sum(c["n_tokens"] for c in chunks)/max(1, len(chunks)),
                "boundary_events": dict(events), "document_checks": doc_checks,
                "preflight": {"documents": min(3, len(documents)), "seconds": preflight_seconds,
                              "split": base_identity.get("split", "unspecified"),
                              "sample_document_ids": [d["doc_id"] for d in documents[:3]],
                              "source_split_counts": dict(Counter(d.get("split", "unspecified") for d in documents[:3])),
                              "gpu_used": False},
                "artifacts": [artifact(directory / "chunks.jsonl"), artifact(directory / "parents.jsonl")]}
    atomic_json(manifest_path, manifest)
    return chunks, parents, manifest, directory


def score_retrieval(query, candidates, bundle, evidence_units, failed=False):
    applicability = query["answerability"]
    references = query["evidence_reference_sets"]
    raw = evidence_metrics(references, [s for c in candidates[:10] for s in c["unit_spans"]], applicability, evidence_units)
    context = evidence_metrics(references, bundle.get("unit_spans", []), applicability, evidence_units)
    if failed and applicability == "answerable":
        for key in ("continuous_coverage", "full_unit_recall", "full_reference_coverage"):
            raw[key] = context[key] = 0.0
    return {"raw_top10": raw, "context": context, "failed": failed,
            "doc_hit_at_10": float(query.get("doc_id") in {c["doc_id"] for c in candidates[:10]}) if applicability == "answerable" else None,
            "continuous_coverage": context["continuous_coverage"],
            "full_unit_recall": context["full_unit_recall"],
            "full_reference_coverage": context["full_reference_coverage"]}


def summarize_run(rows):
    result = {"n_total": len(rows), "n_failed": sum(bool(r.get("error")) for r in rows)}
    for key in ("continuous_coverage", "full_unit_recall", "full_reference_coverage"):
        values = [r["scores"][key] for r in rows if r["scores"][key] is not None]
        successful = [r["scores"][key] for r in rows if not r.get("error") and r["scores"][key] is not None]
        result[key] = {"overall": sum(values)/len(values) if values else None, "n_applicable": len(values),
                       "success_conditional": sum(successful)/len(successful) if successful else None}
    result["latency_ms"] = {"p50": percentile([r["timings"]["total_ms"] for r in rows], .5),
                            "p95": percentile([r["timings"]["total_ms"] for r in rows], .95)}
    result["mean_context_reference_tokens"] = sum(r["context_bundle"].get("budget", {}).get("reference_tokens", 0) for r in rows)/max(1, len(rows))
    result["mean_context_budget_use"] = result["mean_context_reference_tokens"] / 2048
    result["mean_context_pieces"] = sum(len(r["context_bundle"].get("pieces", [])) for r in rows)/max(1, len(rows))
    result["mean_context_duplicate_ratio"] = sum(r.get("context_statistics", {}).get("duplicate_ratio", 0) for r in rows)/max(1, len(rows))
    doc_hits = [r["scores"]["doc_hit_at_10"] for r in rows if r["scores"].get("doc_hit_at_10") is not None]
    result["doc_recall_at_10_chunks"] = sum(doc_hits)/len(doc_hits) if doc_hits else None
    return result


def select_candidates(summaries, tolerance=.01):
    """Prespecified dev-only engineering selection, with structural comparator."""
    chosen = {}
    for family in ("simple", "structured"):
        pool = [s for s in summaries if (s["group"]["method"] in {"fixed", "sentence"}) == (family == "simple")]
        if not pool:
            continue
        best_full = max(s["full_reference_coverage"]["overall"] for s in pool)
        pool = [s for s in pool if s["full_reference_coverage"]["overall"] >= best_full-tolerance]
        best_continuous = max(s["continuous_coverage"]["overall"] for s in pool)
        pool = [s for s in pool if s["continuous_coverage"]["overall"] >= best_continuous-tolerance]
        chosen[family] = min(pool, key=lambda s: (s["index_bytes"], s["latency_ms"]["p50"], s["group"]["id"]))["group"]["id"]
    return chosen


def select_context_strategy(summaries, tolerance=.01):
    """Choose only from actual dev rows; approximate ties prefer less expansion."""
    if not summaries or any(s.get("split") != "dev" for s in summaries):
        raise ValueError("Context choice requires actual dev summaries")
    best = max(s["full_reference_coverage"]["overall"] for s in summaries)
    eligible = [s for s in summaries if s["full_reference_coverage"]["overall"] >= best-tolerance]
    continuous = max(s["continuous_coverage"]["overall"] for s in eligible)
    eligible = [s for s in eligible if s["continuous_coverage"]["overall"] >= continuous-tolerance]
    return min(eligible, key=lambda s: ({"none": 0, "neighbor": 1, "parent": 2}[s["expansion"]],
                                      s["latency_ms"]["p50"]))["expansion"]


def freeze_context_choices(home: Path, data_campaign="20261014_v2"):
    project = Path(__file__).resolve().parents[2]
    chunking = json.loads((project / "configs/v2/chunking.json").read_text(encoding="utf-8"))
    groups = {g["id"]: g for g in chunking["groups"]}
    config = json.loads((project / "configs/v2/retrieval.json").read_text(encoding="utf-8"))
    with Campaign(home, data_campaign + "_retrieval") as campaign:
        candidate_path = campaign.path / "CANDIDATE_SELECTION.json"
        candidates = json.loads(candidate_path.read_text(encoding="utf-8"))
        if candidates.get("split") != "dev" or candidates.get("method") != "hybrid":
            raise ValueError("Context choices require completed Hybrid dev candidate selection")
        selected, evidence = {}, {}
        for group_id in candidates["selected"].values():
            strategies = ["none", "neighbor"] + (["parent"] if groups[group_id]["method"] == "parent_child" else [])
            summaries, artifacts = [], []
            for strategy in strategies:
                run_id = f"E1_{group_id}_hybrid_{strategy}"
                task = campaign.state["tasks"].get(run_id)
                if not task or not verify_artifacts(task.get("outputs", [])) or not task.get("outputs"):
                    raise ValueError("Missing or unverified completed context comparison: " + run_id)
                path = campaign.path / "summaries" / (run_id + ".json")
                summary = json.loads(path.read_text(encoding="utf-8"))
                if summary["n_total"] != 100 or summary["method"] != "hybrid":
                    raise ValueError("Context comparison has wrong frozen denominator or method")
                summaries.append(summary)
                artifacts.append(artifact(path))
            selected[group_id] = select_context_strategy(summaries, config["selection_tolerance"])
            evidence[group_id] = artifacts
        result = {"selected": selected, "split": "dev", "method": "hybrid", "test_used_for_selection": False,
                  "candidate_selection_sha256": file_hash(candidate_path), "evidence": evidence,
                  "tolerance": config["selection_tolerance"],
                  "rule": "full_reference_then_continuous_with_tolerance_then_none_neighbor_parent"}
        destination = campaign.path / "CONTEXT_SELECTION.json"
        if destination.exists() and json.loads(destination.read_text(encoding="utf-8")) != result:
            raise ValueError("Frozen context selection differs; do not mutate test context choice")
        atomic_json(destination, result)
        print(json.dumps({"context_choices": selected, "selection": str(destination)}), flush=True)
        return result


def retrieval_scope(query):
    """Derive legal retrieval input solely from the user's declared scope."""
    scope = query.get("scope", {})
    if scope.get("kind") == "global":
        return None
    ids = scope.get("doc_ids", [])
    if len(ids) != 1:
        raise ValueError("Paper task requires exactly one user-selected document")
    return ids[0]


def run_queries(index, chunks, parents, queries, units, config, group, method,
                expansion, campaign, embedding=None, experiment_id="E1"):
    run_id = f"{experiment_id}_{group['id']}_{method}_{expansion}"
    run_config = {"run_id": run_id, "index_identity": index.identity, "retrieval": config,
                  "group": group, "method": method, "expansion": expansion,
                  "query_ids": sorted(q["query_id"] for q in queries), "experiment_type": "retrieval"}
    rows_path = campaign.path / "records" / f"{run_id}.jsonl"
    store = RecordStore(rows_path, run_config, max_records=len(queries))
    summary_path = campaign.path / "summaries" / f"{run_id}.json"
    if not campaign.begin(run_id, run_config, [sys.executable, *sys.argv]):
        return json.loads(summary_path.read_text(encoding="utf-8"))
    lookup, parent_lookup = {c["chunk_id"]: c for c in chunks}, {p["chunk_id"]: p for p in parents}
    ordered = sorted(queries, key=lambda q: q["query_id"])
    random.Random(config["seed"]).shuffle(ordered)
    for query in ordered[:config["warmup_queries"]]:
        vector = embedding.embed_query(query["question"]) if embedding else None
        index.search(query["question"], method=method, query_vector=vector,
                     embedding_identity=embedding.identity if embedding else None,
                     scope=retrieval_scope(query), k=10)
    for query in ordered:
        if query["query_id"] in store.ids:
            continue
        started = time.perf_counter()
        candidates, selected_candidates, bundle, error, timings = [], [], {}, None, {}
        encoding_metrics = []
        try:
            vector = None
            if method != "bm25":
                before = time.perf_counter()
                vector = embedding.embed_query(query["question"])
                timings["query_encoding_ms"] = (time.perf_counter()-before)*1000
                encoding_metrics = list(embedding.last_metrics)
            before = time.perf_counter()
            candidates = index.search(query["question"], method=method, query_vector=vector,
                                      embedding_identity=embedding.identity if embedding else None,
                                      scope=retrieval_scope(query), k=config["candidate_depth"], candidate_depth=config["candidate_depth"])
            timings["retrieval_ms"] = (time.perf_counter()-before)*1000
            # The index owns component timing. If unavailable, retain only the
            # actually measured combined retrieval time rather than inventing it.
            timings.update(getattr(index, "last_timings", {}))
            before = time.perf_counter()
            selected_candidates = candidates[:config["final_k"]]
            bundle = pack_context(selected_candidates, max_tokens=config["context"]["max_tokens"],
                                  scope="all" if retrieval_scope(query) is None else query["scope"], expansion=expansion, chunk_lookup=lookup,
                                  parent_lookup=parent_lookup).to_dict()
            timings["context_ms"] = (time.perf_counter()-before)*1000
        except Exception as exc:
            error = {"type": type(exc).__name__, "message": str(exc)}
        timings["total_ms"] = (time.perf_counter()-started)*1000
        gold_doc = query.get("doc_id", query.get("metadata", {}).get("gold_doc_id_evaluation_only"))
        scores = score_retrieval({**query, "doc_id": gold_doc}, candidates, bundle, units, failed=bool(error))
        raw = [{"rank": rank, "chunk_id": c["chunk_id"], "doc_id": c["doc_id"], "score": c["score"],
                "canonical_start": c["canonical_start"], "canonical_end": c["canonical_end"],
                "unit_spans": c["unit_spans"]} for rank, c in enumerate(candidates, 1)]
        store.add(query["query_id"], {"campaign_id": campaign.campaign_id, "run_id": run_id,
                  "query_id": query["query_id"], "doc_id": gold_doc, "split": query["split"],
                  "scope": query["scope"], "question": query["question"], "answerability": query["answerability"],
                  "experiment_type": "retrieval", "device": "cpu", "real_model": bool(embedding),
                  "embedding_identity": embedding.identity if embedding else None,
                  "index_signature": index.signature, "raw_retrieval": raw,
                  "selected_retrieval": raw[:len(selected_candidates)], "retrieval_record_schema_version": 2,
                  "context_bundle": bundle,
                  "context_statistics": context_statistics(bundle) if bundle else {}, "scores": scores,
                  "error": error, "timings": timings, "query_encoding_metrics": encoding_metrics, "seed": config["seed"]})
        if len(store.ids) % 20 == 0:
            print(f"{run_id}: {len(store.ids)}/{len(queries)} queries", flush=True)
    summary = {**summarize_run(store.records), "run_id": run_id, "group": group,
               "method": method, "expansion": expansion, "records": str(rows_path),
               "index_signature": index.signature, "split": queries[0]["split"] if queries else "unknown",
               "selection_eligible": experiment_id == "E1"}
    atomic_json(summary_path, summary)
    campaign.finish(run_id, "PASS" if not summary["n_failed"] else "FAILED", [rows_path, summary_path],
                    {"queries": len(store.ids), "frozen_queries": len(queries), "failures": summary["n_failed"]})
    return summary


def run_campaign(home: Path, data_campaign="20261014_v2", stage="bm25", embedding_url=None,
                 embedding_identity_path=None, groups_filter=None, expansion_comparison=False):
    project = Path(__file__).resolve().parents[2]
    chunking = json.loads((project / "configs/v2/chunking.json").read_text(encoding="utf-8"))
    config = json.loads((project / "configs/v2/retrieval.json").read_text(encoding="utf-8"))
    corpus = home / "corpora" / data_campaign
    manifest = verify_corpus_files(corpus, ("documents.jsonl", "queries_dev.jsonl"))
    queries = list(read_jsonl(corpus / "queries_dev.jsonl"))
    doc_ids = {q["doc_id"] for q in queries}
    documents = [d for d in read_jsonl(corpus / "documents.jsonl") if d["doc_id"] in doc_ids]
    units = {u["unit_id"]: u for d in documents for u in d["evidence_units"]}
    tokenizer_path = home / "models/qwen3-embedding-0.6b/tokenizer.json"
    if file_hash(tokenizer_path) != chunking["tokenizer_file_sha256"]:
        raise ValueError("Pinned tokenizer hash mismatch")
    tokenizer_backend = HFJsonTokenizer(tokenizer_path, chunking["tokenizer_id"], chunking["tokenizer_revision"])
    tokenizer = TokenizerAdapter(tokenizer_backend, chunking["tokenizer_id"], chunking["tokenizer_revision"])
    embedding = None
    if stage in {"dense", "all"}:
        if not embedding_url or not embedding_identity_path:
            raise ValueError("Dense stage requires root-verified CPU server URL and identity file")
        identity = json.loads(Path(embedding_identity_path).read_text(encoding="utf-8"))
        if identity.get("device") != "cpu":
            raise ValueError("Stage two embedding identity must explicitly be CPU")
        embedding = LocalEmbeddingService(embedding_url, identity, tokenizer_backend, home / "cache/ragscope/v2_embeddings",
                                          max_input_tokens=config["embedding_total_token_limit"])
    base_identity = {"data_manifest_sha256": file_hash(corpus / "SPLIT_MANIFEST.json"),
                     "corpus_sha256": manifest["files"]["documents.jsonl"]["sha256"], "normalization": "qasper_nfc_structure_v1",
                     "selected_doc_ids": sorted(doc_ids), "split": "dev", "scope": "specified_paper",
                     "tokenizer": {k: chunking[k] for k in ("tokenizer_id", "tokenizer_revision", "tokenizer_file_sha256")}}
    selected_groups = [g for g in chunking["groups"] if not groups_filter or g["id"] in groups_filter]
    summaries, total_chunks = [], 0
    with Campaign(home, data_campaign + "_retrieval") as campaign:
        for group in selected_groups:
            print("Building/verifying " + group["id"], flush=True)
            chunk_task = "chunks_" + group["id"]
            chunk_task_started = campaign.begin(chunk_task, {"base_identity": base_identity, "group": group},
                                                 [sys.executable, *sys.argv])
            chunks, parents, chunk_manifest, directory = build_group(documents, group, tokenizer,
                               home / "indexes" / data_campaign / "dev_chunks", base_identity)
            if chunk_task_started:
                campaign.finish(chunk_task, "PASS", [directory / "chunk_manifest.json", directory / "chunks.jsonl", directory / "parents.jsonl"],
                                {"coverage_and_budgets_verified": True, "chunks": len(chunks), "parents": len(parents)})
            total_chunks += len(chunks) + len(parents)
            if total_chunks > chunking["max_cumulative_chunks"]:
                raise RuntimeError("180000 chunk budget exceeded")
            sparse_identity = {**chunk_manifest["identity"], "engine": "ArrayIndex-v2", "bm25": config["bm25"], "embedding": None}
            index_directory = directory / "sparse"
            if (index_directory / "manifest.json").exists():
                index = ArrayIndex.load(index_directory, sparse_identity)
            else:
                index = ArrayIndex(chunks, sparse_identity)
                index.save(index_directory)
            expansion = "parent" if group["method"] == "parent_child" else "none"
            if stage in {"bm25", "all"}:
                summary = run_queries(index, chunks, parents, queries, units, config, group, "bm25", expansion, campaign)
                summary.update(chunk_count=len(chunks), parent_count=len(parents), build_seconds=chunk_manifest["build_seconds"],
                               index_bytes=sum(p.stat().st_size for p in index_directory.rglob("*") if p.is_file())
                               + (directory / "parents.jsonl").stat().st_size)
                summaries.append(summary)
            if embedding:
                dense_identity = {**sparse_identity, "embedding": embedding.identity, "embedding_input": config["embedding_input"]}
                dense_directory = directory / ("dense_" + signature(embedding.identity)[:16])
                if (dense_directory / "manifest.json").exists():
                    index = ArrayIndex.load(dense_directory, dense_identity)
                else:
                    vectors = []
                    embedding_metrics_path = directory / "embedding_metrics.jsonl"
                    metrics = []
                    for offset in range(0, len(chunks), 8):
                        batch = embedding.embed([c["text"] for c in chunks[offset:offset+8]])
                        vectors.extend(batch)
                        metrics.extend({"chunk_id": c["chunk_id"], **m} for c, m in zip(chunks[offset:offset+8], embedding.last_metrics))
                        if offset % 80 == 0:
                            print(f"{group['id']} embedding: {min(offset+8, len(chunks))}/{len(chunks)}", flush=True)
                    index = ArrayIndex(chunks, dense_identity, np.stack(vectors))
                    index.save(dense_directory)
                    write_jsonl(embedding_metrics_path, metrics)
                for method in ("dense", "hybrid"):
                    summary = run_queries(index, chunks, parents, queries, units, config, group, method, expansion, campaign, embedding)
                    summary.update(chunk_count=len(chunks), parent_count=len(parents), build_seconds=chunk_manifest["build_seconds"],
                                   index_bytes=sum(p.stat().st_size for p in dense_directory.rglob("*") if p.is_file())
                                   + (directory / "parents.jsonl").stat().st_size)
                    summaries.append(summary)
            if expansion_comparison:
                for strategy in ("none", "neighbor", "parent"):
                    if strategy == "parent" and not parents:
                        continue
                    if strategy == expansion:
                        continue
                    method = "hybrid" if embedding else "bm25"
                    extra = run_queries(index, chunks, parents, queries, units, config, group, method, strategy, campaign, embedding)
                    extra.update(chunk_count=len(chunks), parent_count=len(parents), build_seconds=chunk_manifest["build_seconds"],
                                 index_bytes=sum(p.stat().st_size for p in (dense_directory if embedding else index_directory).rglob("*") if p.is_file())
                                 + (directory / "parents.jsonl").stat().st_size)
                    summaries.append(extra)
        subset_suffix = ("_" + "-".join(g["id"] for g in selected_groups)) if groups_filter else ""
        if expansion_comparison:
            subset_suffix += "_context"
        output = campaign.path / f"E1_{stage}{subset_suffix}_SUMMARY.json"
        atomic_json(output, {"data_campaign": data_campaign, "stage": stage, "groups": summaries,
                             "cumulative_chunk_records": total_chunks, "max_chunk_budget": chunking["max_cumulative_chunks"],
                             "config_hash": signature({"chunking": chunking, "retrieval": config}), "cpu_only": True})
        selection_method = "hybrid" if embedding else "bm25"
        selection_summaries = [s for s in summaries if s["method"] == selection_method
                              and s["expansion"] == ("parent" if s["group"]["method"] == "parent_child" else "none")]
        if len(selected_groups) == 11:
            candidates = select_candidates(selection_summaries, config["selection_tolerance"])
            selection = {"selected": candidates, "method": selection_method, "split": "dev",
                         "rule": config["quality_selection"], "tolerance": config["selection_tolerance"],
                         "source_summary": str(output), "data_manifest_sha256": base_identity["data_manifest_sha256"]}
            atomic_json(campaign.path / ("CANDIDATE_SELECTION.json" if embedding else "BM25_SCREEN_SELECTION.json"), selection)
        failures = sum(s["n_failed"] for s in summaries)
        campaign.state["status"] = "PASS" if not failures else "FAILED"
        print(json.dumps({"status": campaign.state["status"], "summary": str(output), "rows": sum(s["n_total"] for s in summaries),
                          "failures": failures}), flush=True)
    return summaries


def run_test_contexts(home: Path, data_campaign="20261014_v2"):
    """Prepare real BM25 test contexts and E8, without passing gold scope globally.

    This uses the already frozen BM25 screen candidates and does not feed test
    scores back into selection. Missing dense/hybrid/rerank stay explicit blocks.
    """
    import psutil
    available = psutil.virtual_memory().available
    guard = {"available_bytes": available, "minimum_start_bytes": 768*1024**2,
             "device": "cpu", "cpu_only": True, "status": "PASS" if available >= 768*1024**2 else "BLOCKED_RESOURCE"}
    project = Path(__file__).resolve().parents[2]
    corpus = home / "corpora" / data_campaign
    verify_corpus_files(corpus, ("documents.jsonl", "queries_test.jsonl", "queries_global_test.jsonl"))
    with Campaign(home, data_campaign + "_retrieval") as campaign:
        guard_path = campaign.path / "test_context_resource_guard.json"
        atomic_json(guard_path, guard)
        if guard["status"] != "PASS":
            campaign.event("test_context_waiting_ram", **guard)
            print(json.dumps(guard), flush=True)
            return guard
        selected = json.loads((campaign.path / "BM25_SCREEN_SELECTION.json").read_text(encoding="utf-8"))
        chunking = json.loads((project / "configs/v2/chunking.json").read_text(encoding="utf-8"))
        config = json.loads((project / "configs/v2/retrieval.json").read_text(encoding="utf-8"))
        config["split"] = "test"
        config["bm25"]["idf_scope"] = "frozen_509_paper_corpus"
        config["scope"] = "explicit_query_scope"
        documents = list(read_jsonl(corpus / "documents.jsonl"))
        units = {u["unit_id"]: u for d in documents for u in d["evidence_units"]}
        test_queries = list(read_jsonl(corpus / "queries_test.jsonl"))
        global_queries = list(read_jsonl(corpus / "queries_global_test.jsonl"))
        tokenizer_path = home / "models/qwen3-embedding-0.6b/tokenizer.json"
        if file_hash(tokenizer_path) != chunking["tokenizer_file_sha256"]:
            raise ValueError("Pinned tokenizer hash mismatch")
        backend = HFJsonTokenizer(tokenizer_path, chunking["tokenizer_id"], chunking["tokenizer_revision"])
        tokenizer = TokenizerAdapter(backend, chunking["tokenizer_id"], chunking["tokenizer_revision"])
        base_identity = {"data_manifest_sha256": file_hash(corpus / "SPLIT_MANIFEST.json"),
                         "corpus_sha256": file_hash(corpus / "documents.jsonl"), "normalization": "qasper_nfc_structure_v1",
                         "selected_doc_ids": sorted(d["doc_id"] for d in documents), "split": "frozen_all_splits_corpus",
                         "scope": "global_or_explicit_paper", "tokenizer": {k: chunking[k] for k in
                             ("tokenizer_id", "tokenizer_revision", "tokenizer_file_sha256")}}
        summaries, cumulative = [], 0
        for group in chunking["groups"]:
            if group["id"] not in selected["selected"].values():
                continue
            print("Building fixed all-paper corpus " + group["id"], flush=True)
            chunks, parents, chunk_manifest, directory = build_group(documents, group, tokenizer,
                             home / "indexes" / data_campaign / "all_chunks", base_identity)
            cumulative += len(chunks) + len(parents)
            stored_count = 0
            for path in (home / "indexes" / data_campaign).rglob("chunk_manifest.json"):
                stored = json.loads(path.read_text(encoding="utf-8"))
                stored_count += stored["count"] + stored["parent_count"]
            if stored_count > chunking["max_cumulative_chunks"]:
                raise RuntimeError("Campaign cumulative chunk budget exceeded")
            identity = {**chunk_manifest["identity"], "engine": "ArrayIndex-v2", "bm25": config["bm25"], "embedding": None}
            index_directory = directory / "sparse"
            if (index_directory / "manifest.json").exists():
                index = ArrayIndex.load(index_directory, identity)
            else:
                index = ArrayIndex(chunks, identity)
                index.save(index_directory)
            expansion = "parent" if group["method"] == "parent_child" else "none"
            paper_summary = run_queries(index, chunks, parents, test_queries, units, config, group,
                                        "bm25", expansion, campaign, experiment_id="TEST_CONTEXT")
            paper_summary.update(chunk_count=len(chunks), parent_count=len(parents), index_manifest=str(index_directory / "manifest.json"))
            summaries.append(paper_summary)
            # E8 uses the preselected structured candidate, never a test winner.
            if group["id"] == selected["selected"]["structured"]:
                global_summary = run_queries(index, chunks, parents, global_queries, units, config, group,
                                             "bm25", expansion, campaign, experiment_id="E8")
                global_summary.update(chunk_count=len(chunks), parent_count=len(parents), index_manifest=str(index_directory / "manifest.json"))
                summaries.append(global_summary)
            del index, chunks, parents
        result = {"status": "PARTIAL_BLOCKED", "groups": summaries, "selection_source": selected,
                  "test_used_for_selection": False, "cumulative_all_corpus_chunks": cumulative,
                  "E8_methods": {"bm25": "PASS", "dense": "BLOCKED_RESOURCE", "hybrid": "BLOCKED_RESOURCE", "hybrid_rerank": "BLOCKED_RESOURCE"},
                  "block_evidence": str(home / "runs" / data_campaign / "embedding_preflight.jsonl")}
        atomic_json(campaign.path / "TEST_CONTEXT_E8_SUMMARY.json", result)
        campaign.state["status"] = "PARTIAL_BLOCKED"
        print(json.dumps({"status": result["status"], "actual_rows": sum(r["n_total"] for r in summaries),
                          "summary": str(campaign.path / "TEST_CONTEXT_E8_SUMMARY.json")}), flush=True)
        return result


def verified_dense_index(chunks, sparse_identity, directory, config, embedding, *, index_type=ArrayIndex):
    """Reuse only matching persisted arrays; each completed text is checkpointed."""
    operation_started = time.perf_counter()
    identity = {**sparse_identity, "embedding": embedding.identity, "embedding_input": config["embedding_input"]}
    destination = directory / ("dense_" + signature(embedding.identity)[:16])
    if (destination / "manifest.json").exists():
        return index_type.load(destination, identity), destination
    destination.mkdir(parents=True, exist_ok=True)
    metric_store = RecordStore(destination / "embedding_first_observations.jsonl",
                              {"embedding_identity": embedding.identity, "index_identity": identity,
                               "observation_policy": "first_completed_encoding_or_cache_lookup_per_chunk"},
                              max_records=len(chunks))
    previously_observed = len(metric_store.ids)
    vectors, metrics = [], []
    encoding_started = time.perf_counter()
    for offset in range(0, len(chunks), 8):
        batch = chunks[offset:offset+8]
        # A later request in this batch can fail. Persist already completed
        # measurements in finally, so a restart cannot relabel their original
        # cold costs as cache-hit costs. RecordStore retains the first observed
        # completed call per chunk; no missing historical values are invented.
        try:
            vectors.extend(embedding.embed([c["text"] for c in batch]))
        finally:
            for chunk, measurement in zip(batch, embedding.last_metrics):
                if chunk["chunk_id"] not in metric_store.ids:
                    metric_store.add(chunk["chunk_id"], {"chunk_id": chunk["chunk_id"],
                        "text_sha256": signature(chunk["text"]), "experiment_type": "embedding_operation",
                        "real_model": embedding.identity.get("backend") == "llama.cpp", **measurement})
        metrics.extend({"chunk_id": c["chunk_id"], **m} for c, m in zip(batch, embedding.last_metrics))
        if offset % 80 == 0:
            print(f"Final corpus embedding: {min(offset+8, len(chunks))}/{len(chunks)}", flush=True)
    encoding_seconds = time.perf_counter() - encoding_started
    construction_started = time.perf_counter()
    index = index_type(chunks, identity, np.stack(vectors))
    construction_seconds = time.perf_counter() - construction_started
    save_started = time.perf_counter()
    index.save(destination)
    array_save_seconds = time.perf_counter() - save_started
    write_jsonl(destination / "embedding_metrics.jsonl", metrics)
    # This is the measured successful build attempt, including its actual cache
    # state. It is not a reconstruction of unobserved historical cold work.
    atomic_json(destination / "build_performance.json", {
        "identity_signature": signature(identity), "chunks": len(chunks),
        "encoding_loop_seconds": encoding_seconds,
        "array_construction_seconds": construction_seconds,
        "array_save_seconds": array_save_seconds,
        "successful_attempt_total_seconds": time.perf_counter() - operation_started,
        "previously_completed_chunk_observations": previously_observed,
        "successful_attempt_cache_hits": sum(bool(m["cache_hit"]) for m in metrics),
        "successful_attempt_model_calls": sum(not m["cache_hit"] for m in metrics),
        "first_observed_model_call_seconds": sum(m["seconds"] for m in metric_store.records if not m["cache_hit"]),
        "first_observed_cache_lookup_seconds": sum(m["seconds"] for m in metric_store.records if m["cache_hit"]),
        "interpretation": "Actual successful attempt and durable first observations; shared cache state disclosed. Not full historical cold build time.",
        "embedding_metrics": artifact(destination / "embedding_metrics.jsonl"),
        "first_observations": artifact(destination / "embedding_first_observations.jsonl")})
    return index, destination


def run_final_contexts(home: Path, data_campaign="20261014_v2", dataset="qasper",
                       embedding_url=None, embedding_identity_path=None, methods=None,
                       family=None):
    """Evaluate frozen dev-selected candidates without using test to select.

    QASPER structured uses the all-509-paper corpus for every method. The simple
    comparator encodes only the frozen test paper union and retains all-509
    sparse statistics; it rejects global retrieval. SQuAD uses all official dev
    paragraphs and its separately frozen 100 short questions.
    Neural reranking is owned by the subsequent experiment and remains pending.
    """
    import psutil
    methods = (["hybrid"] if dataset == "squad" else ["bm25", "dense", "hybrid"]) if methods is None else methods
    if dataset not in {"qasper", "squad"} or not set(methods) <= {"bm25", "dense", "hybrid"}:
        raise ValueError("Unsupported final retrieval dataset or methods")
    if not methods:
        raise ValueError("At least one real retrieval method is required")
    if family not in {None, "simple", "structured"} or (dataset == "squad" and family == "simple"):
        raise ValueError("Unsupported final retrieval family")
    available = psutil.virtual_memory().available
    if available < 768*1024**2:
        guard = {"status": "BLOCKED_RESOURCE", "available_bytes": available,
                 "minimum_start_bytes": 768*1024**2, "dataset": dataset, "cpu_only": True}
        atomic_json(home / "runs" / (data_campaign + "_retrieval") / (dataset + "_final_resource_guard.json"), guard)
        return guard
    project = Path(__file__).resolve().parents[2]
    corpus = home / "corpora" / data_campaign
    if dataset == "squad":
        corpus /= "squad_e4"
    verify_corpus_files(corpus, ("documents.jsonl", "queries_test.jsonl")
                        + (("queries_global_test.jsonl",) if dataset == "qasper" else ()))
    chunking = json.loads((project / "configs/v2/chunking.json").read_text(encoding="utf-8"))
    config = json.loads((project / "configs/v2/retrieval.json").read_text(encoding="utf-8"))
    config["split"], config["scope"] = "test", "explicit_query_scope"
    config["bm25"]["idf_scope"] = "frozen_509_paper_corpus" if dataset == "qasper" else "official_squad2_dev_all_paragraphs"
    documents = list(read_jsonl(corpus / "documents.jsonl"))
    queries = list(read_jsonl(corpus / "queries_test.jsonl"))
    global_queries = list(read_jsonl(corpus / "queries_global_test.jsonl")) if dataset == "qasper" else []
    units = {u["unit_id"]: u for d in documents for u in d["evidence_units"]}
    tokenizer_path = home / "models/qwen3-embedding-0.6b/tokenizer.json"
    if file_hash(tokenizer_path) != chunking["tokenizer_file_sha256"]:
        raise ValueError("Pinned tokenizer hash mismatch")
    backend = HFJsonTokenizer(tokenizer_path, chunking["tokenizer_id"], chunking["tokenizer_revision"])
    tokenizer = TokenizerAdapter(backend, chunking["tokenizer_id"], chunking["tokenizer_revision"])
    embedding = None
    if set(methods) & {"dense", "hybrid"}:
        if not embedding_url or not embedding_identity_path:
            raise ValueError("Real Dense/Hybrid require verified CPU embedding service")
        identity = json.loads(Path(embedding_identity_path).read_text(encoding="utf-8"))
        if identity.get("device") != "cpu":
            raise ValueError("CPU-only retrieval stage cannot load GPU backend")
        embedding = LocalEmbeddingService(embedding_url, identity, backend, home / "cache/ragscope/v2_embeddings",
                                          max_input_tokens=config["embedding_total_token_limit"])
    base_identity = {"data_manifest_sha256": file_hash(corpus / "SPLIT_MANIFEST.json"),
                     "corpus_sha256": file_hash(corpus / "documents.jsonl"),
                     "normalization": "qasper_nfc_structure_v1" if dataset == "qasper" else "identity_unicode_v1",
                     "selected_doc_ids": sorted(d["doc_id"] for d in documents),
                     "split": "frozen_all_splits_corpus" if dataset == "qasper" else "official_dev_e4_frozen_corpus",
                     "scope": "global_or_explicit_paper" if dataset == "qasper" else "global",
                     "tokenizer": {k: chunking[k] for k in ("tokenizer_id", "tokenizer_revision", "tokenizer_file_sha256")}}
    summaries = []
    with Campaign(home, data_campaign + "_retrieval") as campaign:
        selection_path = campaign.path / "CANDIDATE_SELECTION.json"
        selected = json.loads(selection_path.read_text(encoding="utf-8"))
        if selected.get("split") != "dev" or selected.get("method") != "hybrid" or set(selected["selected"]) != {"simple", "structured"}:
            raise ValueError("Final retrieval requires both frozen Hybrid dev candidates")
        if selected.get("data_manifest_sha256") != file_hash(home / "corpora" / data_campaign / "SPLIT_MANIFEST.json"):
            raise ValueError("Candidate selection points to a different frozen QASPER split")
        context_selection_path = campaign.path / "CONTEXT_SELECTION.json"
        context_selection = json.loads(context_selection_path.read_text(encoding="utf-8"))
        if context_selection.get("candidate_selection_sha256") != file_hash(selection_path) or context_selection.get("split") != "dev":
            raise ValueError("Final context choices must match frozen dev candidate selection")
        # The structured candidate supplies E2/E4/E5/E8 and routing inputs. Build
        # it first so those downstream consumers can start while the independent
        # simple comparator is encoded. Selection and per-run identities stay
        # unchanged; cache state is measured per query and never hidden.
        ordered_groups = sorted(chunking["groups"],
                                key=lambda g: g["id"] != selected["selected"]["structured"])
        for group in ordered_groups:
            if group["id"] not in selected["selected"].values():
                continue
            if family is not None and group["id"] != selected["selected"][family]:
                continue
            if dataset == "squad" and group["id"] != selected["selected"]["structured"]:
                continue
            print(f"Final {dataset} corpus: {group['id']}", flush=True)
            group_documents, group_identity, index_type = documents, base_identity, ArrayIndex
            index_area = "all_chunks" if dataset == "qasper" else "squad_chunks"
            simple_scope = None
            if dataset == "qasper" and group["id"] == selected["selected"]["simple"]:
                from ragscope.experiments.simple_scope import prepare_simple_scope
                from ragscope.indexing.scoped_index import ScopedArrayIndex
                simple_scope = prepare_simple_scope(home, data_campaign)
                if simple_scope["group"] != group or simple_scope["query_ids"] != [q["query_id"] for q in queries]:
                    raise ValueError("Simple paper-scope preflight differs from requested frozen comparator")
                selected_ids = set(simple_scope["selected_doc_ids"])
                group_documents = [d for d in documents if d["doc_id"] in selected_ids]
                group_identity, index_type = simple_scope["scoped_base_identity"], ScopedArrayIndex
                index_area = "scoped_chunks"
            chunks, parents, chunk_manifest, directory = build_group(group_documents, group, tokenizer,
                home / "indexes" / data_campaign / index_area, group_identity)
            count = sum((manifest := json.loads(p.read_text(encoding="utf-8")))["count"] + manifest["parent_count"]
                        for p in (home / "indexes" / data_campaign).rglob("chunk_manifest.json"))
            if simple_scope is not None:
                count += simple_scope["full_reference_chunks"]
            if count > chunking["max_cumulative_chunks"]:
                raise RuntimeError("Campaign cumulative chunk budget exceeded")
            sparse_identity = {**chunk_manifest["identity"], "engine": "ArrayIndex-v2", "bm25": config["bm25"], "embedding": None}
            if simple_scope is not None:
                from ragscope.indexing.scoped_index import ENGINE
                sparse_identity["engine"] = ENGINE
            sparse_directory = directory / "sparse"
            sparse = None
            if "bm25" in methods:
                if (sparse_directory / "manifest.json").exists():
                    sparse = index_type.load(sparse_directory, sparse_identity)
                else:
                    sparse = index_type(chunks, sparse_identity)
                    sparse.save(sparse_directory)
            dense, dense_directory = verified_dense_index(chunks, sparse_identity, directory, config, embedding,
                index_type=index_type) if embedding else (None, None)
            expansion = context_selection["selected"][group["id"]]
            for method in methods:
                index = sparse if method == "bm25" else dense
                index_directory = sparse_directory if method == "bm25" else dense_directory
                active_embedding = None if method == "bm25" else embedding
                summary = run_queries(index, chunks, parents, queries, units, config, group, method, expansion, campaign,
                                      active_embedding, experiment_id="TEST_CONTEXT" if dataset == "qasper" else "E4_CONTEXT")
                summary.update(chunk_count=len(chunks), parent_count=len(parents), index_manifest=str(index_directory / "manifest.json"))
                if simple_scope is not None:
                    summary.update(scoped_only=True, scoped_documents=len(group_documents),
                        sparse_reference_documents=simple_scope["full_reference_documents"],
                        simple_scope_plan=artifact(home / "runs" / (data_campaign + "_simple_scope_preflight") / "SIMPLE_SCOPE_PLAN.json"))
                summaries.append(summary)
                if global_queries and group["id"] == selected["selected"]["structured"]:
                    summary = run_queries(index, chunks, parents, global_queries, units, config, group, method, expansion, campaign,
                                          active_embedding, experiment_id="E8")
                    summary.update(chunk_count=len(chunks), parent_count=len(parents), index_manifest=str(index_directory / "manifest.json"))
                    summaries.append(summary)
            del sparse, dense, chunks, parents
        result = {"status": "PASS" if not sum(s["n_failed"] for s in summaries) else "FAILED", "dataset": dataset,
                  "methods": methods, "groups": summaries, "selection_source": selected,
                  "selection_sha256": file_hash(selection_path), "test_used_for_selection": False,
                  "context_selection": context_selection, "context_selection_sha256": file_hash(context_selection_path),
                  "cpu_only": True, "scheduled_family": family,
                  "reranking_status": "PENDING_SEPARATE_EXPERIMENT"}
        family_suffix = "_" + family if family is not None else ""
        output = campaign.path / f"FINAL_{dataset}_{'-'.join(methods)}{family_suffix}_CONTEXT_SUMMARY.json"
        atomic_json(output, result)
        print(json.dumps({"status": result["status"], "actual_rows": sum(s["n_total"] for s in summaries), "summary": str(output)}), flush=True)
        return result


def verified_routing_queries(home: Path, data_campaign="20261014_v2", expected_manifest_hash=None):
    """Use root's frozen IDs exactly and verify the same freezer subset rule."""
    from scripts.freeze_generation_v2 import seeded_subset

    corpus = home / "corpora" / data_campaign
    path = home / "runs" / data_campaign / "ROUTING_QUERY_MANIFEST.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    actual = signature({key: value for key, value in manifest.items() if key != "manifest_hash"})
    if manifest.get("manifest_hash") != actual or (expected_manifest_hash is not None and actual != expected_manifest_hash):
        raise ValueError("Frozen routing query manifest hash differs")
    source = manifest["source"]
    if (Path(source["path"]).resolve() != (corpus / "SPLIT_MANIFEST.json").resolve()
            or not verify_artifacts([source]) or manifest.get("seed") != 42):
        raise ValueError("Frozen routing split source/seed differs")
    split_manifest = json.loads((corpus / "SPLIT_MANIFEST.json").read_text(encoding="utf-8"))
    result, all_ids, doc_sets = {}, set(), {}
    for split, count in (("train", 200), ("dev", 60), ("test", 100)):
        query_path = corpus / f"queries_{split}.jsonl"
        if file_hash(query_path) != split_manifest["files"][query_path.name]["sha256"]:
            raise ValueError("Routing source query file changed")
        population = list(read_jsonl(query_path))
        mapping = {q["query_id"]: q for q in population}
        wanted = manifest["query_ids"][split]
        if len(mapping) != len(population) or len(wanted) != count or len(set(wanted)) != count:
            raise ValueError("Wrong or duplicated frozen routing denominator")
        expected_ids = [row["query_id"] for row in seeded_subset(population, count)]
        if wanted != expected_ids or set(wanted) & all_ids:
            raise ValueError("Frozen routing IDs differ from freezer subset or leak across splits")
        rows = [mapping[qid] for qid in wanted]
        if any(q["split"] != split for q in rows):
            raise ValueError("Routing query split metadata differs")
        doc_sets[split] = {retrieval_scope(q) for q in rows}
        if None in doc_sets[split]:
            raise ValueError("Routing training inputs must retain explicit paper scope")
        result[split] = rows
        all_ids.update(wanted)
    if any(doc_sets[a] & doc_sets[b] for a, b in (("train", "dev"), ("train", "test"), ("dev", "test"))):
        raise ValueError("Routing train/dev/test document leakage")
    return result, {"manifest": artifact(path), "manifest_hash": actual, "source": source,
                    "counts": {split: len(rows) for split, rows in result.items()},
                    "document_counts": {split: len(ids) for split, ids in doc_sets.items()},
                    "sampler": "scripts.freeze_generation_v2.seeded_subset", "new_sampling_performed": False}


def run_routing_contexts(home: Path, data_campaign="20261014_v2", embedding_url=None,
                         embedding_identity_path=None, expected_manifest_hash=None):
    """Materialize train200/dev60 BM25+Hybrid sources using final existing index.

    This never builds/encodes a document index or creates a replacement sample.
    The test100 selection is verified for provenance but is not executed here;
    its contexts already belong to the frozen TEST_CONTEXT200 population.
    """
    from ragscope.routing.online_retrieval import create_live_retriever

    populations, source = verified_routing_queries(home, data_campaign, expected_manifest_hash)
    live = create_live_retriever(home, data_campaign, embedding_url=embedding_url,
        embedding_identity_path=embedding_identity_path, routes=("bm25", "hybrid"))
    corpus = home / "corpora" / data_campaign
    units = {unit["unit_id"]: unit for doc in read_jsonl(corpus / "documents.jsonl") for unit in doc["evidence_units"]}
    group = live.index.identity["group"]
    summaries = []
    with Campaign(home, data_campaign + "_retrieval") as campaign:
        for split in ("train", "dev"):
            config = {**live.identity["retrieval_config"], "split": split, "routing_query_source": source}
            for method in ("bm25", "hybrid"):
                summary = run_queries(live.index, live.index.chunks, list(live.parent_lookup.values()),
                    populations[split], units, config, group, method, live.expansion, campaign,
                    live.embedding if method == "hybrid" else None, experiment_id="ROUTE_CONTEXT_" + split.upper())
                summaries.append(summary)
        result = {"status": "PASS" if not sum(s["n_failed"] for s in summaries) else "FAILED",
                  "source": source, "groups": summaries, "selected_group": group,
                  "expansion": live.expansion, "actual_rows": sum(s["n_total"] for s in summaries),
                  "shared_index_identity": live.index.identity, "index_load_timings_ms": live.load_timings_ms,
                  "test_used_for_selection": False, "test100_executed_here": False,
                  "rerank_status": "PENDING_SEPARATE_REAL_BGE_MATERIALIZATION"}
        atomic_json(campaign.path / "ROUTING_CONTEXT_SUMMARY.json", result)
        print(json.dumps({"status": result["status"], "rows": result["actual_rows"],
                          "group": group["id"], "source_manifest_hash": source["manifest_hash"]}), flush=True)
        return result
