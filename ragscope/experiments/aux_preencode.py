"""Index-only preparation for frozen consumers; never opens their campaign."""
from __future__ import annotations
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import psutil
from filelock import FileLock

from ragscope.experiments.ledger import Campaign
from ragscope.experiments.retrieval_campaign import (
    build_group, chunker_for, read_jsonl, verified_dense_index, verify_corpus_files,
)
from ragscope.indexing.array_index import ArrayIndex
from ragscope.indexing.identity import artifact, atomic_json, file_hash, signature, verify_artifacts
from ragscope.indexing.local_embedding import HFJsonTokenizer, LocalEmbeddingService
from ragscope.indexing.scoped_index import ENGINE, ScopedArrayIndex
from ragscope.ingestion.chunkers import TokenizerAdapter


class RequestBoundaryGuard(LocalEmbeddingService):
    """Stop before the next request, preserving metrics for completed requests."""
    deadline = None

    def embed(self, texts):
        import numpy as np
        rows, completed = [], []
        self.last_metrics = []
        try:
            for text in texts:
                if self.deadline is not None and datetime.now(timezone.utc) >= self.deadline:
                    raise TimeoutError("Execution deadline reached before next embedding request")
                if psutil.virtual_memory().available < 1024**3:
                    raise MemoryError("Available RAM below 1 GiB at embedding request boundary")
                rows.extend(super().embed([text]))
                completed.extend(self.last_metrics)
        finally:
            self.last_metrics = completed
        return np.stack(rows) if rows else np.empty((0, self.get_dimension()), dtype=np.float32)


def frozen_job(home, campaign_id, dataset):
    """Mirror final-context identity construction, with read-only selection inputs."""
    if dataset not in {"squad", "qasper"}:
        raise ValueError("Only the two frozen auxiliary datasets are permitted")
    home = Path(home)
    project = Path(__file__).resolve().parents[2]
    corpus = home / "corpora" / campaign_id
    if dataset == "squad":
        corpus /= "squad_e4"
    verify_corpus_files(corpus, ("documents.jsonl", "queries_test.jsonl"))
    chunking = json.loads((project / "configs/v2/chunking.json").read_text(encoding="utf-8"))
    config = json.loads((project / "configs/v2/retrieval.json").read_text(encoding="utf-8"))
    config["split"], config["scope"] = "test", "explicit_query_scope"
    config["bm25"]["idf_scope"] = ("frozen_509_paper_corpus" if dataset == "qasper"
                                      else "official_squad2_dev_all_paragraphs")
    selection_path = home / "runs" / (campaign_id + "_retrieval") / "CANDIDATE_SELECTION.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    context_path = selection_path.with_name("CONTEXT_SELECTION.json")
    context = json.loads(context_path.read_text(encoding="utf-8"))
    if (selection.get("split") != "dev" or selection.get("method") != "hybrid"
            or set(selection["selected"]) != {"simple", "structured"}
            or selection["data_manifest_sha256"] != file_hash(home / "corpora" / campaign_id / "SPLIT_MANIFEST.json")
            or context.get("candidate_selection_sha256") != file_hash(selection_path)
            or context.get("split") != "dev"):
        raise ValueError("Frozen candidate/context selection mismatch")
    family = "simple" if dataset == "qasper" else "structured"
    group = next(g for g in chunking["groups"] if g["id"] == selection["selected"][family])
    documents = list(read_jsonl(corpus / "documents.jsonl"))
    base_identity = {
        "data_manifest_sha256": file_hash(corpus / "SPLIT_MANIFEST.json"),
        "corpus_sha256": file_hash(corpus / "documents.jsonl"),
        "normalization": "qasper_nfc_structure_v1" if dataset == "qasper" else "identity_unicode_v1",
        "selected_doc_ids": sorted(d["doc_id"] for d in documents),
        "split": "frozen_all_splits_corpus" if dataset == "qasper" else "official_dev_e4_frozen_corpus",
        "scope": "global_or_explicit_paper" if dataset == "qasper" else "global",
        "tokenizer": {k: chunking[k] for k in ("tokenizer_id", "tokenizer_revision", "tokenizer_file_sha256")},
    }
    index_type, area, reference_count = ArrayIndex, "squad_chunks", 0
    sources = [artifact(selection_path), artifact(context_path), artifact(corpus / "SPLIT_MANIFEST.json")]
    if dataset == "qasper":
        plan_path = home / "runs" / (campaign_id + "_simple_scope_preflight") / "SIMPLE_SCOPE_PLAN.json"
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        if (plan["status"] != "PASS" or not plan["scoped_only"] or plan["group"] != group
                or plan["query_ids"] != [q["query_id"] for q in read_jsonl(corpus / "queries_test.jsonl")]
                or not verify_artifacts(plan["source_artifacts"] + plan["validation_artifacts"])):
            raise ValueError("Frozen simple scope plan failed read-only verification")
        selected = set(plan["selected_doc_ids"])
        if signature(sorted(selected)) != plan["selected_doc_ids_sha256"]:
            raise ValueError("Frozen simple document membership changed")
        documents = [d for d in documents if d["doc_id"] in selected]
        base_identity = plan["scoped_base_identity"]
        index_type, area, reference_count = ScopedArrayIndex, "scoped_chunks", plan["full_reference_chunks"]
        sources.append(artifact(plan_path))
    return dict(dataset=dataset, documents=documents, group=group, base_identity=base_identity,
                index_type=index_type, area=area, reference_count=reference_count,
                config=config, chunking=chunking, sources=sources)


def encode_job(home, campaign_id, job, tokenizer, embedding, auxiliary):
    """Disjoint target lock plus auxiliary ledger; no main campaign mutation."""
    root = Path(home) / "indexes" / campaign_id / job["area"]
    root.mkdir(parents=True, exist_ok=True)
    expected = {**job["base_identity"], "group": job["group"],
                "chunking_signature": chunker_for(job["group"], tokenizer).chunking_signature}
    directory = root / signature(expected)
    target_lock = FileLock(str(root / (signature(expected) + ".writer.lock")), timeout=0)
    task_id = job["dataset"] + "_" + job["group"]["id"]
    config = {"index_identity": expected, "embedding_identity": embedding.identity,
              "retrieval": job["config"], "sources": job["sources"], "index_only": True}
    with target_lock:
        if not auxiliary.begin(task_id, config, ["index_only", job["dataset"]]):
            return json.loads((auxiliary.path / (task_id + "_READY.json")).read_text(encoding="utf-8"))
        try:
            chunks, parents, manifest, actual = build_group(job["documents"], job["group"], tokenizer,
                                                           root, job["base_identity"])
            if actual != directory or manifest["identity"] != expected:
                raise ValueError("Final consumer index path/identity mismatch")
            count = sum((m := json.loads(p.read_text(encoding="utf-8")))["count"] + m["parent_count"]
                        for p in (Path(home) / "indexes" / campaign_id).rglob("chunk_manifest.json"))
            if count + job["reference_count"] > job["chunking"]["max_cumulative_chunks"]:
                raise RuntimeError("Cumulative chunk budget exceeded")
            workload = {"dataset": job["dataset"], "chunks": len(chunks), "parents": len(parents),
                        "directory": str(directory), "target_lock": str(target_lock.lock_file),
                        "cached_vectors_before": sum(embedding.cache.get(c["text"]) is not None for c in chunks),
                        "body_tokens": sum(len(embedding.tokenizer.encode(c["text"])) for c in chunks),
                        "index_identity_sha256": signature(expected), "main_campaign_opened": False}
            atomic_json(auxiliary.path / (task_id + "_WORKLOAD.json"), workload)
            print(json.dumps(workload), flush=True)
            sparse = {**manifest["identity"], "engine": ENGINE if job["dataset"] == "qasper" else "ArrayIndex-v2",
                      "bm25": job["config"]["bm25"], "embedding": None}
            index, dense_path = verified_dense_index(chunks, sparse, directory, job["config"], embedding,
                                                     index_type=job["index_type"])
            output = {**workload, "status": "PASS", "dense_manifest": artifact(dense_path / "manifest.json"),
                      "chunk_manifest": artifact(directory / "chunk_manifest.json"),
                      "index_signature": signature(index.identity), "quality_experiment_rows": 0}
            path = auxiliary.path / (task_id + "_READY.json")
            atomic_json(path, output)
            auxiliary.finish(task_id, "PASS", [path, dense_path / "manifest.json", directory / "chunk_manifest.json"],
                             {"index_loaded_and_validated": True, "main_campaign_opened": False})
            return output
        except Exception as exc:
            auxiliary.finish(task_id, "BLOCKED" if isinstance(exc, (MemoryError, TimeoutError)) else "FAILED", [],
                             {"error_type": type(exc).__name__, "error": str(exc), "automatic_retry": False})
            raise


def run_indexes(home, campaign_id, deadline):
    home = Path(home)
    chunking_path = Path(__file__).resolve().parents[2] / "configs/v2/chunking.json"
    chunking = json.loads(chunking_path.read_text(encoding="utf-8"))
    tokenizer_path = home / "models/qwen3-embedding-0.6b/tokenizer.json"
    if file_hash(tokenizer_path) != chunking["tokenizer_file_sha256"]:
        raise ValueError("Pinned tokenizer hash mismatch")
    backend = HFJsonTokenizer(tokenizer_path, chunking["tokenizer_id"], chunking["tokenizer_revision"])
    tokenizer = TokenizerAdapter(backend, chunking["tokenizer_id"], chunking["tokenizer_revision"])
    identity = json.loads((home / "runs" / campaign_id / "embedding_identity.json").read_text(encoding="utf-8"))
    if identity.get("device") != "cpu":
        raise ValueError("Auxiliary preparation requires CPU identity")
    embedding = RequestBoundaryGuard("http://127.0.0.1:11436", identity, backend,
                                     home / "cache/ragscope/v2_embeddings", max_input_tokens=1100)
    embedding.deadline = deadline
    results = []
    with Campaign(home, campaign_id + "_aux_preencode") as auxiliary:
        for dataset in ("squad", "qasper"):
            job = frozen_job(home, campaign_id, dataset)
            results.append(encode_job(home, campaign_id, job, tokenizer, embedding, auxiliary))
            del job
        auxiliary.state["status"] = "PASS"
        atomic_json(auxiliary.path / "INDEX_ONLY_SUMMARY.json", {"status": "PASS", "indexes": results,
            "quality_experiment_rows": 0, "main_campaign_opened": False})
    return results
