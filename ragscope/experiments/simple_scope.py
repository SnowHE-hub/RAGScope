"""Freeze and validate the paper-scoped simple comparator without neural calls."""
from __future__ import annotations

import json
from pathlib import Path

from ragscope.experiments.ledger import Campaign
from ragscope.experiments.retrieval_campaign import build_group, read_jsonl, retrieval_scope, verify_corpus_files
from ragscope.indexing.array_index import ArrayIndex
from ragscope.indexing.identity import artifact, atomic_json, file_hash, signature, verify_artifacts
from ragscope.indexing.local_embedding import HFJsonTokenizer
from ragscope.indexing.scoped_index import ENGINE, ScopedArrayIndex
from ragscope.ingestion.chunkers import TokenizerAdapter


def prepare_simple_scope(home, data_campaign="20261014_v2"):
    home = Path(home)
    project = Path(__file__).resolve().parents[2]
    corpus = home / "corpora" / data_campaign
    verify_corpus_files(corpus, ("documents.jsonl", "queries_test.jsonl"))
    run = home / "runs" / (data_campaign + "_retrieval")
    selection_path = run / "CANDIDATE_SELECTION.json"
    context_path = run / "CONTEXT_SELECTION.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    context = json.loads(context_path.read_text(encoding="utf-8"))
    if (selection["split"] != "dev" or selection["method"] != "hybrid"
            or selection["data_manifest_sha256"] != file_hash(corpus / "SPLIT_MANIFEST.json")
            or context["candidate_selection_sha256"] != file_hash(selection_path)):
        raise ValueError("Scoped comparator requires matching frozen dev choices")
    chunking = json.loads((project / "configs/v2/chunking.json").read_text(encoding="utf-8"))
    retrieval = json.loads((project / "configs/v2/retrieval.json").read_text(encoding="utf-8"))
    group = next(g for g in chunking["groups"] if g["id"] == selection["selected"]["simple"])
    docs = list(read_jsonl(corpus / "documents.jsonl"))
    queries = list(read_jsonl(corpus / "queries_test.jsonl"))
    scopes = {retrieval_scope(q) for q in queries}
    if None in scopes or not scopes or any(q["split"] != "test" for q in queries):
        raise ValueError("Simple comparator requires only explicit frozen test paper scopes")
    selected_ids = sorted(scopes)
    if not set(selected_ids) <= {d["doc_id"] for d in docs}:
        raise ValueError("Frozen test scope is outside source corpus")
    source_artifacts = [artifact(p) for p in (selection_path, context_path,
        corpus / "SPLIT_MANIFEST.json", corpus / "documents.jsonl", corpus / "queries_test.jsonl",
        project / "configs/v2/chunking.json", project / "configs/v2/retrieval.json")]
    with Campaign(home, data_campaign + "_simple_scope_preflight") as campaign:
        plan_path = campaign.path / "SIMPLE_SCOPE_PLAN.json"
        task_id = "simple_scope_semantic_equivalence"
        started = campaign.begin(task_id,
            {"source_artifacts": source_artifacts, "scope_doc_ids": selected_ids,
             "query_ids": [q["query_id"] for q in queries]},
            ["scripts/prepare_simple_scope_v2.py", "--home", str(home), "--campaign", data_campaign])
        if plan_path.exists():
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            if (plan["status"] != "PASS" or plan["source_artifacts"] != source_artifacts
                    or not verify_artifacts(source_artifacts + plan["validation_artifacts"])
                    or plan["query_ids"] != [q["query_id"] for q in queries]
                    or plan["selected_doc_ids"] != selected_ids):
                raise ValueError("Frozen simple scope preflight differs")
            if started:
                campaign.finish(task_id, "PASS", [plan_path, campaign.path / "BM25_TEST200_PARITY.json"],
                    {"queries": len(queries), "exact_bm25_parity": True, "neural_calls": 0,
                     "prior_parity_artifacts_reverified": True})
            campaign.state["status"] = "PASS"
            return plan
        tokenizer_path = home / "models/qwen3-embedding-0.6b/tokenizer.json"
        if file_hash(tokenizer_path) != chunking["tokenizer_file_sha256"]:
            raise ValueError("Pinned chunk tokenizer changed")
        backend = HFJsonTokenizer(tokenizer_path, chunking["tokenizer_id"], chunking["tokenizer_revision"])
        tokenizer = TokenizerAdapter(backend, chunking["tokenizer_id"], chunking["tokenizer_revision"])
        base = {"data_manifest_sha256": file_hash(corpus / "SPLIT_MANIFEST.json"),
            "corpus_sha256": file_hash(corpus / "documents.jsonl"), "normalization": "qasper_nfc_structure_v1",
            "selected_doc_ids": sorted(d["doc_id"] for d in docs), "split": "frozen_all_splits_corpus",
            "scope": "global_or_explicit_paper", "tokenizer": {k: chunking[k] for k in
                ("tokenizer_id", "tokenizer_revision", "tokenizer_file_sha256")}}
        full, parents, manifest, directory = build_group(docs, group, tokenizer, campaign.path / "reference", base)
        if parents:
            raise ValueError("Simple comparator must not require parent hierarchy")
        reference = {"chunks": artifact(directory / "chunks.jsonl"),
            "chunk_manifest": artifact(directory / "chunk_manifest.json"),
            "chunk_signature": manifest["signature"], "count": len(full),
            "chunks_content_sha256": signature(full), "doc_ids": base["selected_doc_ids"]}
        scoped_base = {**base, "selected_doc_ids": selected_ids, "selected_doc_ids_sha256": signature(selected_ids),
            "split": "frozen_test_scope_union", "scope": "explicit_paper_only", "scoped_only": True,
            "sparse_reference": reference}
        bm25 = {**retrieval["bm25"], "idf_scope": "frozen_509_paper_corpus"}
        sparse_identity = {**manifest["identity"], **scoped_base, "engine": ENGINE, "bm25": bm25, "embedding": None}
        scoped_chunks = [c for c in full if c["doc_id"] in set(selected_ids)]
        full_index = ArrayIndex(full, {**manifest["identity"], "engine": "ArrayIndex-v2", "bm25": bm25, "embedding": None})
        scoped_index = ScopedArrayIndex(scoped_chunks, sparse_identity)
        compared = []
        for query in queries:
            scope = retrieval_scope(query)
            kwargs = {"scope": scope, "method": "bm25", "k": 50, "candidate_depth": 50}
            expected = full_index.search(query["question"], **kwargs)
            observed = scoped_index.search(query["question"], **kwargs)
            left = [(c["chunk_id"], c["score"]) for c in expected]
            right = [(c["chunk_id"], c["score"]) for c in observed]
            if left != right:
                raise ValueError("Scoped simple BM25 ranking/score differs for " + query["query_id"])
            compared.append({"query_id": query["query_id"], "scope": scope, "top50": left,
                "exact_rank_and_score_match": True})
        parity_path = campaign.path / "BM25_TEST200_PARITY.json"
        atomic_json(parity_path, {"status": "PASS", "actual_queries": len(compared), "rows": compared,
            "neural_calls": 0, "full_sparse_reference": reference,
            "note": "All exact paper-scope BM25 top50 IDs/scores verified; no label-driven selection."})
        plan = {"status": "PASS", "scoped_only": True, "group": group,
            "source_artifacts": source_artifacts, "validation_artifacts": [artifact(parity_path),
                reference["chunks"], reference["chunk_manifest"], *manifest["artifacts"]],
            "query_ids": [q["query_id"] for q in queries], "selected_doc_ids": selected_ids,
            "selected_doc_ids_sha256": signature(selected_ids), "scoped_base_identity": scoped_base,
            "full_reference_documents": len(docs), "scoped_documents": len(selected_ids),
            "full_reference_chunks": len(full), "scoped_chunks": len(scoped_chunks),
            "neural_vectors_avoided": len(full) - len(scoped_chunks),
            "neural_body_tokens_avoided": sum(c["n_tokens"] for c in full)-sum(c["n_tokens"] for c in scoped_chunks),
            "full_corpus_neural_claim_allowed": False, "neural_calls": 0,
            "reason": "E6 simple uses explicit frozen test paper scopes; E8/routing/UI use structured. Preserve full-509 BM25 IDF/avgdl while encoding exactly the required paper union."}
        atomic_json(plan_path, plan)
        campaign.finish(task_id, "PASS", [plan_path, parity_path],
            {"queries": len(queries), "exact_bm25_parity": True, "neural_calls": 0,
             "prior_parity_artifacts_reverified": False})
        campaign.state["status"] = "PASS"
        return plan
