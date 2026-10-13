"""Live retrieval over frozen, already built indexes; no gold or model loading.

The factory verifies completed selection/index artifacts. Retrieval accepts only
question text, explicit user scope and a route. Model services/rerankers must be
provided by the caller; importing this module never downloads or loads weights.
"""
from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from time import perf_counter

from ragscope.generation.context import pack_context
from ragscope.indexing.array_index import ArrayIndex
from ragscope.indexing.identity import artifact, file_hash, signature, verify_artifacts
from ragscope.indexing.local_embedding import HFJsonTokenizer, LocalEmbeddingService


ROUTES = ("bm25", "hybrid", "hybrid_rerank")


class RetrievalNotReady(RuntimeError):
    """Required frozen evidence or completed index is absent."""


def _read(path):
    if not path.is_file():
        raise RetrievalNotReady("Required frozen artifact is not ready: " + str(path))
    return json.loads(path.read_text(encoding="utf-8"))


def _scope(scope, index):
    if not isinstance(scope, dict):
        raise ValueError("Scope must be an explicit document/global mapping")
    allowed = {"kind", "doc_ids", "doc_id", "corpus", "corpus_doc_ids_hash"}
    if set(scope) - allowed:
        raise ValueError("Scope contains unsupported metadata; gold is not retrieval input")
    if scope.get("kind") == "global":
        if scope.get("doc_ids") or scope.get("doc_id"):
            raise ValueError("Global scope cannot also name a document")
        return None, {key: deepcopy(value) for key, value in scope.items()}
    if scope.get("kind", "document") not in {"document", "paper"}:
        raise ValueError("Unsupported scope kind")
    ids = scope.get("doc_ids", [scope["doc_id"]] if "doc_id" in scope else [])
    if not isinstance(ids, list) or len(ids) != 1 or not isinstance(ids[0], str):
        raise ValueError("Paper retrieval requires exactly one explicit document")
    if "doc_id" in scope and scope["doc_id"] != ids[0]:
        raise ValueError("Conflicting document scope")
    if ids[0] not in index.by_doc:
        raise ValueError("Selected document is absent from the frozen corpus")
    return ids[0], {"kind": "document", "doc_ids": list(ids)}


class LiveRetriever:
    """Single-worker measured retrieval with a caller-owned CPU reranker."""

    def __init__(self, index, parents, *, expansion, identity, routes=ROUTES,
                 embedding=None, reranker=None):
        if not routes or len(set(routes)) != len(routes) or not set(routes) <= set(ROUTES):
            raise ValueError("Invalid live retrieval routes")
        if expansion not in {"none", "neighbor", "parent"}:
            raise ValueError("Invalid frozen expansion")
        if set(routes) & {"hybrid", "hybrid_rerank"}:
            if embedding is None or index.vectors is None:
                raise RetrievalNotReady("Hybrid requires a completed dense index and live embedding service")
            if embedding.identity != index.identity.get("embedding") or embedding.identity.get("device") != "cpu":
                raise ValueError("Live embedding identity must match the frozen CPU index")
        if "hybrid_rerank" in routes:
            if reranker is None:
                raise RetrievalNotReady("HybridRerank requires the caller's verified CPU reranker")
            rid = getattr(reranker, "identity", {})
            if (getattr(reranker, "device", None) != "cpu" or rid.get("device") != "cpu"
                    or not rid.get("revision") or rid.get("aggregation") != "windowed_max"
                    or not isinstance(rid.get("max_length"), int) or not 4 <= rid["max_length"] <= 512):
                raise ValueError("Reranker must have pinned CPU/query-inclusive window identity")
            actual_device = getattr(getattr(reranker, "model", None), "device", None)
            if actual_device is not None and str(actual_device).split(":", 1)[0] != "cpu":
                raise ValueError("Injected reranker model actually resides outside CPU")
        self.index, self.expansion = index, expansion
        self.embedding, self.reranker = embedding, reranker
        self.routes, self.identity = tuple(routes), deepcopy(identity)
        self.load_timings_ms = {}
        self.chunk_lookup = {c["chunk_id"]: c for c in index.chunks}
        self.parent_lookup = {p["chunk_id"]: p for p in parents}
        if expansion == "parent" and any(c.get("parent_id") not in self.parent_lookup for c in index.chunks):
            raise RetrievalNotReady("Frozen parent expansion is missing parent source records")

    def retrieve(self, question, scope, route, *, model_budget_checks=()):
        """Return real candidates, a ContextBundle and timings; preserve failures.

        Query encoding time includes actual cache lookup; cache_hit observations
        distinguish that path from inference. Index/model load time is separate
        factory/caller work and must not be described as included here.
        """
        if not isinstance(question, str) or not question.strip():
            raise ValueError("A nonempty question string is required; query/gold objects are prohibited")
        if route not in self.routes:
            raise ValueError("Route was not enabled at factory construction")
        model_budget_checks = tuple(model_budget_checks)
        index_scope, visible_scope = _scope(scope, self.index)
        started = perf_counter()
        timings = {"query_encoding_ms": 0.0, "search_ms": 0.0, "rerank_ms": 0.0, "context_ms": 0.0}
        result = {"route": route, "scope": visible_scope, "question": question,
                  "experiment_type": "online_retrieval", "retrieval_identity": deepcopy(self.identity),
                  "index_signature": self.index.signature, "raw_retrieval": [], "selected_retrieval": [],
                  "context_bundle": {}, "retrieval_context_bundle": {},
                  "rerank_windows": [], "query_encoding_metrics": [],
                  "embedding_identity": self.embedding.identity if route != "bm25" else None,
                  "reranker_identity": self.reranker.identity if route == "hybrid_rerank" else None,
                  "error": None, "candidate_depth": 50, "final_k": 10}
        try:
            vector = None
            if route != "bm25":
                before = perf_counter()
                try:
                    vector = self.embedding.embed_query(question)
                finally:
                    timings["query_encoding_ms"] = (perf_counter()-before)*1000
                    result["query_encoding_metrics"] = deepcopy(self.embedding.last_metrics)
            before = perf_counter()
            try:
                candidates = self.index.search(question, method="bm25" if route == "bm25" else "hybrid",
                    query_vector=vector, embedding_identity=self.embedding.identity if route != "bm25" else None,
                    scope=index_scope, k=50, candidate_depth=50)
            finally:
                timings["search_ms"] = (perf_counter()-before)*1000
                timings.update(deepcopy(getattr(self.index, "last_timings", {})))
            result["raw_retrieval"] = deepcopy(candidates)
            result["actual_candidate_count"] = len(candidates)
            if route == "hybrid_rerank":
                before = perf_counter()
                try:
                    selected = self.reranker.rerank(question, candidates, top_k=10)
                finally:
                    timings["rerank_ms"] = (perf_counter()-before)*1000
                    result["rerank_windows"] = deepcopy(self.reranker.last_window_records)
                source = {c["chunk_id"]: c for c in candidates}
                if len(selected) > 10 or len({c["chunk_id"] for c in selected}) != len(selected):
                    raise ValueError("Reranker violated the unique top10 output contract")
                for chunk in selected:
                    if chunk["chunk_id"] not in source or any(chunk[key] != source[chunk["chunk_id"]][key]
                        for key in ("text", "doc_id", "canonical_start", "canonical_end", "unit_spans")):
                        raise ValueError("Reranker changed source provenance")
            else:
                selected = candidates[:10]
            result["selected_retrieval"] = deepcopy(selected)
            before = perf_counter()
            try:
                initial = pack_context(selected, max_tokens=2048,
                    reference_tokenizer="cl100k_base", scope="all" if index_scope is None else visible_scope,
                    expansion=self.expansion, chunk_lookup=self.chunk_lookup,
                    parent_lookup=self.parent_lookup).to_dict()
                timings["reference_context_ms"] = (perf_counter()-before)*1000
                result["retrieval_context_bundle"] = initial
                if model_budget_checks:
                    # Match freezer.shared_context exactly: a reference-only
                    # pass first, then repack only its retained pieces. A single
                    # pass could admit a previously excluded lower-ranked piece
                    # after rejecting an earlier piece on a model token budget.
                    before_common = perf_counter()
                    try:
                        if not all(check("") for check in model_budget_checks):
                            raise ValueError("Question plus prompt/output budget exceeds a model context before evidence")
                        result["context_bundle"] = pack_context(initial["pieces"], max_tokens=2048,
                            scope=visible_scope, model_budget_checks=model_budget_checks).to_dict()
                    finally:
                        timings["common_model_pack_ms"] = (perf_counter()-before_common)*1000
                else:
                    result["context_bundle"] = initial
                    timings["common_model_pack_ms"] = 0.0
            finally:
                timings["context_ms"] = (perf_counter()-before)*1000
        except Exception as exc:
            result["error"] = {"type": type(exc).__name__, "message": str(exc)}
        timings["total_ms"] = (perf_counter()-started)*1000
        result["timings_ms"] = timings
        observations = result["query_encoding_metrics"]
        flags = [row["cache_hit"] for row in observations if "cache_hit" in row]
        result["query_cache_state"] = ("not_used" if route == "bm25" else "unknown" if not flags else
                                       "hit" if all(flags) else "miss" if not any(flags) else "mixed")
        return result


def create_live_retriever(home, campaign="20261014_v2", *, family="structured",
                          embedding_url=None, embedding_identity_path=None,
                          reranker=None, routes=ROUTES):
    """Load existing verified QASPER indexes for the final dev-selected family.

    A dense ArrayIndex already contains the same BM25 corpus/statistics, so when
    neural routes are enabled one verified instance serves all routes. Its real
    dense manifest identity is reported even for BM25; no alternate signature is
    claimed. This avoids retaining two copies of the corpus and sparse counters.
    """
    started = perf_counter()
    if not routes or len(set(routes)) != len(routes) or not set(routes) <= set(ROUTES):
        raise ValueError("Invalid live retrieval routes")
    home = Path(home)
    run = home / "runs" / (campaign + "_retrieval")
    candidate_path, context_path = run / "CANDIDATE_SELECTION.json", run / "CONTEXT_SELECTION.json"
    candidates, contexts = _read(candidate_path), _read(context_path)
    if family not in {"simple", "structured"} or set(candidates.get("selected", {})) != {"simple", "structured"}:
        raise ValueError("A final simple/structured family must be selected")
    if candidates.get("split") != "dev" or candidates.get("method") != "hybrid":
        raise ValueError("Online retrieval requires final Hybrid dev selection")
    if (contexts.get("split") != "dev" or contexts.get("method") != "hybrid"
            or contexts.get("test_used_for_selection") is not False
            or contexts.get("candidate_selection_sha256") != file_hash(candidate_path)):
        raise ValueError("Frozen context/candidate selection provenance differs")
    group = candidates["selected"][family]
    expansion = contexts["selected"].get(group)
    evidence = contexts.get("evidence", {}).get(group, [])
    if not evidence or not verify_artifacts(evidence):
        raise RetrievalNotReady("Completed dev expansion evidence is missing or modified")
    corpus = home / "corpora" / campaign
    data_manifest_path = corpus / "SPLIT_MANIFEST.json"
    data_manifest = _read(data_manifest_path)
    if candidates.get("data_manifest_sha256") != file_hash(data_manifest_path):
        raise ValueError("Selected data split identity differs")
    corpus_hash = data_manifest["files"]["documents.jsonl"]["sha256"]
    if file_hash(corpus / "documents.jsonl") != corpus_hash:
        raise ValueError("Frozen corpus checksum differs")
    matched = []
    for path in (home / "indexes" / campaign / "all_chunks").glob("*/chunk_manifest.json"):
        manifest = _read(path)
        ident = manifest["identity"]
        if (ident.get("group", {}).get("id") == group and ident.get("data_manifest_sha256") == file_hash(data_manifest_path)
                and ident.get("corpus_sha256") == corpus_hash and ident.get("split") == "frozen_all_splits_corpus"
                and ident.get("scope") == "global_or_explicit_paper"):
            matched.append((path.parent, manifest))
    if len(matched) != 1:
        raise RetrievalNotReady("Exactly one completed selected full-corpus chunk index is required")
    directory, chunk_manifest = matched[0]
    if (chunk_manifest["signature"] != signature(chunk_manifest["identity"])
            or not verify_artifacts(chunk_manifest["artifacts"])):
        raise ValueError("Frozen chunk/parent artifacts changed")
    project = Path(__file__).resolve().parents[2]
    config = _read(project / "configs/v2/retrieval.json")
    bm25 = {**config["bm25"], "idf_scope": "frozen_509_paper_corpus"}
    index_identity = {**chunk_manifest["identity"], "engine": "ArrayIndex-v2", "bm25": bm25, "embedding": None}
    embedding = None
    if set(routes) & {"hybrid", "hybrid_rerank"}:
        if not embedding_url:
            raise RetrievalNotReady("Explicit live loopback embedding endpoint is required")
        identity_path = Path(embedding_identity_path) if embedding_identity_path else home / "runs" / campaign / "embedding_identity.json"
        model_identity = _read(identity_path)
        if model_identity.get("device") != "cpu":
            raise ValueError("Online embedding must be the pinned CPU identity")
        index_identity.update(embedding=model_identity, embedding_input=config["embedding_input"])
        index_directory = directory / ("dense_" + signature(model_identity)[:16])
        _read(index_directory / "manifest.json")  # Fail before tokenizer/cache work if unfinished.
        tokenizer_path = home / "models/qwen3-embedding-0.6b/tokenizer.json"
        tokenizer_identity = chunk_manifest["identity"]["tokenizer"]
        if file_hash(tokenizer_path) != tokenizer_identity["tokenizer_file_sha256"]:
            raise ValueError("Frozen tokenizer checksum differs")
        tokenizer = HFJsonTokenizer(tokenizer_path, tokenizer_identity["tokenizer_id"], tokenizer_identity["tokenizer_revision"])
        embedding = LocalEmbeddingService(embedding_url, model_identity, tokenizer,
            home / "cache/ragscope/v2_embeddings", max_input_tokens=config["embedding_total_token_limit"])
    else:
        index_directory = directory / "sparse"
        _read(index_directory / "manifest.json")
    index = ArrayIndex.load(index_directory, index_identity)
    parents = []
    with (directory / "parents.jsonl").open(encoding="utf-8") as handle:
        parents = [json.loads(line) for line in handle if line.strip()]
    frozen_config = {**config, "split": "test", "scope": "explicit_query_scope", "bm25": bm25}
    identity = {"kind": "live_frozen_retrieval", "data_campaign": campaign, "family": family,
                "group": group, "expansion": expansion, "candidate_depth": 50, "final_k": 10,
                "reference_tokens": 2048, "index_signature": index.signature,
                "retrieval_config": frozen_config, "retrieval_config_hash": signature(frozen_config),
                "artifacts": [artifact(p) for p in (candidate_path, context_path, index_directory / "manifest.json")],
                "no_gold_inputs": True}
    retriever = LiveRetriever(index, parents, expansion=expansion, identity=identity,
                              routes=routes, embedding=embedding, reranker=reranker)
    retriever.load_timings_ms = {"index_load_ms": (perf_counter()-started)*1000}
    return retriever
