"""Visible-feature routing, frozen samples and real-generation proxy labels."""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
import math
import re
from time import perf_counter

from ragscope.indexing.identity import signature

ROUTES = ("bm25", "hybrid", "hybrid_rerank")
VISIBLE_METADATA = ("document_length_tokens", "document_type", "language")


@dataclass
class RoutingRequest:
    question: str
    scope: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)

    @classmethod
    def from_query(cls, query):
        metadata = query.get("metadata", {})
        return cls(str(query["question"]), dict(query.get("scope", {})),
                   {key: metadata[key] for key in VISIBLE_METADATA if key in metadata})

    def features_text(self):
        # Scope identifiers are legal deployment input but are not learned shortcuts.
        length = self.metadata.get("document_length_tokens")
        group = "unknown" if not isinstance(length, (int, float)) else "short" if length <= 512 else "medium" if length <= 4096 else "long"
        dtype = self.metadata.get("document_type", "unknown")
        dtype = dtype if dtype in ("paper", "text", "code", "markdown", "html") else "unknown"
        language = self.metadata.get("language", "unknown")
        language = language if language in ("en", "zh", "multilingual") else "unknown"
        return f"{self.question}\nDOC_LENGTH_{group} DOC_TYPE_{dtype} DOC_LANGUAGE_{language}"


@dataclass
class RouteDecision:
    route: str
    reason: str
    probabilities: dict = field(default_factory=dict)
    fallback: bool = False
    elapsed_s: float = 0.0
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.route not in ROUTES:
            raise ValueError("Unknown retrieval route")

    def to_dict(self):
        return asdict(self)


class FixedRouter:
    def __init__(self, route="hybrid"):
        if route not in ROUTES:
            raise ValueError(route)
        self.route = route
        self.identity = {"kind": "fixed", "route": route}

    def choose(self, request: RoutingRequest):
        started = perf_counter()
        return RouteDecision(self.route, "fixed_policy", elapsed_s=perf_counter() - started)


class RuleRouter:
    """Rules use question content; question length alone never implies difficulty."""
    identity = {"kind": "rules", "version": "visible_features_v1"}

    def choose(self, request: RoutingRequest):
        started = perf_counter()
        text = request.question.casefold()
        if re.search(r"\b(compare|contrast|difference|why|combine|relationship|trade.?off)\b|比较|区别|为什么|联系", text):
            route, reason = "hybrid_rerank", "comparison_or_reasoning_terms"
        elif re.search(r'"[^"\n]+"|`[^`\n]+`|\b[A-Za-z]+_[A-Za-z_0-9]+\b|\b[A-Za-z]+\d+[A-Za-z0-9]*\b', request.question):
            route, reason = "bm25", "explicit_literal_or_identifier"
        else:
            route, reason = "hybrid", "semantic_and_lexical_default"
        return RouteDecision(route, reason, elapsed_s=perf_counter() - started)


def freeze_query_ids(queries: list[dict], counts=None, seed=42):
    counts = counts or {"train": 200, "dev": 60, "test": 100}
    result = {}
    seen = set()
    for query in queries:
        if query["query_id"] in seen:
            raise ValueError("Duplicate query IDs before routing freeze")
        seen.add(query["query_id"])
    for split, count in counts.items():
        if split not in {"train", "dev", "test"} or not isinstance(count, int) or count < 1:
            raise ValueError("Invalid routing split/count")
        pool = [query["query_id"] for query in queries if query["split"] == split]
        result[split] = sorted(pool, key=lambda qid: signature([seed, split, qid]))[:count]
    return {"ids": result, "requested_counts": counts, "actual_counts": {key: len(value) for key, value in result.items()},
            "seed": seed, "signature": signature(result)}


def proxy_quality(record):
    """F1 for answerable inputs; explicit abstention correctness for N/A F1."""
    if record.get('answerability') == 'unanswerable':
        if record.get('answer_f1') is not None:
            raise ValueError('Unanswerable answer F1 must remain N/A')
        value = record.get('correct_abstention')
        if value not in (0, 1) or isinstance(value, str):
            raise ValueError('Unanswerable proxy requires explicit abstention correctness')
    else:
        value = record.get('answer_f1')
    if not isinstance(value, (float, int)) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError('Invalid answer quality proxy')
    if record.get('error') and value != 0:
        raise ValueError('Failed generations must retain zero applicable quality')
    return float(value)


def build_proxy_labels(queries: list[dict], records: list[dict], *, split: str,
                       query_ids: list[str], expected_config_hash: str, tolerance: float = .01):
    """Normalize only real, same-generator/path-isolated records into proxy labels.

    Each record must explicitly provide path, answer_f1, evidence_coverage,
    total_latency_s, error, real_model and comparison_config_hash. The shared
    comparison hash intentionally excludes retrieval path, which is the variable.
    """
    if split not in {"train", "dev", "test"} or not expected_config_hash or not 0 <= tolerance <= 1:
        raise ValueError("Invalid proxy-label configuration")
    if len(query_ids) != len(set(query_ids)):
        raise ValueError("Duplicate frozen query IDs")
    wanted = set(query_ids)
    query_map = {query["query_id"]: query for query in queries}
    grouped = {}
    for record in records:
        qid = record["query_id"]
        if qid not in wanted:
            continue
        if record.get("real_model") is not True or record.get("comparison_config_hash") != expected_config_hash:
            raise ValueError("Router labels require real generation with the exact frozen comparison configuration")
        if record.get("split") != split or record.get("path") not in ROUTES:
            raise ValueError("Routing label split/path mismatch")
        key = (qid, record["path"])
        if key in grouped:
            raise ValueError("Duplicate query/path generation record")
        proxy_quality(record)
        for name in ("evidence_coverage", "total_latency_s"):
            value = record.get(name)
            if name == "evidence_coverage" and value is None:
                continue  # Unanswerable evidence metrics are N/A.
            if not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"Invalid {name}")
            if name != "total_latency_s" and value > 1:
                raise ValueError(f"Invalid {name}")
        grouped[key] = record
    examples = []
    for qid in query_ids:
        query = query_map.get(qid)
        if query is None or query["split"] != split:
            raise ValueError("Frozen query is absent or has a different split")
        choices = [grouped.get((qid, route)) for route in ROUTES]
        if any(record is None for record in choices):
            raise ValueError(f"Missing real route records for {qid}; no synthetic labels substituted")
        if any(record.get('answerability', 'answerable') != query.get('answerability', 'answerable') for record in choices):
            raise ValueError('Route/query answerability differs')
        best_f1 = max(proxy_quality(record) for record in choices)
        tied = [record for record in choices if best_f1 - proxy_quality(record) <= tolerance + 1e-12]
        # Cost precedes evidence when F1 is within the declared engineering tolerance.
        selected = min(tied, key=lambda record: (record["total_latency_s"],
                       -(record["evidence_coverage"] if record["evidence_coverage"] is not None else -1),
                       ROUTES.index(record["path"])))
        examples.append({"query_id": qid, "split": split, "question": query["question"],
                         "scope": query.get("scope", {}), "metadata": RoutingRequest.from_query(query).metadata,
                         "label": selected["path"], "all_paths_failed": all(bool(record.get("error")) for record in choices),
                         "proxy_label": True, "comparison_config_hash": expected_config_hash,
                         "quality_proxy": "correct_abstention" if query.get('answerability') == 'unanswerable' else "answer_f1",
                         "route_scores": {record["path"]: {key: record.get(key) for key in
                                          ("answer_f1", "correct_abstention", "answerability", "evidence_coverage", "total_latency_s", "error")} for record in choices}})
    return examples


def validate_training_splits(train, dev):
    if not train or not dev or any(row["split"] != "train" for row in train) or any(row["split"] != "dev" for row in dev):
        raise ValueError("Nonempty, explicit train/dev splits are required; test is prohibited")
    if len(train) > 200 or len(dev) > 60:
        raise ValueError("Frozen router training caps are 200 train / 60 dev questions")
    if any(len({row["query_id"] for row in rows}) != len(rows) for rows in (train, dev)):
        raise ValueError("Duplicate questions in a routing training split")
    if {row["query_id"] for row in train} & {row["query_id"] for row in dev}:
        raise ValueError("Query leakage between train and dev")
    def docs(rows):
        return {doc for row in rows for doc in ([row.get("scope", {}).get("doc_id")] + row.get("scope", {}).get("doc_ids", [])) if doc}
    if docs(train) & docs(dev):
        raise ValueError("Document leakage between train and dev")
    if any(row["label"] not in ROUTES for row in train + dev):
        raise ValueError("Unknown training label")
    return {"train_n": len(train), "dev_n": len(dev),
            "train_label_counts": dict(Counter(row["label"] for row in train)),
            "dev_label_counts": dict(Counter(row["label"] for row in dev)),
            "single_class": len({row["label"] for row in train}) == 1,
            "train_hash": signature(train), "dev_hash": signature(dev)}


def validate_proxy_provenance(train, dev, data_kind):
    if data_kind not in {"real_generation_proxy", "offline_fixture"}:
        raise ValueError("Unknown routing data_kind")
    if data_kind == "real_generation_proxy":
        hashes = {row.get("comparison_config_hash") for row in train + dev}
        if len(hashes) != 1 or not next(iter(hashes)) or any(not row.get("proxy_label") for row in train + dev):
            raise ValueError("Formal routing training requires verified same-configuration generation proxy labels")
