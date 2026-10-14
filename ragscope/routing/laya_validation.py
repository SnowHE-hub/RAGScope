"""Independent, model-free acceptance of the fixed Laya probability population."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path

from filelock import FileLock

from ragscope.experiments.ledger import now, read_records
from ragscope.indexing.identity import artifact, atomic_json, file_hash, signature, verify_artifacts
from .core import ROUTES, RoutingRequest
from .laya_adapter import LAYA_CODE_REVISION, LAYA_MODEL_REVISION, LAYA_WEIGHT_SHA256, PERMUTATIONS


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_decision(decision, identity, order):
    """Return error/fallback counts; failed calls remain valid denominator members."""
    elapsed = decision.get("elapsed_s")
    require(type(elapsed) in (int, float) and math.isfinite(elapsed) and elapsed >= 0,
            "Invalid decision elapsed time")
    require(decision.get("route") in ROUTES and type(decision.get("fallback")) is bool,
            "Invalid decision route/fallback")
    metadata = decision.get("metadata", {})
    require(metadata.get("identity") == identity and metadata.get("candidate_order") == list(order),
            "Decision identity/candidate order mismatch")
    if metadata.get("error"):
        require(decision["route"] == "hybrid" and decision["fallback"] is True
                and decision.get("reason") == "laya_invalid_or_unavailable"
                and not decision.get("probabilities"), "Invalid failed-call fallback")
        require(isinstance(metadata["error"], dict) and metadata["error"].get("type")
                and metadata["error"].get("message"), "Missing retained failure details")
        return 1, 1
    raw = metadata.get("raw_decision", {})
    usage = raw.get("usage", {})
    require(not usage.get("truncated"), "Successful row has truncated input")
    options = usage.get("options", {}).get("retrieval_path", {})
    require(not options or options.get("distinct", 0) >= options.get("total", 3),
            "Successful row has collapsed choices")
    answer = raw.get("answers", {}).get("retrieval_path", {})
    probabilities = answer.get("probabilities", {})
    require(set(probabilities) == set(ROUTES) and all(type(p) in (int, float)
            and math.isfinite(p) and 0 <= p <= 1 for p in probabilities.values()),
            "Invalid raw probability distribution")
    total = sum(probabilities.values())
    require(total > 0 and abs(total - 1) <= .01, "Invalid raw probability sum")
    normalized = {route: probabilities[route] / total for route in ROUTES}
    observed = decision.get("probabilities", {})
    require(set(observed) == set(ROUTES) and all(type(observed[r]) in (int, float)
            and math.isfinite(observed[r]) and abs(observed[r] - normalized[r]) <= 1e-12
            for r in ROUTES), "Normalized probabilities differ from raw SDK output")
    selected = answer.get("choice")
    require(selected in ROUTES and normalized[selected] >= max(normalized.values()) - 1e-4,
            "Raw choice differs from probability maximum")
    low_confidence = max(normalized.values()) < .5
    require(decision["route"] == ("hybrid" if low_confidence else selected)
            and decision["fallback"] is low_confidence
            and decision.get("reason") == ("laya_low_confidence" if low_confidence else "laya_three_way"),
            "Decision differs from frozen confidence/fallback rule")
    return 0, int(low_confidence)


def validate_population(inputs, config, rows, warmups):
    """Validate all 360 queries in all three orders without reading quality labels."""
    require(inputs.get("input_hash") == signature({k: v for k, v in inputs.items() if k != "input_hash"}),
            "Frozen input hash mismatch")
    require(inputs.get("expected_queries") == 360 and inputs.get("expected_decisions") == 1080
            and inputs.get("no_quality_labels_read") is True and inputs.get("seed") == 42
            and inputs.get("min_confidence") == .5
            and inputs.get("orders") == [list(order) for order in PERMUTATIONS], "Input protocol mismatch")
    require(config.get("input_hash") == inputs["input_hash"] and config.get("expected_decisions") == 1080
            and config.get("warmup_queries") == 3, "Record configuration protocol mismatch")
    identity = config["router_identity"]
    requests = {}
    all_ids = set()
    require(set(inputs["populations"]) == {"train", "dev", "test"}, "Population splits differ")
    for split, count in (("train", 200), ("dev", 60), ("test", 100)):
        pop = inputs["populations"][split]
        ids, visible = pop["query_ids"], pop["visible_queries"]
        require(len(ids) == len(set(ids)) == len(visible) == count
                and [q["query_id"] for q in visible] == ids and not (all_ids & set(ids)),
                "Frozen population missing, duplicated, reordered or cross-split")
        all_ids.update(ids)
        for query in visible:
            require(query.get("split") == split, "Visible query split mismatch")
            require(set(query) == {"query_id", "question", "split", "scope", "metadata"},
                    "Unexpected visible input fields")
            requests[(split, query["query_id"])] = signature(asdict(RoutingRequest.from_query(query)))
    expected = {(s, q, i) for s, q in requests for i in range(3)}
    actual = set()
    errors = fallbacks = 0
    chosen = {}
    for row in rows:
        key = row.get("split"), row.get("query_id"), row.get("order_index")
        require(key in expected and key not in actual, "Unexpected or duplicated decision population member")
        actual.add(key)
        split, query_id, order_index = key
        require(type(order_index) is int, "Order index must be an integer")
        require(row.get("record_id") == f"{split}:{query_id}:{order_index}"
                and row.get("order") == list(PERMUTATIONS[order_index])
                and row.get("visible_input_hash") == requests[(split, query_id)], "Row visible input/order mismatch")
        validate_record(row, signature(config), identity)
        error, fallback = validate_decision(row["decision"], identity, PERMUTATIONS[order_index])
        errors += error
        fallbacks += fallback
        chosen.setdefault((split, query_id), set()).add(row["decision"]["route"])
    require(actual == expected and len(rows) == 1080, "Incomplete 1080-decision population")
    warm_ids = inputs["populations"]["dev"]["query_ids"][:3]
    require(len(warmups) == 3 and [r.get("record_id") for r in warmups] == warm_ids,
            "Warmup population differs from first three fixed dev queries")
    for row in warmups:
        validate_record(row, signature({**config, "purpose": "excluded_warmup"}), identity)
        require(validate_decision(row["decision"], identity, PERMUTATIONS[0])[0] == 0, "Failed real warmup")
    return {"n_decisions": len(rows), "n_queries": len(all_ids), "warmups_excluded": len(warmups),
            "errors": errors, "fallbacks": fallbacks,
            "queries_with_order_dependent_selected_route": sum(len(value) > 1 for value in chosen.values())}


def validate_record(row, config_hash, identity):
    require(row.get("record_hash") == signature({k: v for k, v in row.items() if k != "record_hash"}),
            "Missing or changed record hash")
    require(row.get("config_hash") == config_hash and row.get("real_model") is True
            and row.get("model_identity") == identity, "Record model/config identity mismatch")


def validate_run(home, campaign):
    """Only accept a fully exited worker and available writer lock; import no model SDK."""
    import psutil
    home = Path(home).resolve()
    directory = home / "runs" / (campaign + "_laya")
    worker = read(directory / "worker.json")
    try:
        live = psutil.Process(worker["pid"])
        require(abs(live.create_time() - worker["creation_time"]) > .001, "Laya worker still running")
    except psutil.NoSuchProcess:
        pass
    with FileLock(str(directory / "writer.lock"), timeout=0):
        inputs, config, summary = [read(directory / name) for name in ("INPUTS.json", "records.config.json", "SUMMARY.json")]
        require(verify_artifacts(inputs["sources"]), "Frozen source artifact changed")
        route = read(home / "runs" / campaign / "ROUTING_QUERY_MANIFEST.json")
        require(route.get("manifest_hash") == signature({k: v for k, v in route.items() if k != "manifest_hash"})
                and all(inputs["populations"][s]["query_ids"] == route["query_ids"][s]
                        for s in ("train", "dev", "test")), "Parent routing population changed")
        model = home / "models" / "laya" / LAYA_MODEL_REVISION
        manifest_path = model / "model_manifest.json"
        manifest = read(manifest_path)
        require(manifest.get("status") == "VERIFIED" and manifest.get("revision") == LAYA_MODEL_REVISION
                and config["model_manifest"] == artifact(manifest_path) and verify_artifacts(manifest["artifacts"]),
                "Laya checkpoint manifest changed")
        hashes = {a["relative_path"]: a["sha256"] for a in manifest["artifacts"]}
        require(hashes.get("model.safetensors") == LAYA_WEIGHT_SHA256, "Unaudited Laya weights")
        identity = config["router_identity"]
        required = {"kind": "laya_three_choices", "model": "convaiinnovations/laya",
                    "model_revision": LAYA_MODEL_REVISION, "code_revision": LAYA_CODE_REVISION,
                    "device": "cpu", "min_confidence": .5, "real_model": True,
                    "torch_version": "2.9.1+cpu", "cpu_threads": 4, "dtype": "torch.float32",
                    "compile": False, "backend": "eager"}
        require(all(identity.get(k) == v for k, v in required.items())
                and identity.get("checkpoint_hashes_before") == hashes
                and identity.get("checkpoint_hashes_after") == hashes, "Model/runtime identity changed")
        sdk_receipt_path = home / "envs" / "models" / "laya_sdk_installation.json"
        sdk = read(sdk_receipt_path)
        require(identity.get("sdk_installation_receipt") == artifact(sdk_receipt_path)
                and sdk.get("status") == "VERIFIED" and sdk.get("revision") == LAYA_CODE_REVISION
                and sdk.get("version") == "0.4.6" and sdk.get("installed_source_artifacts")
                and verify_artifacts(sdk["installed_source_artifacts"]), "Audited SDK installation changed")
        source = Path(__file__).with_name("laya_campaign.py")
        require(config.get("source_sha256") == file_hash(source), "Executing campaign source changed")
        require(read(directory / "warmups.config.json") == {**config, "purpose": "excluded_warmup"},
                "Warmup configuration changed")
        rows, warmups = [read_records(directory / name) for name in ("records.jsonl", "warmups.jsonl")]
        result = validate_population(inputs, config, rows, warmups)
        require(summary.get("status") == "COMPLETE" and summary.get("input_hash") == inputs["input_hash"]
                and summary.get("router_identity") == identity and summary.get("quality_labels_read") is False
                and summary.get("quality_scoring") == "PENDING_REAL_ROUTE_LABELS"
                and all(summary.get(key) == result[key] for key in ("n_decisions", "n_queries", "errors", "fallbacks"))
                and verify_artifacts(summary["artifacts"]), "Completion summary inconsistent")
        result.update(status="PASS", checked_at=now(), input_hash=inputs["input_hash"], worker=worker,
                      worker_exited=True, quality_labels_read=False, quality_scoring="PENDING_REAL_ROUTE_LABELS",
                      note="Execution/provenance acceptance only; errors and confidence fallbacks remain in all denominators.",
                      artifacts=[artifact(directory / name) for name in ("INPUTS.json", "records.jsonl", "records.config.json",
                                 "warmups.jsonl", "warmups.config.json", "SUMMARY.json", "LOAD_STATE.json")]
                                 + [artifact(source), artifact(Path(__file__))])
        atomic_json(directory / "RAW_ACCEPTANCE.json", result)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--home", default="D:/RAGScope_Runtime")
    parser.add_argument("--campaign", default="20261014_v2")
    args = parser.parse_args()
    result = validate_run(args.home, args.campaign)
    print(json.dumps({k: v for k, v in result.items() if k not in {"artifacts", "worker"}}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
