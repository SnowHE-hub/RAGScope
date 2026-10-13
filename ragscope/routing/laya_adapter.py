"""One local audited Laya checkpoint, three choices, explicit fallback and audit."""
from __future__ import annotations

import math
import json
import os
from pathlib import Path
import sys
from time import perf_counter

from ragscope.indexing.identity import file_hash, artifact, verify_artifacts
from .core import ROUTES, RouteDecision

LAYA_MODEL_REVISION = "7b928d828b7b0e022f929d9bd2e44165aa270148"
LAYA_CODE_REVISION = "c73908d96ff73c0062c340a11bf7c54ec7d8bc4d"
LAYA_WEIGHT_SHA256 = "891102d372688fc2a094dac56a384bc537b87c63f21f9f3dac0be2b7cbc8d86c"
CRITERIA = {"bm25": "Exact terms, quoted phrases or identifiers; lexical retrieval with the lowest retrieval cost.",
            "hybrid": "Mix lexical and semantic retrieval for general evidence questions.",
            "hybrid_rerank": "Hybrid candidates followed by reranking for comparative, ambiguous or multi-part evidence questions."}
PERMUTATIONS = (ROUTES, ("hybrid", "hybrid_rerank", "bm25"), ("hybrid_rerank", "bm25", "hybrid"))


def verify_sdk_installation():
    """Validate the installed audited source without importing Laya or Torch."""
    import importlib.metadata
    receipt_path=Path(sys.prefix)/"laya_sdk_installation.json"
    receipt=json.loads(receipt_path.read_text(encoding="utf-8"))
    distribution=importlib.metadata.distribution("laya")
    installed=Path(distribution.locate_file("laya")).resolve()
    artifacts=receipt.get("installed_source_artifacts",[])
    if (receipt.get("status")!="VERIFIED" or receipt.get("revision")!=LAYA_CODE_REVISION
            or receipt.get("version")!="0.4.6" or distribution.version!="0.4.6" or not artifacts
            or any(not Path(item["path"]).resolve().is_relative_to(installed) for item in artifacts)
            or not verify_artifacts(artifacts)):
        raise ValueError("Laya installed SDK differs from audited fixed revision")
    return artifact(receipt_path)


class LayaRouter:
    def __init__(self, agent, *, min_confidence=.5, model_revision=LAYA_MODEL_REVISION,
                 code_revision=LAYA_CODE_REVISION, real_model=False):
        if not 0 <= min_confidence <= 1:
            raise ValueError("min_confidence must be in [0,1]")
        self.agent, self.min_confidence = agent, min_confidence
        self.identity = {"kind": "laya_three_choices", "model": "convaiinnovations/laya",
                         "model_revision": model_revision, "code_revision": code_revision,
                         "device": "cpu", "min_confidence": min_confidence, "real_model": real_model}

    @classmethod
    def load_local(cls, checkpoint, *, expected_sha256: dict, min_confidence=.5):
        checkpoint = Path(checkpoint).resolve()
        if not checkpoint.is_dir() or expected_sha256.get("model.safetensors", "").lower() != LAYA_WEIGHT_SHA256:
            raise ValueError("Only the audited, hash-verified local English checkpoint is supported")
        required = {"model.safetensors", "rl_agent_config.json", "tokenizer/tokenizer.json",
                    "tokenizer/tokenizer_config.json", "encoder/config.json"}
        if not required.issubset(expected_sha256) or any(not (checkpoint / name).is_file() for name in required):
            raise ValueError("Laya needs every local config/tokenizer file to prevent remote fallback")
        before = {}
        for name, digest in expected_sha256.items():
            path = (checkpoint / name).resolve()
            if not path.is_relative_to(checkpoint) or file_hash(path).lower() != digest.lower():
                raise ValueError("Laya checkpoint hash/path mismatch")
            before[name] = digest.lower()
        sdk_receipt=verify_sdk_installation()
        # A local path plus offline Hub prevents SDK fallback downloads/token use.
        flags=("HF_HUB_OFFLINE","TRANSFORMERS_OFFLINE","HF_HUB_DISABLE_IMPLICIT_TOKEN")
        prior = {key:os.environ.get(key) for key in flags}
        for key in flags:os.environ[key]="1"
        try:
            import laya
            if laya.__version__ != "0.4.6":
                raise ValueError("Laya SDK version differs from the audited 0.4.6")
            agent = laya.load(str(checkpoint), device="cpu", backend="eager", fast=False, compile=False,
                              expected_sha256=expected_sha256)
        finally:
            for key,value in prior.items():
                if value is None:os.environ.pop(key,None)
                else:os.environ[key]=value
        router = cls(agent, min_confidence=min_confidence, real_model=True)
        router.identity["sdk_installation_receipt"]=sdk_receipt
        router.identity["checkpoint_hashes_before"] = before
        router.identity["checkpoint_hashes_after"] = {name: file_hash(checkpoint / name) for name in before}
        return router

    def choose(self, request, *, order=ROUTES):
        if len(order) != 3 or set(order) != set(ROUTES):
            raise ValueError("Laya must receive exactly the three frozen choices")
        started = perf_counter()
        metadata = {"candidate_order": list(order), "identity": self.identity}
        try:
            questions = {"retrieval_path": {"type": "choice",
                "instructions": "Choose one existing evidence retrieval path for this question. Balance answer quality and retrieval cost.",
                "criteria": {route: CRITERIA[route] for route in order}}}
            raw = self.agent.predict(request.features_text(), questions, max_len=512)
            metadata["raw_decision"] = raw
            if raw.get("usage", {}).get("truncated"):
                raise ValueError("Laya truncated the routing input")
            options = raw.get("usage", {}).get("options", {}).get("retrieval_path", {})
            if options and options.get("distinct", 0) < options.get("total", 3):
                raise ValueError("Laya candidate descriptions collapsed within the head token budget")
            answer = raw["answers"]["retrieval_path"]
            probabilities = answer["probabilities"]
            if set(probabilities) != set(ROUTES) or any(not isinstance(value, (float, int)) or
                    not math.isfinite(value) or not 0 <= value <= 1 for value in probabilities.values()):
                raise ValueError("Invalid Laya probability distribution")
            total = sum(probabilities.values())
            if abs(total - 1) > .01 or total == 0:
                raise ValueError("Laya probabilities do not sum to one")
            probabilities = {route: float(probabilities[route] / total) for route in ROUTES}
            selected = answer["choice"]
            if selected not in ROUTES or probabilities[selected] < max(probabilities.values()) - 1e-4:
                raise ValueError("Invalid Laya choice")
            if max(probabilities.values()) < self.min_confidence:
                return RouteDecision("hybrid", "laya_low_confidence", probabilities, True,
                                     perf_counter() - started, metadata)
            return RouteDecision(selected, "laya_three_way", probabilities, False, perf_counter() - started, metadata)
        except Exception as exc:
            metadata["error"] = {"type": type(exc).__name__, "message": str(exc)}
            return RouteDecision("hybrid", "laya_invalid_or_unavailable", fallback=True,
                                 elapsed_s=perf_counter() - started, metadata=metadata)

    def permutation_study(self, request):
        rows = [self.choose(request, order=order).to_dict() for order in PERMUTATIONS]
        return {"decisions": rows, "choice_changed": len({row["route"] for row in rows}) > 1,
                "real_model": self.identity["real_model"]}
