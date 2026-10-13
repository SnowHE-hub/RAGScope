"""Deterministic, provenance-preserving context packing without model loading."""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, is_dataclass
from functools import lru_cache
from pathlib import Path

from ragscope.schema import ContextBundle

CONTEXT_VERSION = "canonical-packing-v2"


def as_mapping(value):
    return asdict(value) if is_dataclass(value) else dict(value)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


@lru_cache(maxsize=1)
def _reference_encoder():
    # Lazy: importing this module never downloads tokenizer data.
    runtime = os.environ.get("RAGSCOPE_HOME", "D:/RAGScope_Runtime" if os.name == "nt" else "/mnt/d/RAGScope_Runtime")
    os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(Path(runtime) / "cache/tiktoken"))
    import tiktoken
    return tiktoken.get_encoding("cl100k_base")


def reference_token_count(text):
    return len(_reference_encoder().encode(text, disallowed_special=()))


def render_piece(piece):
    title = str(piece.get("title") or piece.get("metadata", {}).get("title") or "")
    header = f'[{piece["chunk_id"]}]' + (f" {title}\n" if title else " ")
    return header + piece["text"]


def render_context(bundle):
    return "\n\n".join(render_piece(p) for p in as_mapping(bundle)["pieces"])


def _allowed(doc_id, scope):
    if scope is None or scope == "all":
        return True
    if isinstance(scope, str):
        return doc_id == scope
    if isinstance(scope, dict):
        allowed = scope.get("doc_ids", [scope["doc_id"]] if "doc_id" in scope else None)
        if allowed is None:
            raise ValueError("scope must declare doc_id or doc_ids")
        return doc_id in allowed
    return doc_id in scope


def _spans(piece):
    return [{**as_mapping(s), "doc_id": piece["doc_id"]} for s in piece.get("unit_spans", [])]


def _subtract(start, end, covered):
    ranges = [(start, end)]
    for lo, hi in sorted(covered):
        ranges = [(a, b) for x, y in ranges
                  for a, b in ((x, min(y, lo)), (max(x, hi), y)) if a < b]
    return ranges


def _uncovered_pieces(item, covered):
    """Use exact canonical slices to remove overlaps, never infer coordinates."""
    start, end = item.get("canonical_start"), item.get("canonical_end")
    if start is None or end is None or end - start != len(item["text"]):
        return [item]
    ranges = _subtract(start, end, covered.get(item["doc_id"], []))
    output = []
    for lo, hi in ranges:
        if (lo, hi) == (start, end):
            output.append(item)
            continue
        spans = []
        for span in _spans(item):
            a, b = max(lo, span["start"]), min(hi, span["end"])
            if a < b:
                spans.append({**span, "start": a, "end": b, "chunk_start": a-lo, "chunk_end": b-lo})
        output.append({**item, "chunk_id": f'{item["chunk_id"]}:{lo}-{hi}',
                       "source_chunk_id": item["chunk_id"], "text": item["text"][lo-start:hi-start],
                       "canonical_start": lo, "canonical_end": hi, "unit_spans": spans})
    return output


def pack_context(chunks, max_tokens=2048, *, token_counter=None,
                 reference_tokenizer="cl100k_base", scope=None, expansion="none",
                 chunk_lookup=None, parent_lookup=None, model_budget_checks=()):
    """Pack fragments in ranking order under shared reference/real-model budgets.

    Model checks receive rendered context; close over question and actual chat
    templates to enforce one identical bundle for every compared model.
    Budget rejection is recorded, never silently truncated.
    """
    if max_tokens < 0 or expansion not in {"none", "neighbor", "parent"}:
        raise ValueError("invalid context budget or expansion")
    count = token_counter or reference_token_count
    candidates, dedup, truncations = [], [], []
    chunk_lookup, parent_lookup = chunk_lookup or {}, parent_lookup or {}
    for raw in chunks:
        child = as_mapping(raw)
        if not _allowed(child["doc_id"], scope):
            truncations.append({"chunk_id": child["chunk_id"], "reason": "outside_scope"})
            continue
        selected = [child]
        if expansion == "parent" and child.get("parent_id"):
            parent = parent_lookup.get(child["parent_id"])
            if parent is not None:
                selected = [as_mapping(parent)]
            else:
                truncations.append({"chunk_id": child["chunk_id"], "reason": "parent_missing_used_child"})
        elif expansion == "neighbor":
            selected += [as_mapping(chunk_lookup[key]) for key in
                         (child.get("prev_chunk_id") or child.get("prev_id") or child.get("prev"),
                          child.get("next_chunk_id") or child.get("next_id") or child.get("next"))
                         if key in chunk_lookup]
        for candidate in selected:
            if candidate["doc_id"] != child["doc_id"] or not _allowed(candidate["doc_id"], scope):
                truncations.append({"chunk_id": candidate["chunk_id"], "reason": "cross_document_link"})
                continue
            candidates.append(candidate)

    pieces, seen_ids, seen_text, covered = [], set(), set(), {}
    for original in candidates:
        cid = str(original["chunk_id"])
        if not cid or "[" in cid or "]" in cid or "\n" in cid:
            raise ValueError("citation ID must be nonempty and contain no brackets/newlines")
        content_key = (original["doc_id"], original["text"])
        if cid in seen_ids or content_key in seen_text:
            dedup.append({"chunk_id": cid, "reason": "duplicate_id_or_text"})
            continue
        fragments = _uncovered_pieces(original, covered)
        if len(fragments) != 1 or fragments[0]["chunk_id"] != cid:
            dedup.append({"chunk_id": cid, "reason": "canonical_overlap", "kept_fragment_ids": [p["chunk_id"] for p in fragments]})
        for item in fragments:
            if not item["text"]:
                truncations.append({"chunk_id": cid, "reason": "empty_text"})
                continue
            rendered = "\n\n".join(render_piece(p) for p in pieces + [item])
            tokens = count(rendered)
            reason = ("reference_budget" if tokens > max_tokens else
                      "model_budget" if not all(check(rendered) for check in model_budget_checks) else None)
            if reason:
                truncations.append({"chunk_id": item["chunk_id"], "reason": reason, "candidate_reference_tokens": tokens})
                continue
            piece = json.loads(json.dumps(item, ensure_ascii=False))
            pieces.append(piece)
            if item.get("canonical_start") is not None and item.get("canonical_end") is not None:
                covered.setdefault(item["doc_id"], []).append((item["canonical_start"], item["canonical_end"]))
            seen_ids.add(cid)
            seen_text.add(content_key)
    rendered = "\n\n".join(render_piece(p) for p in pieces)
    budget = {"reference_tokenizer": reference_tokenizer, "reference_tokens": count(rendered),
              "max_reference_tokens": max_tokens, "model_budget_checks": len(model_budget_checks),
              "packing_version": CONTEXT_VERSION, "expansion": expansion}
    payload = {"pieces": pieces, "unit_spans": [s for p in pieces for s in _spans(p)],
               "deduplication": dedup, "budget": budget, "truncations": truncations}
    return ContextBundle(**payload, context_hash=fingerprint(payload))


def validate_bundle(bundle, scope=None):
    data = as_mapping(bundle)
    expected = data.pop("context_hash")
    if fingerprint(data) != expected:
        raise ValueError("ContextBundle hash mismatch")
    ids = [p["chunk_id"] for p in data["pieces"]]
    if len(ids) != len(set(ids)):
        raise ValueError("ContextBundle citation IDs are not unique")
    if any(not _allowed(piece["doc_id"], scope) for piece in data["pieces"]):
        raise ValueError("ContextBundle contains text outside the requested scope")
    return bundle
