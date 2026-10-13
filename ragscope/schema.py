"""JSON-serializable v2 records. All source offsets are canonical, half-open.

Model objects do not load tokenizers, models or optional packages. Unknown input
fields are preserved in metadata when possible so raw dataset provenance survives.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
import hashlib
import json
from typing import Any


def signature(value: Any) -> str:
    """SHA-256 over stable JSON; rejects non-finite values and implicit reprs."""
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class JsonRecord:
    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict):
        names = {item.name for item in fields(cls)}
        selected = {key: val for key, val in value.items() if key in names}
        extras = {key: val for key, val in value.items() if key not in names}
        if extras and "metadata" in names:
            selected["metadata"] = {**selected.get("metadata", {}), **extras}
        elif extras:
            raise ValueError(f"Unexpected {cls.__name__} fields: {sorted(extras)}")
        return cls(**selected)


@dataclass
class Document(JsonRecord):
    doc_id: str
    text: str = ""
    source: str = ""
    source_revision: str = ""
    language: str = "en"
    title: str = ""
    sections: list[dict] = field(default_factory=list)
    raw_checksum: str = ""
    canonical_checksum: str = ""
    split: str = ""
    metadata: dict = field(default_factory=dict)
    evidence_units: list[dict] = field(default_factory=list)


@dataclass
class EvidenceUnit(JsonRecord):
    unit_id: str
    doc_id: str
    section_id: str
    paragraph_index: int
    text: str
    canonical_start: int
    canonical_end: int
    metadata: dict = field(default_factory=dict)


@dataclass
class Chunk(JsonRecord):
    # The first eight fields retain the existing positional construction API.
    chunk_id: str
    doc_id: str
    text: str
    n_tokens: int
    position_in_doc: int
    prev_chunk_id: str | None = None
    next_chunk_id: str | None = None
    metadata: dict = field(default_factory=dict)
    chunking_signature: str = ""
    unit_spans: list[dict] = field(default_factory=list)
    tokenizer_id: str = ""
    tokenizer_revision: str = ""
    parent_id: str | None = None
    canonical_start: int = 0
    canonical_end: int = 0


@dataclass
class Query(JsonRecord):
    query_id: str
    question: str
    split: str = ""
    scope: dict = field(default_factory=dict)
    answerability: str = "answerable"
    answer_refs: list = field(default_factory=list)
    evidence_reference_sets: list[list[str]] = field(default_factory=list)
    annotation_disagreements: list = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


@dataclass
class ContextBundle(JsonRecord):
    pieces: list[dict] = field(default_factory=list)
    unit_spans: list[dict] = field(default_factory=list)
    deduplication: list[dict] = field(default_factory=list)
    budget: dict = field(default_factory=dict)
    truncations: list[dict] = field(default_factory=list)
    context_hash: str = ""


@dataclass
class GenerationRecord(JsonRecord):
    query_id: str
    config: dict = field(default_factory=dict)
    prompt: str = ""
    raw_output: str = ""
    parsed_output: dict = field(default_factory=dict)
    error: dict | None = None
    timings: dict = field(default_factory=dict)
    resources: dict = field(default_factory=dict)
    campaign_id: str = ""
    run_id: str = ""
    model_id: str = ""
    real_model: bool = False
    backend_identity: dict = field(default_factory=dict)
    prompt_hash: str = ""
