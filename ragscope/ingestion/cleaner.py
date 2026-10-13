"""Structure-preserving cleaning with explicitly selectable legacy behavior."""
from __future__ import annotations

import hashlib
from copy import deepcopy
from html.parser import HTMLParser
import logging
import re
import unicodedata
from typing import Dict, List

from ragscope.schema import EvidenceUnit, signature

logger = logging.getLogger(__name__)
MIN_TEXT_LEN = 50
MAX_TEXT_LEN = 5000
_HTML_TAG = re.compile(r"<[^>]+>")
_WHITESPACE = re.compile(r"\s+")


def _clean_text(text: str) -> str:
    """Original v1 normalizer, retained only for explicit legacy reproduction."""
    return _WHITESPACE.sub(" ", _HTML_TAG.sub(" ", unicodedata.normalize("NFKC", text))).strip()


class _HTMLText(HTMLParser):
    BLOCKS = {"p", "div", "section", "article", "h1", "h2", "h3", "h4",
              "h5", "h6", "ul", "ol", "li", "pre", "blockquote", "table", "tr"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.hidden = 0

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style"}:
            self.hidden += 1
        elif not self.hidden and (tag in self.BLOCKS or tag == "br"):
            self.parts.append("\n" if tag in {"li", "br"} else "\n\n")

    def handle_endtag(self, tag):
        if tag in {"script", "style"}:
            self.hidden = max(0, self.hidden - 1)
        elif not self.hidden and tag in self.BLOCKS:
            self.parts.append("\n\n")

    def handle_data(self, data):
        if not self.hidden:
            self.parts.append(data)


def clean_text(text: str, content_type: str = "text") -> str:
    """Normalize line endings without flattening prose or altering code tokens.

    HTML is parsed only when the caller explicitly identifies it as HTML.
    Whitespace and Unicode spelling in code/plaintext are intentionally retained.
    All downstream positions are into this returned canonical string, not raw HTML.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if content_type not in {"text", "plain", "markdown", "code", "html", "text/html"}:
        raise ValueError(f"Unsupported content_type: {content_type}")
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    if content_type in {"html", "text/html"}:
        parser = _HTMLText()
        parser.feed(text)
        parser.close()
        text = "".join(parser.parts)
    return text.strip("\n")


def evidence_units(doc: Dict) -> list[dict]:
    """Validate provided units or derive stable paragraph units before chunking."""
    text = doc.get("text", "")
    supplied = doc.get("evidence_units", [])
    if supplied:
        result = [unit.to_dict() if isinstance(unit, EvidenceUnit) else dict(unit) for unit in supplied]
        seen = set()
        for unit in result:
            start, end = unit["canonical_start"], unit["canonical_end"]
            if not (0 <= start < end <= len(text)) or text[start:end] != unit["text"]:
                raise ValueError(f"Invalid canonical evidence span: {unit['unit_id']}")
            if unit["unit_id"] in seen or unit["doc_id"] != doc["doc_id"]:
                raise ValueError("Duplicate evidence ID or mismatched document")
            seen.add(unit["unit_id"])
        return result
    result = []
    for i, match in enumerate(re.finditer(r"\S[^\n]*(?:\n(?!\s*\n)[^\n]*)*", text)):
        start, end = match.span()
        identity = {"doc_id": doc["doc_id"], "source_revision": doc.get("source_revision", ""),
                    "section_id": "body", "paragraph_index": i, "text": match.group()}
        result.append(EvidenceUnit(
            unit_id=f"{doc['doc_id']}:u:{signature(identity)[:20]}", doc_id=doc["doc_id"],
            section_id="body", paragraph_index=i, text=match.group(),
            canonical_start=start, canonical_end=end).to_dict())
    return result


def clean_documents(docs: List[Dict], *, legacy: bool = False, min_chars: int = 1,
                    max_chars: int | None = None, rejection_log: list | None = None) -> List[Dict]:
    if min_chars < 0 or (max_chars is not None and max_chars < min_chars):
        raise ValueError("Invalid cleaning length bounds")
    cleaned = []
    for doc in docs:
        raw = doc.get("text", "")
        text = _clean_text(raw) if legacy else clean_text(raw, doc.get("content_type", "text"))
        minimum, maximum = (MIN_TEXT_LEN, MAX_TEXT_LEN) if legacy else (min_chars, max_chars)
        reason = "empty" if not text.strip() else "below_min_chars" if len(text) < minimum else (
            "above_max_chars" if maximum is not None and len(text) > maximum else None)
        if reason:
            record = {"doc_id": doc.get("doc_id"), "reason": reason, "canonical_chars": len(text)}
            if rejection_log is not None:
                rejection_log.append(record)
            logger.info("clean_documents rejected %s", record)
            continue
        if legacy:
            cleaned.append({**doc, "text": text})
            continue
        item = {**doc, "text": text, "raw_checksum": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
                "canonical_checksum": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "metadata": {**doc.get("metadata", {}), "normalization_version": "structure_v2",
                             "coordinate_system": "canonical_unicode_codepoint"}}
        if text != raw and doc.get("evidence_units"):
            raise ValueError("Clean before assigning evidence units; supplied offsets would change")
        item["evidence_units"] = evidence_units(item)
        cleaned.append(item)
    return cleaned


def clean_documents_legacy(docs: List[Dict]) -> List[Dict]:
    return clean_documents(docs, legacy=True)


def dedup_documents(docs: List[Dict], *, alias_map: dict | None = None) -> List[Dict]:
    """Exact dedup with ID aliases. Cross-split duplicates are retained and flagged.

    Callers must use aliases to rewrite scope/gold document IDs. Preserving the
    cross-split group avoids quietly moving held-out labels into a training split.
    """
    seen: dict[str, dict] = {}
    result = []
    aliases = alias_map if alias_map is not None else {}
    for doc in docs:
        digest = hashlib.sha256(doc.get("text", "").encode("utf-8")).hexdigest()
        current = {**doc, "metadata": deepcopy(doc.get("metadata", {}))}
        existing = seen.get(digest)
        if existing is None:
            seen[digest] = current
            result.append(current)
            aliases[doc["doc_id"]] = doc["doc_id"]
        elif existing.get("split", "") != doc.get("split", ""):
            for member in (existing, current):
                member["metadata"]["duplicate_group"] = digest
                member["metadata"]["cross_split_duplicate"] = True
            result.append(current)
            aliases[doc["doc_id"]] = doc["doc_id"]
        else:
            aliases[doc["doc_id"]] = existing["doc_id"]
            existing["metadata"].setdefault("aliases", []).append(doc["doc_id"])
            existing["metadata"].setdefault("alias_evidence_units", {})[doc["doc_id"]] = evidence_units(doc)
    return result


def remap_query_aliases(queries: List[Dict], documents: List[Dict], alias_map: dict) -> List[Dict]:
    """Rebind exact duplicate documents and stable evidence IDs without label loss.

    Cross-split flags must still be reconciled by the dataset split builder.
    Missing evidence mappings raise instead of silently producing invalid gold IDs.
    """
    unit_aliases = {}
    for doc in documents:
        target_units = evidence_units(doc)
        targets = {(unit["canonical_start"], unit["canonical_end"], unit["text"]): unit["unit_id"]
                   for unit in target_units}
        for alias_units in doc.get("metadata", {}).get("alias_evidence_units", {}).values():
            for unit in alias_units:
                key = (unit["canonical_start"], unit["canonical_end"], unit["text"])
                if key not in targets:
                    raise ValueError("Duplicate text has incompatible evidence boundaries; retain both documents")
                unit_aliases[unit["unit_id"]] = targets[key]
    result = []
    for query in queries:
        item = deepcopy(query)
        for key in ("doc_id", "gold_doc_id"):
            if key in item:
                item[key] = alias_map.get(item[key], item[key])
        scope = item.get("scope", {})
        if "doc_id" in scope:
            scope["doc_id"] = alias_map.get(scope["doc_id"], scope["doc_id"])
        if "doc_ids" in scope:
            scope["doc_ids"] = list(dict.fromkeys(alias_map.get(value, value) for value in scope["doc_ids"]))
        item["evidence_reference_sets"] = [[unit_aliases.get(unit_id, unit_id) for unit_id in ref]
                                           for ref in item.get("evidence_reference_sets", [])]
        result.append(item)
    return result
