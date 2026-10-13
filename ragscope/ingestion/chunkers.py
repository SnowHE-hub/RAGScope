"""Offline, source-coordinate-preserving chunkers with strict token budgets.

The default unicode_codepoint tokenizer is only an explicit offline reference.
Real embedding experiments must inject that embedding model's pinned tokenizer.
SemanticChunker remains a compatibility alias for StructureRecursiveChunker.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
import re
from typing import Dict, List

from ragscope.ingestion.cleaner import evidence_units
from ragscope.schema import Chunk, Document, signature


class TokenizerAdapter:
    """Wrap an already loaded tokenizer; construction/import never downloads.

    HF tokenizers are identified by an explicit revision and use no special tokens
    for chunk bodies. Complete embedding inputs must separately count their wrapper.
    """
    def __init__(self, backend=None, tokenizer_id: str | None = None, revision: str = "v1"):
        self.backend = backend
        self.tokenizer_id = tokenizer_id or ("unicode_codepoint" if backend is None else
                                             getattr(backend, "name_or_path", type(backend).__name__))
        self.revision = revision

    def encode(self, text: str) -> list[int]:
        if self.backend is None:
            return [ord(character) for character in text]
        try:
            return list(self.backend.encode(text, add_special_tokens=False))
        except TypeError:
            return list(self.backend.encode(text))

    def decode(self, ids: list[int]) -> str:
        if self.backend is None:
            return "".join(chr(token) for token in ids)
        try:
            return self.backend.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        except TypeError:
            return self.backend.decode(ids)

    def count(self, text: str) -> int:
        return len(text) if self.backend is None else len(self.encode(text))


class BaseChunker(ABC):
    method = "base"

    def __init__(self, tokenizer=None) -> None:
        self.tokenizer = tokenizer if isinstance(tokenizer, TokenizerAdapter) else TokenizerAdapter(tokenizer)
        self.enc = self.tokenizer  # Legacy callers use enc.encode.
        self.last_events: list[dict] = []

    def count_tokens(self, text: str) -> int:
        return self.tokenizer.count(text)

    @property
    def chunking_signature(self) -> str:
        return signature({"version": "canonical_chunking_v2", "method": self.method,
                          "parameters": self.parameters, "tokenizer_id": self.tokenizer.tokenizer_id,
                          "tokenizer_revision": self.tokenizer.revision})

    def _document(self, doc) -> dict:
        return doc.to_dict() if isinstance(doc, Document) else doc

    def _fit_end(self, text: str, start: int, stop: int, budget: int) -> int:
        """Find a valid Unicode boundary. Every returned candidate is re-counted.

        Token counts need not be exactly additive or monotonic (BPE merges). Binary
        search is used for efficiency, not to claim a globally maximal token fill.
        """
        if self.tokenizer.backend is None:
            return min(stop, start + budget)
        low, high = start, min(stop, start + max(8, budget * 4))
        if self.count_tokens(text[start:high]) <= budget:
            low = high
            while high < stop:
                high = min(stop, start + (high - start) * 2)
                if self.count_tokens(text[start:high]) > budget:
                    break
                low = high
            if low == stop:
                return stop
        while low + 1 < high:
            mid = (low + high) // 2
            if self.count_tokens(text[start:mid]) <= budget:
                low = mid
            else:
                high = mid
        if low == start:
            if self.count_tokens(text[start:start + 1]) <= budget:
                return start + 1
            raise ValueError("Token budget cannot fit one Unicode code point with this tokenizer")
        return low

    def _overlap_start(self, text: str, start: int, end: int, budget: int) -> int:
        if not budget:
            return end
        if self.tokenizer.backend is None:
            return max(start + 1, end - budget)
        low, high = start + 1, end
        while low < high:
            mid = (low + high) // 2
            if self.count_tokens(text[mid:end]) <= budget:
                high = mid
            else:
                low = mid + 1
        return low

    def _fixed_ranges(self, text: str, start: int, stop: int, budget: int,
                      overlap: int = 0) -> list[tuple[int, int]]:
        ranges = []
        while start < stop:
            end = self._fit_end(text, start, stop, budget)
            if ranges and end <= ranges[-1][1]:
                start = ranges[-1][1]
                end = self._fit_end(text, start, stop, budget)
                self.last_events.append({"reason": "overlap_reduced_for_progress", "start": start})
            # Prefer whitespace boundaries when one exists, preserving exact bytes.
            if end < stop:
                words = list(re.finditer(r"\s+", text[start:end]))
                if words and start + words[-1].end() > start:
                    candidate = start + words[-1].end()
                    if (not ranges or candidate > ranges[-1][1]) and self.count_tokens(text[start:candidate]) <= budget:
                        end = candidate
                else:
                    self.last_events.append({"reason": "unicode_boundary_fallback", "start": start, "end": end})
            ranges.append((start, end))
            if end == stop:
                break
            start = self._overlap_start(text, start, end, overlap)
        return ranges

    def _assemble(self, doc: dict, spans: list[tuple[int, int]], *, signature_override=None) -> list[Chunk]:
        text, doc_id = doc.get("text", ""), doc["doc_id"]
        units = evidence_units(doc)
        config = signature_override or self.chunking_signature
        document_hash = signature({"text": text, "source_revision": doc.get("source_revision", "")})
        ids = [f"{doc_id}:c:{signature([document_hash, config, start, end])[:24]}" for start, end in spans]
        result = []
        for i, (start, end) in enumerate(spans):
            projected = []
            for unit in units:
                left, right = max(start, unit["canonical_start"]), min(end, unit["canonical_end"])
                if left < right:
                    projected.append({"unit_id": unit["unit_id"], "doc_id": doc_id, "start": left,
                                      "end": right, "chunk_start": left - start, "chunk_end": right - start})
            previous_end = spans[i - 1][1] if i else start
            metadata = {**doc.get("metadata", {}), "coordinate_system": "canonical_unicode_codepoint",
                        "overlap_previous_chars": max(0, previous_end - start),
                        "overlap_previous_tokens": self.count_tokens(text[start:max(start, previous_end)]),
                        "title": doc.get("title", ""), "split": doc.get("split", ""),
                        "source_revision": doc.get("source_revision", ""),
                        "boundary_events": [dict(event) for event in self.last_events
                                            if start <= event["start"] < end]}
            result.append(Chunk(ids[i], doc_id, text[start:end], self.count_tokens(text[start:end]), i,
                                ids[i - 1] if i else None, ids[i + 1] if i + 1 < len(ids) else None,
                                metadata, config, projected, self.tokenizer.tokenizer_id,
                                self.tokenizer.revision, canonical_start=start, canonical_end=end))
        return result

    @abstractmethod
    def chunk(self, doc: Dict) -> List[Chunk]:
        ...


def _positive(value: int, name: str):
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


class FixedSizeChunker(BaseChunker):
    method = "fixed"

    def __init__(self, chunk_size: int = 512, overlap: int = 64, *, tokenizer=None) -> None:
        super().__init__(tokenizer)
        _positive(chunk_size, "chunk_size")
        if not isinstance(overlap, int) or isinstance(overlap, bool) or not 0 <= overlap < chunk_size:
            raise ValueError("overlap must be an integer in [0, chunk_size)")
        self.chunk_size, self.overlap = chunk_size, overlap
        self.parameters = {"chunk_size": chunk_size, "overlap": overlap}

    def chunk(self, doc: Dict) -> List[Chunk]:
        doc = self._document(doc)
        self.last_events = []
        text = doc.get("text", "")
        spans = self._fixed_ranges(text, 0, len(text), self.chunk_size, self.overlap) if text.strip() else []
        return self._assemble(doc, spans)


def _sentence_spans(text: str, start: int = 0, end: int | None = None) -> list[tuple[int, int]]:
    end = len(text) if end is None else end
    spans, cursor = [], start
    # Deterministic offline segmentation; no NLTK import/download side effect.
    for match in re.finditer(r"[.!?。！？]+[\"'”’）)]*(?:\s+|(?=[\u4e00-\u9fff])|$)", text[start:end]):
        stop = start + match.end()
        if stop > cursor:
            spans.append((cursor, stop))
            cursor = stop
    if cursor < end:
        spans.append((cursor, end))
    return spans


class SentenceAwareChunker(BaseChunker):
    method = "sentence"

    def __init__(self, max_tokens: int = 400, *, overlap_sentences: int = 1, tokenizer=None) -> None:
        super().__init__(tokenizer)
        _positive(max_tokens, "max_tokens")
        if overlap_sentences not in (0, 1) or isinstance(overlap_sentences, bool):
            raise ValueError("Only 0 or 1 overlapping sentence is supported")
        self.max_tokens, self.overlap_sentences = max_tokens, overlap_sentences
        self.parameters = {"max_tokens": max_tokens, "overlap_sentences": overlap_sentences}

    def _pack(self, text, atoms):
        output, current = [], []
        for atom in atoms:
            if current and self.count_tokens(text[current[0][0]:atom[1]]) > self.max_tokens:
                output.append((current[0][0], current[-1][1]))
                overlap = current[-1:] if self.overlap_sentences and current[-1][2] else []
                if overlap and self.count_tokens(text[overlap[0][0]:atom[1]]) > self.max_tokens:
                    self.last_events.append({"reason": "sentence_overlap_reduced", "start": atom[0]})
                    overlap = []
                current = overlap
            current.append(atom)
        if current:
            output.append((current[0][0], current[-1][1]))
        return output

    def chunk(self, doc: Dict) -> List[Chunk]:
        doc = self._document(doc)
        self.last_events = []
        text = doc.get("text", "")
        atoms = []
        for start, end in _sentence_spans(text) if text.strip() else []:
            if self.count_tokens(text[start:end]) <= self.max_tokens:
                atoms.append((start, end, True))
            else:
                self.last_events.append({"reason": "oversized_sentence_fallback", "start": start, "end": end})
                atoms.extend((left, right, False) for left, right in self._fixed_ranges(text, start, end, self.max_tokens))
        return self._assemble(doc, self._pack(text, atoms))


def _structure_spans(text: str) -> list[tuple[int, int]]:
    spans, cursor, offset, fence = [], 0, 0, None
    for line in text.splitlines(keepends=True):
        stripped = line.lstrip()
        marker = stripped[:3] if stripped.startswith(("```", "~~~")) else None
        starts_block = not fence and (marker or re.match(r"(?:#{1,6}\s|[-*+]\s|\d+[.)]\s)", stripped))
        if starts_block and offset > cursor:
            spans.append((cursor, offset))
            cursor = offset
        if marker:
            if fence == marker:
                fence = None
                spans.append((cursor, offset + len(line)))
                cursor = offset + len(line)
            elif fence is None:
                fence = marker
        if not fence and not line.strip() and offset + len(line) > cursor:
            spans.append((cursor, offset + len(line)))
            cursor = offset + len(line)
        offset += len(line)
    if cursor < len(text):
        spans.append((cursor, len(text)))
    return spans


class StructureRecursiveChunker(SentenceAwareChunker):
    method = "structure_recursive"

    def __init__(self, max_tokens: int = 450, *, tokenizer=None) -> None:
        super().__init__(max_tokens, overlap_sentences=0, tokenizer=tokenizer)
        self.parameters = {"max_tokens": max_tokens, "overlap": 0}

    def chunk(self, doc: Dict) -> List[Chunk]:
        doc = self._document(doc)
        self.last_events = []
        text, atoms = doc.get("text", ""), []
        for start, end in _structure_spans(text) if text.strip() else []:
            if self.count_tokens(text[start:end]) <= self.max_tokens:
                atoms.append((start, end, False))
            else:
                for left, right in _sentence_spans(text, start, end):
                    atoms.extend((a, b, False) for a, b in self._fixed_ranges(text, left, right, self.max_tokens))
                self.last_events.append({"reason": "oversized_structure_fallback", "start": start, "end": end})
        return self._assemble(doc, self._pack(text, atoms))


# Historical name did not use embedding semantic similarity; preserve import API.
SemanticChunker = StructureRecursiveChunker


class ParentChildChunker(BaseChunker):
    method = "parent_child"

    def __init__(self, child_tokens: int = 256, child_overlap: int = 32,
                 parent_tokens: int = 1024, *, tokenizer=None) -> None:
        super().__init__(tokenizer)
        _positive(parent_tokens, "parent_tokens")
        self.child = FixedSizeChunker(child_tokens, child_overlap, tokenizer=self.tokenizer)
        if parent_tokens < child_tokens:
            raise ValueError("parent_tokens must be >= child_tokens")
        self.parent = StructureRecursiveChunker(parent_tokens, tokenizer=self.tokenizer)
        self.parameters = {"child_tokens": child_tokens, "child_overlap": child_overlap,
                           "parent_tokens": parent_tokens}
        self.parents: dict[str, Chunk] = {}

    def chunk_with_parents(self, doc: Dict) -> tuple[list[Chunk], list[Chunk]]:
        doc = self._document(doc)
        self.last_events = []
        self.child.last_events = []
        parents = self.parent.chunk(doc)
        parent_spans = [(parent.canonical_start, parent.canonical_end) for parent in parents]
        parent_signature = signature({"parent_of": self.chunking_signature})
        parents = self._assemble(doc, parent_spans, signature_override=parent_signature)
        spans, parent_ids = [], []
        for parent in parents:
            local = self.child._fixed_ranges(doc["text"], parent.canonical_start, parent.canonical_end,
                                             self.child.chunk_size, self.child.overlap)
            spans.extend(local)
            parent_ids.extend([parent.chunk_id] * len(local))
        self.last_events = self.parent.last_events + self.child.last_events
        children = self._assemble(doc, spans)
        for child, parent_id in zip(children, parent_ids):
            child.parent_id = parent_id
        self.parents = {parent.chunk_id: parent for parent in parents}
        return children, parents

    def chunk(self, doc: Dict) -> List[Chunk]:
        return self.chunk_with_parents(doc)[0]
