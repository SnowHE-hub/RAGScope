"""
Three chunking strategies for RAGScope.

FixedSizeChunker   — token-budget sliding window, word-boundary safe
SentenceAwareChunker — NLTK sentence tokeniser with one-sentence overlap
SemanticChunker    — three-level recursive split (paragraph → sentence → token)
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import nltk
import tiktoken
from loguru import logger

# ---------------------------------------------------------------------------
# NLTK bootstrap (handles both old punkt and new punkt_tab)
# ---------------------------------------------------------------------------

def _ensure_nltk() -> None:
    for resource in ("punkt", "punkt_tab"):
        try:
            nltk.data.find(f"tokenizers/{resource}")
            return
        except LookupError:
            pass
    for resource in ("punkt_tab", "punkt"):
        try:
            nltk.download(resource, quiet=True)
            return
        except Exception:
            pass


_ensure_nltk()

# ---------------------------------------------------------------------------
# Chunk dataclass
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    n_tokens: int
    position_in_doc: int
    prev_chunk_id: Optional[str]
    next_chunk_id: Optional[str]
    metadata: Dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# BaseChunker
# ---------------------------------------------------------------------------

class BaseChunker(ABC):
    def __init__(self) -> None:
        self.enc = tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, text: str) -> int:
        return len(self.enc.encode(text))

    def _build_chunks(self, doc: Dict, texts: List[str]) -> List[Chunk]:
        """Assemble Chunk objects from a list of text pieces, wiring the linked list."""
        doc_id = doc["doc_id"]
        meta = doc.get("metadata", {})
        n = len(texts)
        chunks: List[Chunk] = []

        for i, text in enumerate(texts):
            chunk_id = f"{doc_id}_chunk_{i}"
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    text=text,
                    n_tokens=self.count_tokens(text),
                    position_in_doc=i,
                    prev_chunk_id=f"{doc_id}_chunk_{i - 1}" if i > 0 else None,
                    next_chunk_id=f"{doc_id}_chunk_{i + 1}" if i < n - 1 else None,
                    metadata=meta,
                )
            )

        return chunks

    @abstractmethod
    def chunk(self, doc: Dict) -> List[Chunk]:
        ...


# ---------------------------------------------------------------------------
# FixedSizeChunker
# ---------------------------------------------------------------------------

class FixedSizeChunker(BaseChunker):
    """
    Sliding window over word tokens.

    Uses prefix-sum of per-word token counts for O(n) window computation.
    Word boundaries (spaces) are always respected — never cuts mid-word.
    """

    def __init__(self, chunk_size: int = 512, overlap: int = 64) -> None:
        super().__init__()
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, doc: Dict) -> List[Chunk]:
        text = doc.get("text", "").strip()
        if not text:
            return []

        words = text.split()
        if not words:
            return []

        # Token counts per word (first word has no leading space)
        word_tokens: List[int] = [len(self.enc.encode(words[0]))]
        for w in words[1:]:
            word_tokens.append(len(self.enc.encode(" " + w)))

        # Prefix sum over token counts
        prefix = [0] * (len(words) + 1)
        for i, t in enumerate(word_tokens):
            prefix[i + 1] = prefix[i] + t

        chunk_texts: List[str] = []
        start = 0

        while start < len(words):
            # Extend window right as long as token budget allows
            end = start
            while end < len(words) and prefix[end + 1] - prefix[start] <= self.chunk_size:
                end += 1

            # end is now exclusive upper bound; handle single over-budget word
            if end == start:
                end = start + 1

            chunk_texts.append(" ".join(words[start:end]))

            # All words consumed — no more chunks needed
            if end >= len(words):
                break

            # Compute overlap: walk backwards from `end` collecting ≤ overlap tokens
            overlap_start = end
            overlap_accum = 0
            for i in range(end - 1, start - 1, -1):
                wt = prefix[i + 1] - prefix[i]
                if overlap_accum + wt > self.overlap:
                    break
                overlap_accum += wt
                overlap_start = i

            # Always advance at least one word to prevent infinite loop
            next_start = overlap_start if overlap_start > start else start + 1
            start = next_start

        return self._build_chunks(doc, chunk_texts)


# ---------------------------------------------------------------------------
# SentenceAwareChunker
# ---------------------------------------------------------------------------

class SentenceAwareChunker(BaseChunker):
    """
    Accumulates NLTK sentences up to max_tokens per chunk.

    - Does NOT split individual sentences (they get their own chunk if oversized).
    - Adjacent chunks share one sentence of overlap.
    """

    def __init__(self, max_tokens: int = 400) -> None:
        super().__init__()
        self.max_tokens = max_tokens

    def chunk(self, doc: Dict) -> List[Chunk]:
        text = doc.get("text", "").strip()
        if not text:
            return []

        sentences = nltk.sent_tokenize(text)
        if not sentences:
            return []

        chunk_texts: List[str] = []
        current_sents: List[str] = []
        current_tokens = 0

        for sent in sentences:
            sent_tokens = self.count_tokens(sent)

            # Oversized sentence: flush current, emit alone, no overlap
            if sent_tokens >= self.max_tokens:
                if current_sents:
                    chunk_texts.append(" ".join(current_sents))
                    current_sents = []
                    current_tokens = 0
                chunk_texts.append(sent)
                continue

            # Would overflow: flush and carry last sentence as overlap
            if current_sents and current_tokens + sent_tokens > self.max_tokens:
                chunk_texts.append(" ".join(current_sents))
                current_sents = current_sents[-1:]
                current_tokens = self.count_tokens(current_sents[0])

            current_sents.append(sent)
            current_tokens += sent_tokens

        if current_sents:
            chunk_texts.append(" ".join(current_sents))

        return self._build_chunks(doc, chunk_texts)


# ---------------------------------------------------------------------------
# SemanticChunker
# ---------------------------------------------------------------------------

class SemanticChunker(BaseChunker):
    """
    Three-level recursive split:
      Level 1 — paragraph boundaries (\\n\\n)
      Level 2 — sentence boundaries (NLTK)
      Level 3 — fixed token window (word-safe)

    Atomic units (paragraphs or sentences that fit) are then greedily packed
    into chunks of at most max_tokens.
    """

    def __init__(self, max_tokens: int = 450) -> None:
        super().__init__()
        self.max_tokens = max_tokens

    # --- public ---

    def chunk(self, doc: Dict) -> List[Chunk]:
        text = doc.get("text", "").strip()
        if not text:
            return []
        units = self._split_into_units(text)
        chunk_texts = self._group_units(units)
        return self._build_chunks(doc, chunk_texts)

    # --- private helpers ---

    def _split_into_units(self, text: str) -> List[str]:
        """Recursively decompose text into atomic units ≤ max_tokens."""
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        units: List[str] = []

        for para in paragraphs:
            if self.count_tokens(para) <= self.max_tokens:
                units.append(para)
                continue

            # Level 2: split oversized paragraph into sentences
            sentences = nltk.sent_tokenize(para)
            for sent in sentences:
                if self.count_tokens(sent) <= self.max_tokens:
                    units.append(sent)
                else:
                    # Level 3: split oversized sentence by token budget
                    units.extend(self._fixed_split(sent))

        return units

    def _fixed_split(self, text: str) -> List[str]:
        """Word-boundary-safe fixed-token split (last resort, no overlap)."""
        words = text.split()
        chunks: List[str] = []
        current: List[str] = []
        current_tokens = 0

        for i, word in enumerate(words):
            wt = len(self.enc.encode((" " if current else "") + word))
            if current and current_tokens + wt > self.max_tokens:
                chunks.append(" ".join(current))
                current = []
                current_tokens = 0
            current.append(word)
            current_tokens += wt

        if current:
            chunks.append(" ".join(current))

        return chunks

    def _group_units(self, units: List[str]) -> List[str]:
        """Greedily pack atomic units into chunks of at most max_tokens."""
        if not units:
            return []

        chunks: List[str] = []
        current: List[str] = []
        current_tokens = 0

        for unit in units:
            ut = self.count_tokens(unit)

            if current and current_tokens + ut > self.max_tokens:
                chunks.append(" ".join(current))
                current = []
                current_tokens = 0

            current.append(unit)
            current_tokens += ut

        if current:
            chunks.append(" ".join(current))

        return chunks
