"""
Prompt templates and context-formatting utilities for RAGScope.
"""

from __future__ import annotations

from typing import Dict, List

import tiktoken

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

RAG_SYSTEM_PROMPT: str = (
    "You are a precise question-answering assistant.\n"
    "Rules:\n"
    "1. Answer ONLY using the provided context. "
    "If insufficient, say 'I cannot answer based on the provided context.'\n"
    "2. Cite sources using [chunk_id] notation\n"
    "3. Be concise — under 3 sentences unless detail is required\n"
    '4. Output valid JSON: {"answer": str, "citations": [chunk_id], "confidence": float 0-1}'
)

RAG_USER_TEMPLATE: str = (
    "Question: {question}\n\n"
    "Context:\n"
    "{context}\n\n"
    "Answer in JSON format."
)

# ---------------------------------------------------------------------------
# Context formatting
# ---------------------------------------------------------------------------

_ENC = tiktoken.get_encoding("cl100k_base")


def format_context(
    retrieved_chunks: List[Dict],
    max_tokens: int = 3000,
) -> str:
    """Format retrieved chunks into a numbered context string.

    Each chunk is rendered as::

        [chunk_id] <text>

    Chunks are appended in retrieval-score order until *max_tokens* would be
    exceeded; remaining chunks are silently dropped.
    """
    lines: List[str] = []
    total = 0

    for chunk in retrieved_chunks:
        chunk_id = chunk.get("chunk_id", "?")
        text     = chunk.get("text", "").strip()
        line     = f"[{chunk_id}] {text}"
        n_tok    = len(_ENC.encode(line))

        if total + n_tok > max_tokens:
            break

        lines.append(line)
        total += n_tok

    return "\n\n".join(lines)
