"""
Index-building pipeline.

Part 1: chunk documents with three strategies → Parquet.
Part 2: build dense ChromaDB vector indexes.
Part 3: build sparse BM25 indexes.

Usage
-----
# Quick validation with default 'fixed' strategy:
    python scripts/build_indexes.py

# Build a specific strategy:
    python scripts/build_indexes.py --chunking sentence
    python scripts/build_indexes.py --chunking semantic

# Build all three strategies:
    python scripts/build_indexes.py --chunking all
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from loguru import logger
from tqdm import tqdm

from ragscope.ingestion.chunkers import (
    Chunk,
    FixedSizeChunker,
    SemanticChunker,
    SentenceAwareChunker,
)
from ragscope.indexing.dense_index import DenseIndex
from ragscope.indexing.embedding import EmbeddingService
from ragscope.indexing.sparse_index import BM25Index

# ---------------------------------------------------------------------------
# Strategy registry
# ---------------------------------------------------------------------------

# Maps strategy name → (parquet_path, chroma_collection_name)
STRATEGY_MAP: Dict[str, tuple] = {
    "fixed":    ("data/chunks/fixed_512.parquet",      "ragscope_fixed"),
    "sentence": ("data/chunks/sentence_aware.parquet", "ragscope_sentence"),
    "semantic": ("data/chunks/semantic.parquet",       "ragscope_semantic"),
}

# Maps strategy name → chunker factory (called lazily)
CHUNKER_FACTORY = {
    "fixed":    lambda: FixedSizeChunker(chunk_size=512, overlap=64),
    "sentence": lambda: SentenceAwareChunker(max_tokens=400),
    "semantic": lambda: SemanticChunker(max_tokens=450),
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_corpus(path: str = "data/processed/corpus.jsonl") -> List[Dict]:
    p = Path(path)
    if not p.exists():
        logger.error(f"Corpus not found: {p}. Run scripts/ingest.py first.")
        sys.exit(1)
    docs = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    logger.info(f"Loaded {len(docs):,} documents from {p}")
    return docs


def _chunks_to_df(chunks: List[Chunk]) -> pd.DataFrame:
    rows = [asdict(c) for c in chunks]
    for row in rows:
        row["metadata"] = json.dumps(row["metadata"])
    return pd.DataFrame(rows)


def _print_chunk_stats(df: pd.DataFrame, name: str) -> None:
    logger.info(
        f"  {name}: {len(df):,} chunks | "
        f"avg {df['n_tokens'].mean():.1f} tok | "
        f"max {df['n_tokens'].max()} | "
        f"min {df['n_tokens'].min()}"
    )


# ---------------------------------------------------------------------------
# Part 1 — chunking
# ---------------------------------------------------------------------------

def chunk_corpus(chunker, docs: List[Dict], out_path: str) -> pd.DataFrame:
    """Chunk all docs with the given chunker, save to Parquet, and print stats."""
    name = type(chunker).__name__
    all_chunks: List[Chunk] = []

    for doc in tqdm(docs, desc=name, unit="doc"):
        all_chunks.extend(chunker.chunk(doc))

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = _chunks_to_df(all_chunks)
    df.to_parquet(str(out), index=False)
    logger.info(f"Saved → {out}")
    _print_chunk_stats(df, name)
    return df


def run_chunking_pipeline(docs: List[Dict]) -> None:
    """Part 1: chunk the corpus with all three strategies."""
    logger.info("=" * 60)
    logger.info("Part 1: Chunking pipeline")
    for strategy, (parquet_path, _) in STRATEGY_MAP.items():
        logger.info(f"--- {strategy} ---")
        chunk_corpus(CHUNKER_FACTORY[strategy](), docs, parquet_path)
    logger.info("=" * 60)
    logger.info("Chunking pipeline complete.")


def _ensure_chunks(strategy: str) -> pd.DataFrame:
    """Load parquet for *strategy*, building chunks first if necessary."""
    parquet_path, _ = STRATEGY_MAP[strategy]
    if not Path(parquet_path).exists():
        logger.warning(f"Parquet not found for '{strategy}'. Building chunks now…")
        docs = load_corpus()
        chunk_corpus(CHUNKER_FACTORY[strategy](), docs, parquet_path)
    df = pd.read_parquet(parquet_path)
    logger.info(f"Loaded {len(df):,} chunks from {parquet_path}")
    return df


# ---------------------------------------------------------------------------
# Part 2 — dense index (ChromaDB)
# ---------------------------------------------------------------------------

def build_dense(
    strategy: str,
    chunks_df: pd.DataFrame,
    embedding_service: EmbeddingService,
) -> DenseIndex:
    _, collection_name = STRATEGY_MAP[strategy]
    dense = DenseIndex(collection_name=collection_name)

    if dense.exists():
        logger.info(f"  DenseIndex '{collection_name}' already exists — skipping build")
    else:
        dense.build(chunks_df, embedding_service)

    n_vec = dense.collection.count()
    dim = embedding_service.get_dimension()
    logger.info(f"  Dense: {n_vec:,} vectors | dim={dim}")
    return dense


# ---------------------------------------------------------------------------
# Part 3 — sparse index (BM25)
# ---------------------------------------------------------------------------

def build_sparse(strategy: str, chunks_df: pd.DataFrame) -> BM25Index:
    bm25_path = f"data/indexes/bm25_{strategy}.pkl"

    if Path(bm25_path).exists():
        logger.info(f"  BM25 index already exists at {bm25_path} — reusing")
        bm25 = BM25Index.load(bm25_path)
    else:
        bm25 = BM25Index(index_path=bm25_path)
        bm25.build(chunks_df)

    logger.info(f"  BM25: {len(bm25.chunks):,} chunks")
    return bm25


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def build_strategy(strategy: str, embedding_service: EmbeddingService) -> None:
    """End-to-end index build for a single chunking strategy."""
    logger.info("=" * 60)
    logger.info(f"Building indexes for strategy: '{strategy}'")

    chunks_df = _ensure_chunks(strategy)
    _print_chunk_stats(chunks_df, strategy)

    logger.info("--- Part 2: Dense index ---")
    build_dense(strategy, chunks_df, embedding_service)

    logger.info("--- Part 3: Sparse index ---")
    build_sparse(strategy, chunks_df)

    logger.info(f"Strategy '{strategy}' complete.")


def build_all_strategies(embedding_service: EmbeddingService) -> None:
    """Build dense + sparse indexes for all three chunking strategies."""
    for strategy in STRATEGY_MAP:
        build_strategy(strategy, embedding_service)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build RAGScope chunk + vector + BM25 indexes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--chunking",
        choices=["fixed", "sentence", "semantic", "all"],
        default="fixed",
        help="Chunking strategy to build indexes for ('all' builds every strategy).",
    )
    args = parser.parse_args()

    logger.info(f"build_indexes.py — strategy: {args.chunking}")
    svc = EmbeddingService()

    if args.chunking == "all":
        build_all_strategies(svc)
    else:
        build_strategy(args.chunking, svc)

    logger.info("Done.")
