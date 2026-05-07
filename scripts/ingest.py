"""
End-to-end ingestion pipeline:
  load → clean → dedup → merge → save
"""

import json
import sys
from pathlib import Path

# Make project root importable when run as `python scripts/ingest.py`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger

from ragscope.ingestion.cleaner import clean_documents, dedup_documents
from ragscope.ingestion.loaders import load_arxiv_corpus, load_squad_corpus


def main() -> None:
    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = Path("data/raw")

    # --- 1. Load ---
    logger.info("=" * 60)
    logger.info("Step 1/7  Loading SQuAD corpus")
    squad_docs = load_squad_corpus(max_docs=20000, cache_dir=str(raw_dir))

    logger.info("=" * 60)
    logger.info("Step 2/7  Loading arXiv corpus")
    arxiv_docs = load_arxiv_corpus(categories=["cs.LG"], max_docs=3000, cache_dir=str(raw_dir))

    # --- 3. Clean ---
    logger.info("=" * 60)
    logger.info("Step 3/7  Cleaning SQuAD documents")
    squad_clean = clean_documents(squad_docs)

    logger.info("Step 3/7  Cleaning arXiv documents")
    arxiv_clean = clean_documents(arxiv_docs)

    # --- 4. Dedup ---
    logger.info("=" * 60)
    logger.info("Step 4/7  Deduplicating SQuAD documents")
    squad_final = dedup_documents(squad_clean)

    logger.info("Step 4/7  Deduplicating arXiv documents")
    arxiv_final = dedup_documents(arxiv_clean)

    # --- 5. Merge and save corpus ---
    logger.info("=" * 60)
    logger.info("Step 5/7  Merging and saving corpus")
    all_docs = squad_final + arxiv_final
    corpus_path = processed_dir / "corpus.jsonl"
    with open(corpus_path, "w", encoding="utf-8") as fh:
        for doc in all_docs:
            fh.write(json.dumps(doc, ensure_ascii=False) + "\n")
    logger.info(f"Saved {len(all_docs)} documents → {corpus_path}")

    # --- 6. Copy eval queries ---
    logger.info("=" * 60)
    logger.info("Step 6/7  Saving eval queries")
    raw_eval_path = raw_dir / "squad_eval_queries.jsonl"
    out_eval_path = processed_dir / "eval_queries.jsonl"
    eval_queries: list = []

    if raw_eval_path.exists():
        eval_queries = [
            json.loads(line)
            for line in raw_eval_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        logger.warning(f"Eval queries file not found: {raw_eval_path}")

    with open(out_eval_path, "w", encoding="utf-8") as fh:
        for q in eval_queries:
            fh.write(json.dumps(q, ensure_ascii=False) + "\n")
    logger.info(f"Saved {len(eval_queries)} eval queries → {out_eval_path}")

    # --- 7. Final stats ---
    logger.info("=" * 60)
    logger.info("Step 7/7  Final statistics")
    logger.info(f"  Total documents : {len(all_docs)}")
    logger.info(f"    SQuAD         : {len(squad_final)}")
    logger.info(f"    arXiv         : {len(arxiv_final)}")
    logger.info(f"  Eval queries    : {len(eval_queries)}")
    logger.info("Ingestion pipeline complete.")


if __name__ == "__main__":
    main()
