import json
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict

import requests
from loguru import logger
from tqdm import tqdm

SQUAD_URL = "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json"
ARXIV_API = "http://export.arxiv.org/api/query"
ATOM_NS = "http://www.w3.org/2005/Atom"


def _atom(tag: str) -> str:
    return f"{{{ATOM_NS}}}{tag}"


def load_squad_corpus(
    max_docs: int = 20000,
    cache_dir: str = "data/raw/",
) -> List[Dict]:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = cache_dir / "squad_corpus.jsonl"
    eval_path = cache_dir / "squad_eval_queries.jsonl"

    if corpus_path.exists() and eval_path.exists():
        logger.info(f"SQuAD cache hit — reading {corpus_path}")
        docs = [json.loads(l) for l in corpus_path.read_text(encoding="utf-8").splitlines() if l]
        n_eval = sum(1 for _ in eval_path.read_text(encoding="utf-8").splitlines() if _)
        logger.info(f"Loaded {len(docs)} docs, {n_eval} eval queries from cache")
        return docs

    # --- download raw JSON ---
    raw_path = cache_dir / "train-v2.0.json"
    if not raw_path.exists():
        logger.info(f"Downloading SQuAD 2.0 from {SQUAD_URL}")
        try:
            resp = requests.get(SQUAD_URL, stream=True, timeout=120)
            resp.raise_for_status()
            total = int(resp.headers.get("content-length", 0))
            with open(raw_path, "wb") as fh, tqdm(
                total=total, unit="B", unit_scale=True, desc="squad download"
            ) as pbar:
                for chunk in resp.iter_content(chunk_size=65536):
                    fh.write(chunk)
                    pbar.update(len(chunk))
        except Exception as exc:
            logger.warning(f"SQuAD download failed: {exc}")
            return []

    # --- parse ---
    with open(raw_path, encoding="utf-8") as fh:
        data = json.load(fh)

    seen_contexts: set = set()
    corpus: List[Dict] = []
    eval_queries: List[Dict] = []
    doc_counter = 0
    query_counter = 0

    for article in tqdm(data["data"], desc="parsing SQuAD"):
        title = article.get("title", "")
        for para in article.get("paragraphs", []):
            context = para["context"].strip()
            if context in seen_contexts:
                continue
            seen_contexts.add(context)

            current_doc_id = f"squad_{doc_counter}"
            corpus.append(
                {"doc_id": current_doc_id, "title": title, "text": context, "source": "squad"}
            )

            # first answerable QA pair → eval query
            for qa in para.get("qas", []):
                if not qa.get("is_impossible", True):
                    answers = qa.get("answers", [])
                    if answers:
                        eval_queries.append(
                            {
                                "query_id": f"squad_q_{query_counter}",
                                "question": qa["question"],
                                "answer": answers[0]["text"],
                                "gold_doc_id": current_doc_id,
                            }
                        )
                        query_counter += 1
                        break  # one query per paragraph

            doc_counter += 1
            if len(corpus) >= max_docs:
                break
        if len(corpus) >= max_docs:
            break

    # --- save ---
    with open(corpus_path, "w", encoding="utf-8") as fh:
        for doc in corpus:
            fh.write(json.dumps(doc, ensure_ascii=False) + "\n")
    with open(eval_path, "w", encoding="utf-8") as fh:
        for q in eval_queries:
            fh.write(json.dumps(q, ensure_ascii=False) + "\n")

    logger.info(f"SQuAD → corpus: {len(corpus)} docs | eval queries: {len(eval_queries)}")
    return corpus


def load_arxiv_corpus(
    categories: List[str] = None,
    max_docs: int = 3000,
    cache_dir: str = "data/raw/",
) -> List[Dict]:
    if categories is None:
        categories = ["cs.LG"]

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = cache_dir / "arxiv_corpus.jsonl"

    if corpus_path.exists():
        logger.info(f"arXiv cache hit — reading {corpus_path}")
        docs = [json.loads(l) for l in corpus_path.read_text(encoding="utf-8").splitlines() if l]
        logger.info(f"Loaded {len(docs)} arXiv docs from cache")
        return docs

    docs: List[Dict] = []
    global_id = 0
    batch_size = 300
    per_category = max(1, max_docs // len(categories))

    for category in categories:
        cat_count = 0
        start = 0

        with tqdm(total=per_category, desc=f"arXiv {category}") as pbar:
            while cat_count < per_category:
                n_want = min(batch_size, per_category - cat_count)
                params = {
                    "search_query": f"cat:{category}",
                    "start": start,
                    "max_results": n_want,
                }
                try:
                    resp = requests.get(ARXIV_API, params=params, timeout=60)
                    resp.raise_for_status()
                except Exception as exc:
                    logger.warning(
                        f"arXiv API error (cat={category}, start={start}): {exc}"
                    )
                    break

                root = ET.fromstring(resp.content)
                entries = root.findall(_atom("entry"))
                if not entries:
                    break

                for entry in entries:
                    if cat_count >= per_category:
                        break

                    arxiv_id_el = entry.find(_atom("id"))
                    title_el = entry.find(_atom("title"))
                    summary_el = entry.find(_atom("summary"))
                    published_el = entry.find(_atom("published"))
                    cat_els = entry.findall(_atom("category"))

                    if title_el is None or summary_el is None:
                        continue

                    summary_text = (summary_el.text or "").strip().replace("\n", " ")
                    if not summary_text:
                        continue

                    docs.append(
                        {
                            "doc_id": f"arxiv_{global_id}",
                            "title": (title_el.text or "").strip().replace("\n", " "),
                            "text": summary_text,
                            "source": "arxiv",
                            "metadata": {
                                "arxiv_id": (arxiv_id_el.text or "").strip()
                                if arxiv_id_el is not None
                                else "",
                                "published": (published_el.text or "").strip()
                                if published_el is not None
                                else "",
                                "categories": [el.get("term", "") for el in cat_els],
                            },
                        }
                    )
                    global_id += 1
                    cat_count += 1
                    pbar.update(1)

                start += len(entries)
                if len(entries) < n_want:
                    break
                time.sleep(3)  # arXiv polite delay between pages

    with open(corpus_path, "w", encoding="utf-8") as fh:
        for doc in docs:
            fh.write(json.dumps(doc, ensure_ascii=False) + "\n")

    logger.info(f"arXiv → corpus: {len(docs)} docs saved to {corpus_path}")
    return docs
