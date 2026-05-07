import hashlib
import re
import unicodedata
from typing import List, Dict

from loguru import logger

_HTML_TAG = re.compile(r"<[^>]+>")
_WHITESPACE = re.compile(r"\s+")

MIN_TEXT_LEN = 50
MAX_TEXT_LEN = 5000


def _clean_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = _HTML_TAG.sub(" ", text)
    text = _WHITESPACE.sub(" ", text).strip()
    return text


def clean_documents(docs: List[Dict]) -> List[Dict]:
    cleaned: List[Dict] = []
    filtered = 0

    for doc in docs:
        text = _clean_text(doc.get("text", ""))

        if len(text) < MIN_TEXT_LEN or len(text) > MAX_TEXT_LEN:
            filtered += 1
            continue

        cleaned.append({**doc, "text": text})

    logger.info(
        f"clean_documents: {len(docs)} in → {filtered} filtered "
        f"(len<{MIN_TEXT_LEN} or >{MAX_TEXT_LEN}) → {len(cleaned)} out"
    )
    return cleaned


def dedup_documents(docs: List[Dict]) -> List[Dict]:
    seen: set = set()
    deduped: List[Dict] = []

    for doc in docs:
        digest = hashlib.md5(doc.get("text", "").encode("utf-8")).hexdigest()
        if digest not in seen:
            seen.add(digest)
            deduped.append(doc)

    removed = len(docs) - len(deduped)
    logger.info(
        f"dedup_documents: {len(docs)} in → {removed} duplicates removed → {len(deduped)} out"
    )
    return deduped
