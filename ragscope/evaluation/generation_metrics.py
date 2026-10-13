"""Deterministic answer overlap metrics. No network/NLTK data at import."""
from __future__ import annotations

import re
import string
from collections import Counter

_ARTICLE_RE = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def _normalize(text):
    text = text.lower().translate(_PUNCT_TABLE)
    return " ".join(_ARTICLE_RE.sub(" ", text).split())


def _references(ground_truth):
    return [ground_truth] if isinstance(ground_truth, str) else list(ground_truth)


def exact_match(prediction, ground_truth):
    return max((float(_normalize(prediction) == _normalize(ref)) for ref in _references(ground_truth)), default=0.0)


def _pair_f1(prediction, reference):
    pred, gold = _normalize(prediction).split(), _normalize(reference).split()
    if not pred or not gold:
        return float(pred == gold)
    common = sum((Counter(pred) & Counter(gold)).values())
    return 2 * common / (len(pred) + len(gold))


def token_f1(prediction, ground_truth):
    """Best overlap across all legitimate answer references, not first annotator."""
    return max((_pair_f1(prediction, ref) for ref in _references(ground_truth)), default=0.0)


def _pair_rouge_l(prediction, reference):
    """Unstemmed LCS F1 over the same normalized tokens as EM/token F1."""
    pred, gold = _normalize(prediction).split(), _normalize(reference).split()
    if not pred or not gold:
        return float(pred == gold)
    previous = [0] * (len(gold) + 1)
    for token in pred:
        current = [0]
        for i, target in enumerate(gold, 1):
            current.append(previous[i-1] + 1 if token == target else max(previous[i], current[-1]))
        previous = current
    return 2 * previous[-1] / (len(pred) + len(gold))


def rouge_l(prediction, ground_truth):
    """v2 ROUGE-L: best multi-reference normalized word LCS F1, no stemming.

    This declared definition differs from legacy rouge-score's stemmed scorer.
    Old CSVs must remain labeled legacy and cannot be silently mixed with v2.
    """
    return max((_pair_rouge_l(prediction, ref) for ref in _references(ground_truth)), default=0.0)


def answer_reference_texts(answer_refs):
    texts = []
    for ref in answer_refs:
        if isinstance(ref, str):
            texts.append(ref)
        elif isinstance(ref, dict):
            if ref.get("unanswerable"):
                continue
            if ref.get("yes_no") is not None:
                texts.append("yes" if ref["yes_no"] else "no")
            elif ref.get("free_form_answer"):
                texts.append(ref["free_form_answer"])
            elif ref.get("extractive_spans"):
                texts.append(", ".join(ref["extractive_spans"]))
            elif isinstance(ref.get("text"), str):
                texts.append(ref["text"])
            elif isinstance(ref.get("answer"), str):
                texts.append(ref["answer"])
    return list(dict.fromkeys(texts))
