"""
Generation quality metrics for RAGScope evaluation.

All string-comparison metrics apply the same SQuAD-style normalisation:
lowercase → strip articles → strip punctuation → collapse whitespace.
"""

from __future__ import annotations

import re
import string
from collections import Counter
from functools import lru_cache
from typing import Optional

from rouge_score import rouge_scorer

# ---------------------------------------------------------------------------
# Shared normalisation
# ---------------------------------------------------------------------------

_ARTICLE_RE = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def _normalize(text: str) -> str:
    """SQuAD-style text normalisation used by all metrics."""
    text = text.lower()
    text = _ARTICLE_RE.sub(" ", text)          # strip articles
    text = text.translate(_PUNCT_TABLE)         # strip punctuation
    text = " ".join(text.split())               # collapse whitespace
    return text


# ---------------------------------------------------------------------------
# Exact Match
# ---------------------------------------------------------------------------

def exact_match(prediction: str, ground_truth: str) -> float:
    """Binary 1/0: normalised prediction equals normalised ground truth.

    Normalisation: lowercase → strip articles (a/an/the) →
    strip punctuation → collapse whitespace.

    Formula::

        EM = 1 if norm(prediction) == norm(ground_truth) else 0
    """
    return float(_normalize(prediction) == _normalize(ground_truth))


# ---------------------------------------------------------------------------
# Token F1
# ---------------------------------------------------------------------------

def token_f1(prediction: str, ground_truth: str) -> float:
    """Word-overlap F1 score (SQuAD official evaluation).

    Formula::

        common  = Σ min(count_pred(t), count_gold(t))  for t in vocab
        P       = common / |pred_tokens|
        R       = common / |gold_tokens|
        F1      = 2·P·R / (P + R)      (0 if P + R == 0)

    Both sequences are normalised before tokenisation.
    """
    pred_tokens = _normalize(prediction).split()
    gold_tokens = _normalize(ground_truth).split()

    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0

    pred_counter = Counter(pred_tokens)
    gold_counter = Counter(gold_tokens)
    common = sum((pred_counter & gold_counter).values())

    precision = common / len(pred_tokens)
    recall    = common / len(gold_tokens)

    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


# ---------------------------------------------------------------------------
# ROUGE-L
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _get_scorer() -> rouge_scorer.RougeScorer:
    return rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)


def rouge_l(prediction: str, ground_truth: str) -> float:
    """ROUGE-L F1 score (Longest Common Subsequence based).

    Uses the *rouge-score* library.

    Formula::

        Precision_lcs = LCS(pred, gold) / |pred|
        Recall_lcs    = LCS(pred, gold) / |gold|
        ROUGE-L F1    = 2·P_lcs·R_lcs / (P_lcs + R_lcs)
    """
    scores = _get_scorer().score(target=ground_truth, prediction=prediction)
    return scores["rougeL"].fmeasure
