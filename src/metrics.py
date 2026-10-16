"""Step 10 — Evaluation metrics."""

import re

from rapidfuzz.distance import Levenshtein


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip().lower())


def anls(prediction: str, ground_truths: list[str], threshold: float = 0.5) -> float:
    """Average Normalized Levenshtein Similarity — the official DocVQA metric.

    score = max over ground truths of (1 - NL) if NL < threshold else 0,
    where NL is the normalised Levenshtein distance.
    """

    pred = _norm(prediction)
    best = 0.0
    for gt in ground_truths:
        gt_n = _norm(gt)
        if not gt_n and not pred:
            return 1.0
        nl = Levenshtein.normalized_distance(pred, gt_n)
        score = 1.0 - nl if nl < threshold else 0.0
        best = max(best, score)
    return best


def exact_match(prediction: str, ground_truths: list[str]) -> float:
    pred = _norm(prediction)
    return 1.0 if any(pred == _norm(gt) for gt in ground_truths) else 0.0


def character_error_rate(hypothesis: str, reference: str) -> float:
    ref = _norm(reference)
    if not ref:
        return 0.0 if not _norm(hypothesis) else 1.0
    return Levenshtein.distance(_norm(hypothesis), ref) / len(ref)


def word_error_rate(hypothesis: str, reference: str) -> float:
    ref = _norm(reference).split()
    if not ref:
        return 0.0 if not _norm(hypothesis) else 1.0
    return Levenshtein.distance(_norm(hypothesis).split(), ref) / len(ref)


def answer_in_text(ground_truths: list[str], text: str) -> float:
    """1.0 if any ground-truth answer appears verbatim (case-insensitive) in text."""

    t = _norm(text)
    return 1.0 if any(_norm(gt) and _norm(gt) in t for gt in ground_truths) else 0.0
