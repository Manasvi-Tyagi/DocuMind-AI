"""Offline scoring helpers for short document answers."""


def normalize(text):
    return " ".join(str(text).lower().split())


def edit_distance(a, b):
    """Exact Levenshtein distance using bit vectors (characters or word tokens)."""
    if len(a) > len(b):
        a, b = b, a
    if not a:
        return len(b)
    masks = {}
    for i, token in enumerate(a):
        masks[token] = masks.get(token, 0) | (1 << i)
    positive, negative = (1 << len(a)) - 1, 0
    high_bit = 1 << (len(a) - 1)
    score = len(a)
    for token in b:
        equal = masks.get(token, 0)
        vertical = equal | negative
        horizontal = (((equal & positive) + positive) ^ positive) | equal
        plus = negative | ~(horizontal | positive)
        minus = positive & horizontal
        score += bool(plus & high_bit) - bool(minus & high_bit)
        plus = (plus << 1) | 1
        minus <<= 1
        positive = minus | ~(vertical | plus)
        negative = plus & vertical
    return score


def answer_scores(prediction, references):
    """Best exact match and normalized Levenshtein similarity (0 below 0.5)."""
    if not references:
        raise ValueError(
            "Each evaluation question needs at least one reference answer."
        )
    prediction = normalize(prediction)
    similarities = []
    for reference in references:
        reference = normalize(reference)
        size = max(len(prediction), len(reference))
        similarities.append(
            1 - edit_distance(prediction, reference) / size if size else 1.0
        )
    best = max(similarities)
    return {
        "exact_match": float(prediction in [normalize(r) for r in references]),
        "anls": best if best > 0.5 else 0.0,
    }
