"""Step 9 — Evidence grounding.

* ``check_grounding``: is an LLM-produced value actually present in the OCR text?
  (hallucination control — computed by the application, not by the LLM)
* ``locate_value``: find the image region (OCR word boxes) containing a value.
* ``draw_boxes``: highlight those regions on the page image.
"""

import re

from PIL import Image, ImageDraw
from rapidfuzz import fuzz

from src.ocr import OCRWord
from src.schemas import BBox, Grounding

NOT_FOUND = "NOT_FOUND"


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s.]", " ", text)  # keep letters, digits, '.'
    text = re.sub(r"(?<!\d)\.|\.(?!\d)", " ", text)  # drop '.' unless between digits
    return re.sub(r"\s+", " ", text).strip()


def _compact_numbers(text: str) -> str:
    """'1 234.50' -> '1234.50' so '$1,234.50' and '1234.50' compare equal."""

    return re.sub(r"(?<=\d) (?=\d)", "", text)


def check_grounding(value: str | None, ocr_text: str, threshold: int = 90) -> Grounding:
    """'match' if the value is found (exactly or fuzzily) in the OCR text."""

    if not value or value.strip().upper() == NOT_FOUND:
        return "n/a"

    norm_value = _compact_numbers(normalize_text(value))
    norm_text = _compact_numbers(normalize_text(ocr_text))
    if not norm_value or not norm_text:
        return "no_match"

    if norm_value in norm_text:
        return "match"

    # Fuzzy match tolerates small OCR errors; only for non-trivial strings.
    if len(norm_value) >= 4 and fuzz.partial_ratio(norm_value, norm_text) >= threshold:
        return "match"

    return "no_match"


def locate_value(
    value: str | None,
    words: list[OCRWord],
    page: int,
    threshold: float = 85.0,
) -> BBox | None:
    """Return the union box of the OCR word window best matching ``value``."""

    if not value or not words or value.strip().upper() == NOT_FOUND:
        return None

    target = normalize_text(value)
    if not target:
        return None
    n_tokens = max(1, len(target.split()))

    best_score, best_window = 0.0, None
    # Allow windows slightly shorter/longer than the value (OCR splits/merges words).
    for size in {max(1, n_tokens - 1), n_tokens, n_tokens + 1}:
        for start in range(0, len(words) - size + 1):
            window = words[start : start + size]
            candidate = normalize_text(" ".join(w.text for w in window))
            if not candidate:
                continue
            score = fuzz.ratio(target, candidate)
            if score > best_score:
                best_score, best_window = score, window

    if best_window is None or best_score < threshold:
        return None

    left = min(w.left for w in best_window)
    top = min(w.top for w in best_window)
    right = max(w.left + w.width for w in best_window)
    bottom = max(w.top + w.height for w in best_window)
    return BBox(page=page, left=left, top=top, width=right - left, height=bottom - top)


def draw_boxes(image: Image.Image, boxes: list[tuple[BBox, str]]) -> Image.Image:
    """Draw labelled rectangles on a copy of the image."""

    if not boxes:
        return image

    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    stroke = max(3, image.width // 400)
    for box, label in boxes:
        pad = stroke * 2
        rect = (box.left - pad, box.top - pad, box.left + box.width + pad, box.top + box.height + pad)
        draw.rectangle(rect, outline=(230, 40, 40), width=stroke)
        if label:
            draw.text((rect[0], max(0, rect[1] - 14 - stroke)), label[:40], fill=(230, 40, 40))
    return canvas
