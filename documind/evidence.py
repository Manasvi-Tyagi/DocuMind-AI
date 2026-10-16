"""Locate literal evidence in parsed/OCR words, never model-invented boxes."""

import io
import re

from PIL import Image, ImageDraw
import pytesseract

from .config import get_settings


def _tokens(text):
    return re.findall(r"\w+|[^\w\s]", str(text).casefold())


def _words(page):
    existing = getattr(page, "word_boxes", [])
    if existing:
        return existing
    image = Image.open(io.BytesIO(page.image)).convert("RGB")
    executable = get_settings().tesseract_cmd
    if executable:
        pytesseract.pytesseract.tesseract_cmd = executable
    data = pytesseract.image_to_data(
        image, lang="eng", output_type=pytesseract.Output.DICT
    )
    words = []
    for i, text in enumerate(data["text"]):
        if text.strip():
            x, y, w, h = [data[k][i] for k in ("left", "top", "width", "height")]
            words.append(
                {
                    "text": text,
                    "box": [
                        x / image.width,
                        y / image.height,
                        (x + w) / image.width,
                        (y + h) / image.height,
                    ],
                }
            )
    page.word_boxes = words
    return words


def locate_quote(pages, quote, claimed_page=None):
    """Return source boxes only for an exact token-sequence match.

    A located quote confirms text location, not correctness of the extracted value.
    Scanned-image matches depend on OCR and may miss handwriting or unusual layout.
    """
    wanted = _tokens(quote)
    if not wanted:
        return {
            "status": "unverified",
            "matches": [],
            "reason": "No evidence quote supplied.",
        }
    if claimed_page is not None and claimed_page not in {p.number for p in pages}:
        return {
            "status": "unverified",
            "matches": [],
            "reason": "The cited page was not analyzed.",
        }
    matches, errors = [], []
    for page in pages:
        if claimed_page is not None and page.number != claimed_page:
            continue
        try:
            words = _words(page)
        except (
            pytesseract.TesseractNotFoundError,
            pytesseract.TesseractError,
        ) as error:
            errors.append(type(error).__name__)
            continue
        tokens, indices = [], []
        for i, word in enumerate(words):
            parts = _tokens(word["text"])
            tokens.extend(parts)
            indices.extend([i] * len(parts))
        for start in range(len(tokens) - len(wanted) + 1):
            if tokens[start : start + len(wanted)] == wanted:
                boxes = [
                    words[i]["box"] for i in set(indices[start : start + len(wanted)])
                ]
                matches.append(
                    {
                        "page": page.number,
                        "box": [
                            max(0, min(b[0] for b in boxes)),
                            max(0, min(b[1] for b in boxes)),
                            min(1, max(b[2] for b in boxes)),
                            min(1, max(b[3] for b in boxes)),
                        ],
                    }
                )
    return {
        "status": "located" if matches else "unverified",
        "matches": matches,
        "reason": (
            "Exact evidence quote located in parsed/OCR text."
            if matches
            else "Quote not found verbatim in available source text."
            + (" OCR unavailable: " + ", ".join(errors) if errors else "")
        ),
    }


def highlight_matches(page, matches):
    image = Image.open(io.BytesIO(page.image)).convert("RGB")
    draw = ImageDraw.Draw(image)
    for match in matches:
        if match["page"] == page.number:
            x0, y0, x1, y1 = match["box"]
            draw.rectangle(
                (
                    x0 * image.width,
                    y0 * image.height,
                    x1 * image.width,
                    y1 * image.height,
                ),
                outline="#db5e00",
                width=4,
            )
    return image
