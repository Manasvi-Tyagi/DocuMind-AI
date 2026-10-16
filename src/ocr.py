"""Step 3 — OCR with Tesseract (word text + bounding boxes + confidence)."""

import time
from dataclasses import dataclass, field

import pytesseract
from PIL import Image
from pytesseract import Output

from src.config import settings

if settings.tesseract_cmd:
    pytesseract.pytesseract.tesseract_cmd = settings.tesseract_cmd


@dataclass
class OCRWord:
    text: str
    confidence: float
    left: int
    top: int
    width: int
    height: int
    line_key: tuple[int, int, int]


@dataclass
class OCRResult:
    text: str
    words: list[OCRWord] = field(default_factory=list)
    mean_confidence: float = 0.0
    latency_s: float = 0.0


def run_ocr(image: Image.Image, lang: str | None = None) -> OCRResult:
    start = time.perf_counter()
    data = pytesseract.image_to_data(
        image,
        lang=lang or settings.tesseract_lang,
        output_type=Output.DICT,
    )

    words: list[OCRWord] = []
    for i, raw_text in enumerate(data["text"]):
        text = raw_text.strip()
        conf = float(data["conf"][i])
        if not text or conf < 0:  # conf == -1 marks non-word layout rows
            continue
        words.append(
            OCRWord(
                text=text,
                confidence=conf,
                left=int(data["left"][i]),
                top=int(data["top"][i]),
                width=int(data["width"][i]),
                height=int(data["height"][i]),
                line_key=(int(data["block_num"][i]), int(data["par_num"][i]), int(data["line_num"][i])),
            )
        )

    # Rebuild text line by line in Tesseract's reading order.
    lines: list[str] = []
    current_key = None
    for word in words:
        if word.line_key != current_key:
            lines.append(word.text)
            current_key = word.line_key
        else:
            lines[-1] += " " + word.text

    mean_conf = sum(w.confidence for w in words) / len(words) if words else 0.0
    return OCRResult(
        text="\n".join(lines),
        words=words,
        mean_confidence=round(mean_conf, 1),
        latency_s=round(time.perf_counter() - start, 3),
    )
