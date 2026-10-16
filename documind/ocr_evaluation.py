"""OCR transcript metrics and a deterministic, non-LLM QA baseline."""

import re
from .evaluation import edit_distance, normalize

STOP_WORDS = set(
    "a an the what which who is are was were of in on at to for from and all this that document please tell me how much does do".split()
)


def transcript_scores(prediction, reference):
    prediction, reference = normalize(prediction), normalize(reference)
    if not reference:
        raise ValueError("A nonempty reference transcript is required for CER/WER.")
    words = reference.split()
    char_errors = edit_distance(reference, prediction)
    word_errors = edit_distance(words, prediction.split())
    return {
        "character_errors": char_errors,
        "reference_characters": len(reference),
        "word_errors": word_errors,
        "reference_words": len(words),
        "cer": char_errors / len(reference),
        "wer": word_errors / len(words),
    }


def dataset_ocr_text(annotation):
    """Read supplied OCR as a machine reference, never as a human transcript."""
    ocr = annotation.get("ocr_results") or {}
    pages = ocr.get("recognitionResults") or ocr.get("analyzeResult", {}).get(
        "readResults", []
    )
    return "\n".join(
        line.get("text", "") for page in pages for line in page.get("lines", [])
    )


def ocr_only_answer(text, question):
    """Select a label/value line by question overlap, without models or gold answers.

    This deliberately simple baseline cannot reason over tables or infer relations.
    It abstains when no line shares a meaningful question term.
    """
    terms = set(re.findall(r"\w+", question.lower())) - STOP_WORDS
    best, best_score = "", 0
    for line in text.splitlines():
        line = line.strip()
        words = set(re.findall(r"\w+", line.lower()))
        score = len(terms & words) / max(len(words), 1)
        if score > best_score:
            best, best_score = line, score
    if not best:
        return ""
    if ":" in best:
        label, value = best.split(":", 1)
        if value.strip() and terms & set(re.findall(r"\w+", label.lower())):
            return value.strip()
    return best


def reference_answer_present(text, answers):
    """Oracle answer-presence diagnostic; not a predicted QA answer."""
    text = normalize(text)
    return any(
        bool(re.search(r"(?<!\w)" + re.escape(normalize(answer)) + r"(?!\w)", text))
        for answer in answers
        if normalize(answer)
    )
