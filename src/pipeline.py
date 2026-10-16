"""Document AI + LLM pipeline (Steps 4-9)."""

from dataclasses import dataclass, field
from typing import TypeVar

from PIL import Image
from pydantic import BaseModel

from src import prompts
from src.evidence import NOT_FOUND, check_grounding, locate_value
from src.json_utils import LLMOutputError, parse_model
from src.llm.base import LLMBackend, Part
from src.ocr import OCRResult
from src.schemas import (
    AmountExtraction,
    ConsistencyCheck,
    ExtractionResult,
    FieldPresence,
    FieldPresenceResult,
    GroundedField,
    MissingFieldReport,
    OtherIssue,
    OtherIssues,
    StructuredAnswer,
)

T = TypeVar("T", bound=BaseModel)
TOLERANCE = 0.01  # currency rounding tolerance for arithmetic checks


@dataclass
class PageInput:
    page_number: int
    image: Image.Image
    ocr: OCRResult


@dataclass
class CallInfo:
    """Transparency data shown in the UI: raw output and latency."""

    raw: str = ""
    latency_s: float = 0.0


@dataclass
class QAResult:
    method: str
    answer: str
    latency_s: float
    raw: str = ""
    structured: StructuredAnswer | None = None
    grounding: str = "n/a"
    bbox: object = None


@dataclass
class ExtractionOutput:
    fields: list[GroundedField]
    call: CallInfo = field(default_factory=CallInfo)


# ---------- helpers ----------

def image_parts(pages: list[PageInput]) -> list[Part]:
    parts: list[Part] = []
    for page in pages:
        parts.append(f"Page {page.page_number}:")
        parts.append(page.image)
    return parts


def ocr_text_block(pages: list[PageInput]) -> str:
    blocks = [
        f"--- OCR text of page {p.page_number} ---\n{p.ocr.text or '[no text recognised]'}"
        for p in pages
    ]
    return "\n\n".join(blocks)


def all_ocr_text(pages: list[PageInput]) -> str:
    return "\n".join(p.ocr.text for p in pages)


def _page_for(pages: list[PageInput], page_number: int | None) -> PageInput | None:
    for p in pages:
        if p.page_number == page_number:
            return p
    return pages[0] if len(pages) == 1 else None


def locate(value: str | None, pages: list[PageInput], page_number: int | None):
    """Find the value's box: on the cited page first, then on any page."""

    cited = _page_for(pages, page_number)
    candidates = [cited] if cited else []
    candidates += [p for p in pages if p is not cited]
    for page in candidates:
        box = locate_value(value, page.ocr.words, page.page_number)
        if box:
            return box
    return None


def call_structured(
    backend: LLMBackend,
    parts: list[Part],
    model: type[T],
    list_key: str | None = None,
) -> tuple[T, CallInfo]:
    """Call the LLM in JSON mode, validate with Pydantic, retry once on bad JSON."""

    response = backend.generate(parts, json_mode=True)
    info = CallInfo(raw=response.text, latency_s=response.latency_s)
    try:
        return parse_model(response.text, model, list_key), info
    except LLMOutputError:
        retry = backend.generate(
            parts + ["Your previous reply was not valid JSON for the requested format. Reply again with valid JSON only."],
            json_mode=True,
        )
        info = CallInfo(raw=retry.text, latency_s=round(info.latency_s + retry.latency_s, 3))
        return parse_model(retry.text, model, list_key), info


def _clean_short(text: str) -> str:
    text = text.strip().strip('"').strip()
    return text.splitlines()[0].strip() if text else text


def _ground_fields(result_fields, pages: list[PageInput]) -> list[GroundedField]:
    text = all_ocr_text(pages)
    grounded = []
    for f in result_fields:
        value = f.value if f.value and f.value.strip().upper() != NOT_FOUND else None
        grounded.append(
            GroundedField(
                **{**f.model_dump(), "value": value},
                ocr_grounding=check_grounding(value, text) if value else "n/a",
                bbox=locate(value, pages, f.page) if value else None,
            )
        )
    return grounded


# ---------- Step 3/4/8/10: question answering ----------

def answer_ocr_only(backend: LLMBackend, pages: list[PageInput], question: str) -> QAResult:
    """OCR baseline: the LLM only sees Tesseract text, not the image."""

    prompt = ocr_text_block(pages) + "\n\n" + prompts.SHORT_ANSWER_OCR.format(question=question)
    r = backend.generate([prompt])
    answer = _clean_short(r.text)
    return QAResult("ocr_llm", answer, r.latency_s, r.text, grounding=check_grounding(answer, all_ocr_text(pages)))


def answer_multimodal(backend: LLMBackend, pages: list[PageInput], question: str) -> QAResult:
    """Multimodal: the LLM sees only the document image(s) and the question."""

    r = backend.generate(image_parts(pages) + [prompts.SHORT_ANSWER_IMAGE.format(question=question)])
    answer = _clean_short(r.text)
    return QAResult("multimodal", answer, r.latency_s, r.text, grounding=check_grounding(answer, all_ocr_text(pages)))


def answer_hybrid(backend: LLMBackend, pages: list[PageInput], question: str) -> QAResult:
    """OCR + multimodal: image(s) plus the Tesseract text."""

    parts = image_parts(pages) + [ocr_text_block(pages), prompts.SHORT_ANSWER_HYBRID.format(question=question)]
    r = backend.generate(parts)
    answer = _clean_short(r.text)
    return QAResult("hybrid", answer, r.latency_s, r.text, grounding=check_grounding(answer, all_ocr_text(pages)))


def answer_structured(backend: LLMBackend, pages: list[PageInput], question: str) -> QAResult:
    """Structured answer with value, confidence, evidence and page."""

    parts = image_parts(pages) + [prompts.STRUCTURED_ANSWER.format(question=question)]
    result, info = call_structured(backend, parts, StructuredAnswer)
    value = result.value if result.value and result.value.strip().upper() != NOT_FOUND else None
    return QAResult(
        method="structured",
        answer=value or NOT_FOUND,
        latency_s=info.latency_s,
        raw=info.raw,
        structured=result,
        grounding=check_grounding(value, all_ocr_text(pages)) if value else "n/a",
        bbox=locate(value, pages, result.page) if value else None,
    )


QA_METHODS = {
    "ocr_llm": answer_ocr_only,
    "multimodal": answer_multimodal,
    "hybrid": answer_hybrid,
    "structured": answer_structured,
}


def summarize(backend: LLMBackend, pages: list[PageInput]) -> tuple[str, float]:
    r = backend.generate(image_parts(pages) + [prompts.SUMMARY])
    return r.text, r.latency_s


# ---------- Step 5: structured extraction ----------

def extract_fields(
    backend: LLMBackend,
    pages: list[PageInput],
    field_names: list[str] | None = None,
) -> ExtractionOutput:
    if field_names:
        prompt = prompts.EXTRACT_TEMPLATE.format(field_list=", ".join(field_names))
    else:
        prompt = prompts.EXTRACT_ALL
    result, info = call_structured(backend, image_parts(pages) + [prompt], ExtractionResult, "fields")
    return ExtractionOutput(_ground_fields(result.fields, pages), info)


# ---------- Step 6: missing-field detection ----------

def detect_missing_fields(
    backend: LLMBackend,
    pages: list[PageInput],
    template_name: str,
    required_fields: list[str],
) -> tuple[MissingFieldReport, CallInfo]:
    prompt = prompts.MISSING_FIELDS.format(field_list=", ".join(required_fields))
    result, info = call_structured(backend, image_parts(pages) + [prompt], FieldPresenceResult, "fields")

    # Keep exactly the required fields, in order. A field the model skipped is
    # reported as missing rather than silently dropped.
    by_name = {r.field.strip().lower(): r for r in result.fields}
    ordered = []
    for name in required_fields:
        item = by_name.get(name.strip().lower())
        if item is None:
            item = FieldPresence(field=name, present=False, evidence="Not reported by the model.")
        else:
            item.field = name
        ordered.append(item)

    report = MissingFieldReport(template=template_name, required_fields=required_fields, results=ordered)
    return report, info


# ---------- Step 7: consistency checking ----------

def arithmetic_checks(amounts: AmountExtraction) -> list[ConsistencyCheck]:
    """Deterministic checks on the numbers the LLM read from the document."""

    checks = []

    if amounts.subtotal is not None and amounts.tax is not None and amounts.total is not None:
        expected = round(amounts.subtotal + amounts.tax, 2)
        checks.append(
            ConsistencyCheck(
                name="Total = Subtotal + Tax",
                passed=abs(expected - amounts.total) <= TOLERANCE,
                expected=expected,
                found=amounts.total,
                details=f"{amounts.subtotal} + {amounts.tax} = {expected}; document total is {amounts.total}.",
            )
        )
    else:
        missing = [n for n in ("subtotal", "tax", "total") if getattr(amounts, n) is None]
        checks.append(
            ConsistencyCheck(
                name="Total = Subtotal + Tax",
                passed=None,
                details=f"Cannot check: {', '.join(missing)} not found on the document.",
            )
        )

    item_amounts = [li.amount for li in amounts.line_items if li.amount is not None]
    if item_amounts and amounts.subtotal is not None:
        expected = round(sum(item_amounts), 2)
        checks.append(
            ConsistencyCheck(
                name="Subtotal = Sum of line items",
                passed=abs(expected - amounts.subtotal) <= TOLERANCE,
                expected=expected,
                found=amounts.subtotal,
                details=f"{len(item_amounts)} line item(s) sum to {expected}; document subtotal is {amounts.subtotal}.",
            )
        )
    elif item_amounts and amounts.subtotal is None and amounts.tax is None and amounts.total is not None:
        expected = round(sum(item_amounts), 2)
        checks.append(
            ConsistencyCheck(
                name="Total = Sum of line items (no subtotal/tax printed)",
                passed=abs(expected - amounts.total) <= TOLERANCE,
                expected=expected,
                found=amounts.total,
                details=f"{len(item_amounts)} line item(s) sum to {expected}; document total is {amounts.total}.",
            )
        )

    return checks


def check_consistency(
    backend: LLMBackend,
    pages: list[PageInput],
    include_llm_review: bool = False,
) -> tuple[AmountExtraction, list[ConsistencyCheck], list[OtherIssue], CallInfo]:
    amounts, info = call_structured(backend, image_parts(pages) + [prompts.AMOUNTS], AmountExtraction)
    checks = arithmetic_checks(amounts)

    issues: list[OtherIssue] = []
    if include_llm_review:
        review, review_info = call_structured(backend, image_parts(pages) + [prompts.OTHER_ISSUES], OtherIssues, "issues")
        issues = review.issues
        info = CallInfo(raw=info.raw + "\n\n" + review_info.raw, latency_s=round(info.latency_s + review_info.latency_s, 3))

    return amounts, checks, issues, info
