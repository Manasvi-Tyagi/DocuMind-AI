import io

import pymupdf
import pytest
from PIL import Image, ImageDraw

from src.document_processor import crop_margins, normalize_resolution, process_document
from src.evidence import check_grounding, locate_value
from src.json_utils import LLMOutputError, parse_model
from src.llm.base import LLMBackend, LLMResponse
from src.metrics import anls, character_error_rate, exact_match, word_error_rate
from src.ocr import run_ocr
from src.pipeline import (
    PageInput,
    answer_structured,
    arithmetic_checks,
    detect_missing_fields,
    extract_fields,
)
from src.schemas import AmountExtraction, ExtractionResult, LineItem


def make_pdf(lines: list[str]) -> bytes:
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    for i, line in enumerate(lines):
        page.insert_text((60, 80 + 30 * i), line, fontsize=14)
    data = doc.tobytes()
    doc.close()
    return data


@pytest.fixture(scope="module")
def invoice_page() -> PageInput:
    pdf = make_pdf(["INVOICE", "Invoice Number: INV-1045", "Subtotal: 100.00", "Tax: 10.00", "Total: 110.00"])
    page = process_document("x.pdf", pdf, max_pages=5, dpi=200, target_long_side=2000, remove_margins=False)[0]
    return PageInput(page.page_number, page.image, run_ocr(page.image))


class StubBackend(LLMBackend):
    """Returns canned replies in order — lets us test the pipeline without a model."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def generate(self, parts, json_mode=False):
        self.calls.append((parts, json_mode))
        return LLMResponse(self.replies.pop(0), 0.01)


# ---------- preprocessing ----------

def test_pdf_rendering_and_normalization():
    pages = process_document("a.pdf", make_pdf(["Hello"]), 5, 200, 1500, False)
    assert len(pages) == 1
    assert max(pages[0].image.size) == 1500
    assert "Hello" in pages[0].embedded_text


def test_page_limit():
    with pytest.raises(ValueError):
        process_document("a.pdf", make_pdf(["x"]), 0, 200, 1500, False)


def test_image_upload():
    buf = io.BytesIO()
    Image.new("RGB", (400, 300), "white").save(buf, "PNG")
    pages = process_document("scan.png", buf.getvalue(), 5, 200, 800, False)
    assert pages[0].image.size == (800, 600)


def test_unsupported_type():
    with pytest.raises(ValueError):
        process_document("a.docx", b"xx", 5, 200, 800, False)


def test_crop_margins_removes_border():
    img = Image.new("RGB", (1000, 1000), "white")
    ImageDraw.Draw(img).rectangle((300, 300, 700, 700), fill="black")
    cropped = crop_margins(img, padding=10, min_content_fraction=0.05)
    assert cropped.size == (421, 421)


def test_crop_margins_skips_nearly_empty_page():
    img = Image.new("RGB", (1000, 1000), "white")
    ImageDraw.Draw(img).rectangle((500, 500, 510, 510), fill="black")
    assert crop_margins(img).size == (1000, 1000)


def test_normalize_upscales():
    assert normalize_resolution(Image.new("RGB", (500, 250)), 1000).size == (1000, 500)


# ---------- OCR + evidence ----------

def test_tesseract_reads_rendered_text(invoice_page):
    assert "INV-1045" in invoice_page.ocr.text
    assert invoice_page.ocr.words


def test_locate_value(invoice_page):
    box = locate_value("INV-1045", invoice_page.ocr.words, page=1)
    assert box is not None and box.width > 0 and box.page == 1
    assert locate_value("completely absent phrase", invoice_page.ocr.words, page=1) is None


def test_grounding(invoice_page):
    assert check_grounding("110.00", invoice_page.ocr.text) == "match"
    assert check_grounding("$ 999.99", invoice_page.ocr.text) == "no_match"
    assert check_grounding(None, invoice_page.ocr.text) == "n/a"


# ---------- metrics ----------

def test_anls():
    assert anls("Seoul, Korea", ["seoul, korea"]) == 1.0
    assert anls("totally different", ["NPT"]) == 0.0
    assert 0.5 < anls("Seoul Korea", ["Seoul, Korea"]) < 1.0


def test_exact_match_and_error_rates():
    assert exact_match(" NPT ", ["npt"]) == 1.0
    assert character_error_rate("abcd", "abce") == 0.25
    assert word_error_rate("a b c", "a b d") == pytest.approx(1 / 3)


# ---------- JSON parsing ----------

def test_parse_fenced_json_and_bare_list():
    text = '```json\n[{"field": "Name", "value": "A", "confidence": "HIGH", "evidence": "top"}]\n```'
    result = parse_model(text, ExtractionResult, "fields")
    assert result.fields[0].value == "A"


def test_parse_invalid_json_raises():
    with pytest.raises(LLMOutputError):
        parse_model("no json here", ExtractionResult)


# ---------- pipeline with stub backend ----------

def test_structured_answer_grounding_and_bbox(invoice_page):
    stub = StubBackend(['{"answer": "The total is 110.00", "value": "110.00", "confidence": "HIGH", '
                        '"evidence": "bottom", "page": 1}'])
    r = answer_structured(stub, [invoice_page], "What is the total?")
    assert r.answer == "110.00" and r.grounding == "match" and r.bbox is not None
    assert stub.calls[0][1] is True  # JSON mode was requested


def test_structured_answer_retries_bad_json(invoice_page):
    stub = StubBackend(["oops", '{"answer": "Not found", "value": null, "confidence": "LOW", "evidence": ""}'])
    r = answer_structured(stub, [invoice_page], "Who signed?")
    assert r.answer == "NOT_FOUND" and len(stub.calls) == 2


def test_extraction_flags_hallucinated_value(invoice_page):
    stub = StubBackend(['{"fields": [{"field": "Invoice Number", "value": "INV-1045", "confidence": "HIGH", '
                        '"evidence": "top", "page": 1}, {"field": "Buyer", "value": "Acme Corp", '
                        '"confidence": "LOW", "evidence": "", "page": 1}]}'])
    out = extract_fields(stub, [invoice_page], ["Invoice Number", "Buyer"])
    assert out.fields[0].ocr_grounding == "match"
    assert out.fields[1].ocr_grounding == "no_match"


def test_missing_fields_keeps_required_order(invoice_page):
    stub = StubBackend(['{"fields": [{"field": "name", "present": true, "value": "A"}]}'])
    report, _ = detect_missing_fields(stub, [invoice_page], "Custom", ["Name", "Signature"])
    assert [r.field for r in report.results] == ["Name", "Signature"]
    assert report.missing == ["Signature"]


def test_arithmetic_checks():
    ok = arithmetic_checks(AmountExtraction(subtotal=100, tax=10, total=110,
                                            line_items=[LineItem(amount=60), LineItem(amount=40)]))
    assert [c.passed for c in ok] == [True, True]

    bad = arithmetic_checks(AmountExtraction(subtotal=100, tax=10, total=120))
    assert bad[0].passed is False and bad[0].expected == 110

    unknown = arithmetic_checks(AmountExtraction(total=50))
    assert unknown[0].passed is None
