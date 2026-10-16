"""Pydantic models for everything the LLM returns and the app produces."""

from typing import Literal

from pydantic import BaseModel, Field

Confidence = Literal["HIGH", "MEDIUM", "LOW"]
# Result of checking a value against the Tesseract OCR text of the page(s).
Grounding = Literal["match", "no_match", "n/a"]


# ---------- Returned by the LLM ----------

class ExtractedField(BaseModel):
    """Step 5 output format: {field, value, confidence, evidence}."""

    field: str
    value: str | None = Field(
        default=None, description="Exact value as written; null if not present."
    )
    confidence: Confidence = "LOW"
    evidence: str = Field(
        default="", description="Where on the document the value appears, e.g. 'top-right section'."
    )
    page: int | None = None


class ExtractionResult(BaseModel):
    fields: list[ExtractedField] = []


class StructuredAnswer(BaseModel):
    answer: str = Field(description="Natural-language answer to the question.")
    value: str | None = Field(
        default=None, description="Exact text span from the document that answers the question."
    )
    confidence: Confidence = "LOW"
    evidence: str = ""
    page: int | None = None


class FieldPresence(BaseModel):
    field: str
    present: bool
    value: str | None = None
    evidence: str = ""
    page: int | None = None


class FieldPresenceResult(BaseModel):
    fields: list[FieldPresence] = []


class LineItem(BaseModel):
    description: str = ""
    amount: float | None = None


class AmountExtraction(BaseModel):
    """Numbers the LLM reads from an invoice/receipt. Arithmetic is done in Python."""

    currency: str | None = None
    line_items: list[LineItem] = []
    subtotal: float | None = None
    tax: float | None = None
    total: float | None = None
    evidence: str = ""


class OtherIssue(BaseModel):
    issue: str
    evidence: str = ""


class OtherIssues(BaseModel):
    issues: list[OtherIssue] = []


# ---------- Produced by the application ----------

class BBox(BaseModel):
    page: int
    left: int
    top: int
    width: int
    height: int


class GroundedField(ExtractedField):
    """An extracted field plus the application's own verification against OCR."""

    ocr_grounding: Grounding = "n/a"
    bbox: BBox | None = None


class MissingFieldReport(BaseModel):
    template: str
    required_fields: list[str]
    results: list[FieldPresence]

    @property
    def missing(self) -> list[str]:
        return [r.field for r in self.results if not r.present]


class ConsistencyCheck(BaseModel):
    name: str
    passed: bool | None = Field(
        description="True/False, or None when the required values were not found."
    )
    expected: float | None = None
    found: float | None = None
    details: str = ""
