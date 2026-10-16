from typing import Any, Literal
from pydantic import BaseModel, Field, field_validator

ALLOWED_CATEGORIES = {
    "Resume",
    "Invoice",
    "Application Form",
    "Report",
    "Statement",
    "Scanned Form",
    "Certificate",
    "Receipt",
    "Other / Uncertain",
}


class ExtractedField(BaseModel):
    """Validated metadata around a flexible document value."""

    value: Any = None
    confidence: Literal["HIGH", "MEDIUM", "LOW"] = "LOW"
    evidence: str | None = None
    page: int | None = Field(default=None, gt=0, strict=True)
    region: str | None = None

    @field_validator("confidence", mode="before")
    @classmethod
    def normalize_confidence(cls, value):
        return value.strip().upper() if isinstance(value, str) else value


class StructuredDocument(BaseModel):
    category: str
    summary: str
    fields: dict[str, ExtractedField] = Field(default_factory=dict)
    batches: list["StructuredDocument"] = Field(default_factory=list)

    @field_validator("fields", mode="before")
    @classmethod
    def normalize_fields(cls, fields):
        if not isinstance(fields, dict):
            raise ValueError("fields must be a JSON object keyed by field name")
        return {
            name: (
                value
                if isinstance(value, (ExtractedField, dict))
                and (
                    isinstance(value, ExtractedField)
                    or any(
                        key in value
                        for key in ("value", "confidence", "evidence", "page", "region")
                    )
                )
                else {"value": value}
            )
            for name, value in fields.items()
        }
