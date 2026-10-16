"""Offline regression checks for the modular document services."""

import io
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import pymupdf

from documind import ai
from documind.config import Settings
from documind.extraction import extract_document_text
from documind.validation import check_document_consistency, detect_missing_fields


def completion(content, finish_reason="stop"):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content), finish_reason=finish_reason
            )
        ]
    )


class ServiceTests(unittest.TestCase):
    def test_pdf_text_extraction(self):
        with pymupdf.open() as pdf:
            page = pdf.new_page()
            page.insert_text((72, 72), "Invoice INV-1045 Total 100")
            upload = io.BytesIO(pdf.tobytes())
        upload.name = "invoice.pdf"
        self.assertIn("Invoice INV-1045 Total 100", extract_document_text(upload))

    @patch.object(
        ai,
        "get_settings",
        return_value=Settings(
            "test-key", "openai/gpt-oss-120b", None, ai_provider="groq"
        ),
    )
    @patch.object(ai, "Groq")
    def test_analysis_flow_retries_truncated_classification(self, groq, settings):
        create = groq.return_value.__enter__.return_value.chat.completions.create
        data = {
            "category": "Invoice",
            "summary": "Invoice for 100",
            "fields": {
                "subtotal": {"value": 100},
                "tax": {"value": 0},
                "total": {"value": 100},
            },
        }
        create.side_effect = [
            completion("", "length"),
            completion("Invoice"),
            completion(json.dumps(data)),
            completion("The total is 100."),
        ]
        category = ai.classify_document("Invoice Total 100")
        document = ai.extract_structured_data("Invoice Total 100", category)
        answer = ai.answer_question("Invoice Total 100", "What is the total?")
        self.assertEqual(category, "Invoice")
        self.assertEqual(document.fields["total"].value, 100)
        self.assertEqual(answer, "The total is 100.")
        calls = create.call_args_list
        self.assertEqual(calls[1].kwargs["max_completion_tokens"], 4096)
        self.assertEqual(calls[0].kwargs["reasoning_effort"], "low")
        self.assertEqual(groq.call_args.kwargs["base_url"], "https://api.groq.com")
        checks = check_document_consistency(document.model_dump(), category)
        self.assertTrue(checks)
        missing = detect_missing_fields(document.model_dump(), category)
        self.assertTrue(
            any(
                row["field"] == "Invoice Number" and row["status"] == "Not found"
                for row in missing
            )
        )

    @patch.object(
        ai,
        "get_settings",
        return_value=Settings("", "openai/gpt-oss-120b", None, ai_provider="groq"),
    )
    def test_missing_key_has_actionable_error(self, settings):
        with self.assertRaisesRegex(ValueError, "API key not found"):
            ai.classify_document("Invoice")


if __name__ == "__main__":
    unittest.main()
