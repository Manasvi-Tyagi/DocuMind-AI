import io
import unittest
from unittest.mock import patch

import pymupdf
from PIL import Image

from documind.ai import document_messages, extract_structured_data
from documind.documents import prepare_document, pages_text
from documind.evaluation import answer_scores
from documind.validation import detect_missing_fields
from types import SimpleNamespace as N


class MultimodalTests(unittest.TestCase):
    def test_selected_pdf_pages_keep_original_numbers(self):
        with pymupdf.open() as pdf:
            for number in range(1, 5):
                pdf.new_page().insert_text((72, 72), f"Source page {number}")
            upload = io.BytesIO(pdf.tobytes())
        upload.name = "sample.pdf"
        pages = prepare_document(upload, [2, 4])
        self.assertEqual([p.number for p in pages], [2, 4])
        self.assertIn("Page 4:", pages_text(pages))
        content = document_messages("Instructions", pages_text(pages), pages)[1][
            "content"
        ]
        images = [part for part in content if part["type"] == "image_url"]
        self.assertEqual(len(images), 2)
        self.assertTrue(
            images[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")
        )
        with self.assertRaises(ValueError):
            prepare_document(upload, [1, 2, 3, 4])

    @patch("documind.documents.pytesseract.image_to_string", return_value="")
    def test_image_without_ocr_still_has_vision_input(self, ocr):
        upload = io.BytesIO()
        Image.new("RGB", (100, 100), "white").save(upload, format="PNG")
        upload.name = "scan.png"
        pages = prepare_document(upload, [1])
        self.assertEqual(pages[0].text, "")
        self.assertTrue(pages[0].image)

    @patch("documind.ai.create_completion")
    def test_vision_extraction_uses_vision_model_and_json_mode(self, create):
        create.return_value = N(
            choices=[
                N(
                    message=N(
                        content='{"category":"Invoice","summary":"Example","fields":{"total":{"value":100,"page":2}}}'
                    )
                )
            ]
        )
        page = N(number=2, data_url="data:image/jpeg;base64,dGVzdA==")
        with patch(
            "documind.ai.get_settings",
            return_value=N(
                groq_api_key="test",
                groq_model="text-model",
                groq_vision_model="vision-model",
            ),
        ):
            result = extract_structured_data(
                "Page 2: Total 100", "Invoice", pages=[page]
            )
        self.assertEqual(result.fields["total"].page, 2)
        self.assertEqual(create.call_args.kwargs["model"], "vision-model")
        self.assertEqual(
            create.call_args.kwargs["response_format"], {"type": "json_object"}
        )

    def test_scoring_and_custom_required_fields(self):
        self.assertEqual(answer_scores("  INV-1 ", ["inv-1"])["exact_match"], 1)
        self.assertEqual(answer_scores("abcd", ["abce"])["anls"], 0.75)
        self.assertEqual(answer_scores("wrong", ["100"])["anls"], 0)
        results = detect_missing_fields(
            {"fields": {"name": {"value": "Ada"}}},
            "Other / Uncertain",
            ["name", "signature_status"],
        )
        self.assertEqual([r["status"] for r in results], ["Found", "Not found"])


class CategoryPreservationTests(unittest.TestCase):
    @patch("documind.ai.create_completion")
    def test_custom_type_is_preserved_when_model_echoes_invoice(self, create):
        create.return_value = N(
            choices=[
                N(
                    message=N(
                        content='{"category":"Invoice","summary":"A purchase order","fields":{"order_number":{"value":"PO-1"}}}'
                    )
                )
            ]
        )
        with patch(
            "documind.ai.get_settings",
            return_value=N(groq_api_key="test", groq_model="text"),
        ):
            result = extract_structured_data("Order number PO-1", "Purchase Order")
        self.assertEqual(result.category, "Purchase Order")
        self.assertEqual(result.fields["order_number"].value, "PO-1")
        prompt = create.call_args.kwargs["messages"][0]["content"]
        self.assertIn('"category": "Purchase Order"', prompt)
        self.assertIn("Do not classify again", prompt)
