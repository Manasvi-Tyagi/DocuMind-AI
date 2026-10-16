import io
import unittest
from PIL import Image
from documind.documents import DocumentPage
from documind.evidence import locate_quote, highlight_matches


class EvidenceTests(unittest.TestCase):
    def page(self):
        image = io.BytesIO()
        Image.new("RGB", (200, 200), "white").save(image, format="JPEG")
        return DocumentPage(
            2,
            image.getvalue(),
            "Total: 220",
            "PDF text",
            word_boxes=[
                {"text": "Total:", "box": [0.1, 0.1, 0.3, 0.2]},
                {"text": "220", "box": [0.4, 0.1, 0.6, 0.2]},
            ],
        )

    def test_literal_quote_has_real_source_box(self):
        page = self.page()
        result = locate_quote([page], "Total: 220", 2)
        self.assertEqual(result["status"], "located")
        self.assertEqual(result["matches"][0]["box"], [0.1, 0.1, 0.6, 0.2])
        self.assertEqual(highlight_matches(page, result["matches"]).size, (200, 200))

    def test_invented_quote_or_page_is_unverified(self):
        self.assertEqual(
            locate_quote([self.page()], "Total: 999", 2)["status"], "unverified"
        )
        self.assertEqual(locate_quote([self.page()], "Total: 220", 7)["matches"], [])

    def test_native_pdf_words_supply_boxes_without_ocr(self):
        import pymupdf
        from documind.documents import prepare_document

        with pymupdf.open() as pdf:
            page = pdf.new_page()
            page.insert_text((72, 72), "Total: 220")
            upload = io.BytesIO(pdf.tobytes())
        upload.name = "invoice.pdf"
        pages = prepare_document(upload, [1])
        result = locate_quote(pages, "Total: 220", 1)
        self.assertEqual(result["status"], "located")
        self.assertTrue(all(0 <= v <= 1 for v in result["matches"][0]["box"]))
