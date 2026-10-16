from types import SimpleNamespace as N
import unittest
from unittest.mock import patch
from documind.batching import page_batches, retrieve_pages, merge_documents
from documind.models import StructuredDocument
from documind.semantic import review_semantics
from documind.validation import detect_missing_fields


class BatchSemanticTests(unittest.TestCase):
    def test_retrieval_prioritizes_explicit_pages_and_matching_terms(self):
        pages = [
            N(number=i, text="Invoice tax 100" if i == 4 else "Application name Ada")
            for i in range(1, 8)
        ]
        self.assertEqual([len(b) for b in page_batches(pages)], [3, 3, 1])
        self.assertIn(
            4, [p.number for p in retrieve_pages(pages, "What is the invoice tax?")]
        )
        self.assertIn(7, [p.number for p in retrieve_pages(pages, "Summarize page 7")])

    def test_merge_keeps_conflicting_fields_and_batch_scopes(self):
        docs = [
            StructuredDocument(
                category="Invoice", summary="First", fields={"total": {"value": 100}}
            ),
            StructuredDocument(
                category="Invoice", summary="Second", fields={"total": {"value": 200}}
            ),
        ]
        merged = merge_documents(docs, [[N(number=1)], [N(number=4)]], "Invoice")
        self.assertEqual(merged.fields["total"].value, 100)
        self.assertEqual(merged.fields["total__pages_4"].value, 200)
        self.assertEqual(len(merged.batches), 2)

    @patch("documind.ai.create_completion")
    def test_semantic_flags_keep_unmatched_quotes_unverified(self, create):
        create.return_value = N(
            choices=[
                N(
                    message=N(
                        content='{"issues":[{"description":"Status conflict","first_quote":"Approved","second_quote":"Rejected"}]}'
                    )
                )
            ]
        )
        result = review_semantics("The application is Approved.")
        self.assertFalse(result[0]["quotes_present_in_text"])
        self.assertEqual(result[0]["status"], "Needs human review")

    def test_unsigned_signature_counts_as_missing(self):
        rows = detect_missing_fields(
            {"fields": {"signature_status": {"value": "Unsigned"}}},
            "Scanned Form",
            ["signature_status"],
        )
        self.assertEqual(rows[0]["status"], "Not found")


class TextBatchTests(unittest.TestCase):
    @patch("documind.analysis.classify_document", return_value="Report")
    @patch("documind.analysis.extract_structured_data")
    def test_text_only_long_documents_are_batched(self, extract, classify):
        from documind.analysis import classify_pages, extract_pages

        pages = [N(number=i, text=f"Page {i}") for i in range(1, 5)]
        extract.return_value = StructuredDocument(
            category="Report", summary="Batch", fields={}
        )
        self.assertEqual(classify_pages("all", pages, vision=False), "Report")
        self.assertEqual(classify.call_count, 2)
        result = extract_pages("all", "Report", pages, vision=False)
        self.assertEqual(len(result.batches), 2)
        self.assertEqual(extract.call_count, 2)


class DateAmbiguityTests(unittest.TestCase):
    def test_ambiguous_numeric_dates_are_not_assumed(self):
        from documind.validation import _parse_date

        self.assertIsNone(_parse_date("01/02/2026"))
        self.assertEqual(str(_parse_date("2026-02-01")), "2026-02-01")
        self.assertEqual(str(_parse_date("13/02/2026")), "2026-02-13")
