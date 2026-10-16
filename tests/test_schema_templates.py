import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
from pydantic import ValidationError
from documind.models import StructuredDocument
from documind import categories
from documind.validation import detect_missing_fields


class SchemaTemplateTests(unittest.TestCase):
    def test_fields_validate_metadata_and_keep_nested_values(self):
        doc = StructuredDocument.model_validate(
            {
                "category": "Invoice",
                "summary": "Test",
                "fields": {
                    "items": {
                        "value": [{"amount": 100}],
                        "confidence": "high",
                        "page": 1,
                    },
                    "legacy": 12,
                },
            }
        )
        self.assertEqual(doc.fields["items"].confidence, "HIGH")
        self.assertEqual(doc.fields["legacy"].value, 12)
        for metadata in [{"confidence": "CERTAIN"}, {"page": 0}, {"evidence": []}]:
            with self.assertRaises(ValidationError):
                StructuredDocument.model_validate(
                    {
                        "category": "Invoice",
                        "summary": "Test",
                        "fields": {"total": {"value": 100, **metadata}},
                    }
                )

    def test_templates_persist_and_drive_missing_field_rules(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(
            categories, "DATABASE", Path(folder) / "categories.sqlite3"
        ):
            categories.save_required_template(
                "Lease Agreement", ["Tenant Name", "signature_status", "Tenant Name"]
            )
            self.assertEqual(
                categories.get_required_template("lease agreement"),
                ["tenant_name", "signature_status"],
            )
            results = detect_missing_fields(
                {"fields": {"tenant_name": {"value": "Ada"}}}, "Lease Agreement"
            )
            self.assertEqual([r["status"] for r in results], ["Found", "Not found"])
            with self.assertRaises(ValueError):
                categories.save_required_template("Lease Agreement", ["bad/name"])
