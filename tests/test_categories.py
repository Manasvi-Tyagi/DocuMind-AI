import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as N
import unittest
from unittest.mock import patch

from documind import categories, ai


class CategoryTests(unittest.TestCase):
    def test_custom_categories_persist_deduplicate_and_classify(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(
            categories, "DATABASE", Path(folder) / "categories.sqlite3"
        ):
            self.assertEqual(
                categories.add_category("  Purchase   Order "), "Purchase Order"
            )
            self.assertEqual(
                categories.add_category("purchase order"), "Purchase Order"
            )
            self.assertEqual(categories.get_categories().count("Purchase Order"), 1)
            with patch.object(ai, "create_completion") as create, patch.object(
                ai,
                "get_settings",
                return_value=N(groq_api_key="test", groq_model="test"),
            ):
                create.return_value = N(
                    choices=[N(message=N(content="Purchase Order"))]
                )
                self.assertEqual(
                    ai.classify_document("Purchase order number PO-1"), "Purchase Order"
                )
                prompt = create.call_args.kwargs["messages"][0]["content"]
                self.assertIn("Purchase Order", prompt)
                create.return_value = N(
                    choices=[N(message=N(content="Unregistered type"))]
                )
                self.assertEqual(ai.classify_document("Unknown"), "Other / Uncertain")
            with self.assertRaises(ValueError):
                categories.add_category("   ")
            with self.assertRaises(ValueError):
                categories.add_category("x" * 81)

    def test_new_type_is_saved_and_reused_on_next_document(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(
            categories, "DATABASE", Path(folder) / "categories.sqlite3"
        ):

            def response(content):
                return N(choices=[N(message=N(content=content))])

            proposal = {
                "category": "Purchase Order",
                "description": "Purchase order number, supplier and delivery details",
                "confidence": "HIGH",
                "evidence": "Purchase Order No: PO-123",
            }
            with patch.object(
                ai,
                "get_settings",
                return_value=N(groq_api_key="test", groq_model="test"),
            ), patch.object(ai, "create_completion") as create:
                create.side_effect = [
                    response("Other / Uncertain"),
                    response(json.dumps(proposal)),
                ]
                self.assertEqual(
                    ai.classify_document(
                        "Purchase Order No: PO-123\nSupplier: Example"
                    ),
                    "Purchase Order",
                )
                self.assertIn("Purchase Order", categories.get_categories())
                create.side_effect = [response("Purchase Order")]
                self.assertEqual(
                    ai.classify_document("Purchase Order No: PO-456"), "Purchase Order"
                )
                prompt = create.call_args.kwargs["messages"][0]["content"]
                self.assertIn(proposal["description"], prompt)
            self.assertEqual(
                len(
                    [
                        n
                        for n in categories.get_categories()
                        if n.casefold() == "purchase order"
                    ]
                ),
                1,
            )

    def test_unsupported_new_type_is_not_added(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(
            categories, "DATABASE", Path(folder) / "categories.sqlite3"
        ):
            for confidence, quote in [
                ("LOW", "Purchase Order"),
                ("HIGH", "invented quote"),
            ]:
                proposal = {
                    "category": "Purchase Order",
                    "description": "Supplier order",
                    "confidence": confidence,
                    "evidence": quote,
                }
                with patch.object(
                    ai,
                    "get_settings",
                    return_value=N(groq_api_key="test", groq_model="test"),
                ), patch.object(
                    ai,
                    "create_completion",
                    return_value=N(
                        choices=[N(message=N(content=json.dumps(proposal)))]
                    ),
                ):
                    self.assertEqual(
                        ai.discover_category("Purchase Order"), "Other / Uncertain"
                    )
                self.assertNotIn("Purchase Order", categories.get_categories())
