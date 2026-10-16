import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.evaluate import evaluate


class EvaluationTests(unittest.TestCase):
    @patch("scripts.evaluate.answer_question")
    @patch("scripts.evaluate.prepare_document")
    def test_reports_scores_and_failures_without_dropping_rows(self, prepare, answer):
        from types import SimpleNamespace

        prepare.return_value = [
            SimpleNamespace(number=1, text="Total 220", method="PDF text")
        ]
        answer.side_effect = ["220", RuntimeError("test failure")]
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "doc.pdf").write_bytes(b"test document")
            manifest = root / "questions.jsonl"
            manifest.write_text(
                json.dumps(
                    {"document": "doc.pdf", "question": "Total?", "answers": ["220"]}
                )
                + "\n"
            )
            output = root / "results.json"
            with patch("sys.stdout", new_callable=io.StringIO):
                evaluate(manifest, 1, output)
            result = json.loads(output.read_text())
        self.assertEqual(result["summary"]["text"]["exact_match"], 1)
        self.assertEqual(result["summary"]["vision"]["failed"], 1)
        self.assertIsNone(result["summary"]["vision"]["anls"])
        self.assertIn("error", result["results"][0]["vision"])

    @patch("scripts.evaluate.answer_question", return_value="220")
    @patch("scripts.evaluate.prepare_document")
    def test_vision_rerun_reuses_text_without_repeating_api_call(self, prepare, answer):
        from types import SimpleNamespace

        prepare.return_value = [
            SimpleNamespace(number=1, text="Total 220", method="Tesseract OCR")
        ]
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "doc.png").write_bytes(b"mock image")
            row = {
                "id": 1,
                "document": "doc.png",
                "question": "Total?",
                "answers": ["220"],
            }
            manifest = root / "questions.jsonl"
            manifest.write_text(json.dumps(row) + "\n")
            previous = root / "previous.json"
            previous.write_text(
                json.dumps(
                    {
                        "results": [
                            {
                                "id": 1,
                                "document_id": "doc.png",
                                "question": "Total?",
                                "answers": ["220"],
                                "text": {
                                    "answer": "220",
                                    "exact_match": 1.0,
                                    "anls": 1.0,
                                    "seconds": 0.5,
                                },
                            }
                        ]
                    }
                )
            )
            with patch("sys.stdout", new_callable=io.StringIO):
                evaluate(
                    manifest, 1, root / "output.json", modes=["vision"], reuse=previous
                )
            result = json.loads((root / "output.json").read_text())
        self.assertEqual(answer.call_count, 1)
        self.assertIsNotNone(answer.call_args.kwargs["pages"])
        self.assertEqual(result["results"][0]["text"]["seconds"], 0.5)
        self.assertEqual(result["summary"]["vision"]["succeeded"], 1)

    @patch("scripts.evaluate.answer_question")
    @patch("scripts.evaluate.prepare_document")
    def test_rate_limit_stops_later_requests_for_that_mode(self, prepare, answer):
        import httpx
        from groq import RateLimitError
        from types import SimpleNamespace

        prepare.return_value = [
            SimpleNamespace(number=1, text="Total 220", method="Tesseract OCR")
        ]
        response = httpx.Response(
            429, request=httpx.Request("POST", "https://api.groq.com")
        )
        answer.side_effect = RateLimitError(
            "Quota exhausted", response=response, body={}
        )
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "doc.png").write_bytes(b"mock")
            rows = [
                {
                    "id": n,
                    "document": "doc.png",
                    "question": "Total?",
                    "answers": ["220"],
                }
                for n in [1, 2]
            ]
            manifest = root / "questions.jsonl"
            manifest.write_text("\n".join(json.dumps(row) for row in rows))
            with patch("sys.stdout", new_callable=io.StringIO):
                evaluate(manifest, 2, root / "output.json", modes=["vision"])
            result = json.loads((root / "output.json").read_text())
        self.assertEqual(answer.call_count, 1)
        self.assertEqual(result["summary"]["vision"]["failed"], 1)
        self.assertEqual(result["summary"]["vision"]["not_run"], 1)
