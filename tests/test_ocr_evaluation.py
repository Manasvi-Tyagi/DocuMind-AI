import io
import json
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace as N
import unittest
from unittest.mock import patch

from documind.evaluation import edit_distance
from documind.ocr_evaluation import (
    transcript_scores,
    dataset_ocr_text,
    ocr_only_answer,
    reference_answer_present,
)
from scripts.evaluate_ocr import evaluate_ocr


def reference_distance(a, b):
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(
                min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (x != y))
            )
        previous = current
    return previous[-1]


class OCRTests(unittest.TestCase):
    def test_fast_distance_matches_dynamic_programming(self):
        generator = random.Random(13)
        for _ in range(300):
            a = "".join(generator.choices("abcd", k=generator.randrange(40)))
            b = "".join(generator.choices("abcd", k=generator.randrange(40)))
            self.assertEqual(edit_distance(a, b), reference_distance(a, b))
            self.assertEqual(edit_distance(list(a), list(b)), reference_distance(a, b))

    def test_transcript_metrics(self):
        metrics = transcript_scores("cat sat", "cat hat")
        self.assertAlmostEqual(metrics["cer"], 1 / 7)
        self.assertEqual(metrics["wer"], 0.5)
        self.assertGreater(transcript_scores("a b c d", "a")["wer"], 1)
        with self.assertRaises(ValueError):
            transcript_scores("anything", "")

    def test_baseline_has_no_reference_answer_input(self):
        self.assertEqual(
            ocr_only_answer("Invoice Number: INV-1\nTotal: 100", "What is the total?"),
            "100",
        )
        self.assertEqual(ocr_only_answer("Total: 100", "Who signed the document?"), "")
        self.assertFalse(reference_answer_present("Total: 100", ["10"]))
        self.assertTrue(reference_answer_present("Total: 100", ["100"]))

    @patch("scripts.evaluate_ocr.prepare_document")
    def test_machine_reference_is_labeled_and_pages_deduplicated(self, prepare):
        prepare.return_value = [
            N(text="Total: 100", method="Tesseract OCR", warning="")
        ]
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "image.png").write_bytes(b"mocked image")
            (root / "annotation.json").write_text(
                json.dumps(
                    {
                        "ocr_results": {
                            "recognitionResults": [{"lines": [{"text": "Total: 100"}]}]
                        }
                    }
                )
            )
            row = {
                "document": "image.png",
                "annotation": "annotation.json",
                "question": "What is the total?",
                "answers": ["100"],
            }
            (root / "questions.jsonl").write_text(
                json.dumps(row) + "\n" + json.dumps(row) + "\n"
            )
            with patch("sys.stdout", new_callable=io.StringIO):
                result = evaluate_ocr(root / "questions.jsonl", 2, root / "report.json")
            self.assertEqual(result["ocr_only_exact_match"], 1)
            self.assertEqual(result["machine_transcript_comparison"]["pages"], 1)
            self.assertEqual(result["human_verified_transcript_comparison"]["pages"], 0)
            self.assertEqual(prepare.call_count, 1)
