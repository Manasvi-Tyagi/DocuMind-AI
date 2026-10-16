import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from scripts.download_docvqa import download_split, iter_samples, validate


def shard(split, question_id, source_id):
    image = io.BytesIO()
    Image.new("RGB", (20, 20), "white").save(image, format="PNG")
    annotation = {
        "questionId": question_id,
        "image": "documents/doc.png",
        "docId": question_id,
        "ucsf_document_id": source_id,
        "ucsf_document_page_no": "7",
        "question": "Total?",
        "answers": ["100"],
        "data_split": split,
        "ocr_results": {"status": "Succeeded"},
    }
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w") as archive:
        for name, data in [
            (f"{question_id}.json", json.dumps(annotation).encode()),
            (f"{question_id}.png", image.getvalue()),
        ]:
            member = tarfile.TarInfo(name)
            member.size = len(data)
            archive.addfile(member, io.BytesIO(data))
    return stream.getvalue()


class DatasetDownloadTests(unittest.TestCase):
    def test_pairs_and_preserves_metadata(self):
        rows = list(iter_samples(io.BytesIO(shard("train", 1, "source-a"))))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][1]["annotation"]["ucsf_document_page_no"], "7")
        self.assertTrue(rows[0][1]["image"])

    def test_download_validation_and_resume(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            for split, label, ident, origin in [
                ("train", "train", 1, "a"),
                ("validation", "val", 2, "b"),
            ]:
                name = f"docvqa-{split}-000.tar"
                with patch(
                    "scripts.download_docvqa.urllib.request.urlopen",
                    return_value=io.BytesIO(shard(label, ident, origin)),
                ):
                    records, _ = download_split(
                        output,
                        split,
                        1,
                        [name],
                        "pinned",
                        13,
                        {"a"} if split == "validation" else set(),
                    )
                self.assertEqual(records[0]["source_page"], "7")
                self.assertEqual(records[0]["page"], 1)
            summary = validate(output)
            self.assertEqual(summary["source_document_overlap"], 0)
            self.assertEqual(summary["validation"]["questions"], 1)
            with patch("scripts.download_docvqa.urllib.request.urlopen") as network:
                download_split(
                    output, "train", 1, ["docvqa-train-000.tar"], "pinned", 13, set()
                )
                network.assert_not_called()
            image = next((output / "train" / "images").iterdir())
            image.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                validate(output)
