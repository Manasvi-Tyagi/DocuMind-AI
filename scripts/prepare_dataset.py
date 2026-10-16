"""Step 1 — Download a manageable DocVQA subset from pixparse/docvqa-wds.

Each WebDataset sample is one question: ``<key>.json`` + ``<key>.png``.
For every question we keep: document image, question, answers, document ID
and page information. The reference OCR shipped with DocVQA is saved
separately so Tesseract accuracy can be measured (Step 10).

Usage:
    python scripts/prepare_dataset.py --split validation --num-questions 2000
"""

import argparse
import json
import sys
import tarfile
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO_ID = "pixparse/docvqa-wds"
SHARDS_PER_SPLIT = {"train": 384, "validation": 64, "test": 64}


def reference_ocr(sample: dict) -> dict:
    """Keep only the text lines (with boxes) of the reference OCR."""

    results = (sample.get("ocr_results") or {}).get("recognitionResults") or []
    pages = []
    for page in results:
        pages.append(
            {
                "page": page.get("page"),
                "width": page.get("width"),
                "height": page.get("height"),
                "lines": [
                    {"text": line.get("text", ""), "boundingBox": line.get("boundingBox")}
                    for line in page.get("lines", [])
                ],
            }
        )
    return {"pages": pages}


def process_shard(tar_path: Path, out_dir: Path, split: str, rows: list[dict], limit: int) -> None:
    images_dir = out_dir / "images"
    ocr_dir = out_dir / "ocr"

    with tarfile.open(tar_path) as tar:
        members = {m.name: m for m in tar.getmembers() if m.isfile()}
        keys = sorted({name.rsplit(".", 1)[0] for name in members})
        for key in keys:
            if len(rows) >= limit:
                return
            json_member = members.get(f"{key}.json")
            png_member = members.get(f"{key}.png")
            if json_member is None or png_member is None:
                continue

            sample = json.load(tar.extractfile(json_member))
            answers = sample.get("answers") or []
            if split != "test" and not answers:
                continue

            # Several questions share one page image; store each image once.
            image_name = Path(sample.get("image") or f"{key}.png").name
            image_path = images_dir / image_name
            if not image_path.exists():
                image_path.write_bytes(tar.extractfile(png_member).read())

            ocr_path = ocr_dir / (Path(image_name).stem + ".json")
            if not ocr_path.exists() and sample.get("ocr_results"):
                ocr_path.write_text(json.dumps(reference_ocr(sample)))

            rows.append(
                {
                    "question_id": sample.get("questionId"),
                    "question": sample.get("question"),
                    "answers": answers,
                    "question_types": sample.get("question_types", []),
                    "doc_id": sample.get("docId"),
                    "ucsf_document_id": sample.get("ucsf_document_id"),
                    "page_no": sample.get("ucsf_document_page_no"),
                    "image_path": str(image_path.relative_to(out_dir)),
                    "ocr_path": str(ocr_path.relative_to(out_dir)) if ocr_path.exists() else None,
                    "split": split,
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=SHARDS_PER_SPLIT, default="validation")
    parser.add_argument("--num-questions", type=int, default=2000, help="Recommended: 2000-5000")
    parser.add_argument("--out", type=Path, default=Path("data/docvqa"))
    parser.add_argument("--keep-shards", action="store_true", help="Keep downloaded .tar shards")
    args = parser.parse_args()

    if args.split == "test":
        print("Note: the DocVQA test split has no public answers; it cannot be scored.", file=sys.stderr)

    out_dir = args.out / args.split
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    (out_dir / "ocr").mkdir(parents=True, exist_ok=True)
    shard_dir = args.out / "_shards"

    rows: list[dict] = []
    for index in range(SHARDS_PER_SPLIT[args.split]):
        if len(rows) >= args.num_questions:
            break
        filename = f"docvqa-{args.split}-{index:03d}.tar"
        print(f"Downloading {filename} ... ({len(rows)}/{args.num_questions} questions so far)")
        tar_path = Path(hf_hub_download(REPO_ID, filename, repo_type="dataset", local_dir=shard_dir))
        process_shard(tar_path, out_dir, args.split, rows, args.num_questions)
        if not args.keep_shards:
            tar_path.unlink(missing_ok=True)

    metadata = out_dir / "metadata.jsonl"
    with metadata.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    n_images = len({r["image_path"] for r in rows})
    print(f"Saved {len(rows)} questions over {n_images} document images to {metadata}")
    print(
        "Licensing: the images originate from the UCSF Industry Documents Library. "
        "Review its access conditions before redistributing them."
    )


if __name__ == "__main__":
    main()
