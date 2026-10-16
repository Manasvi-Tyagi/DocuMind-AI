"""Build a reproducible subset from downloaded DocVQA question annotations and images.

python -m scripts.prepare_subset path/to/val_v1.0.json path/to/images --limit 2000
"""

import argparse
import hashlib
import json
from pathlib import Path
import random
import shutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("annotations", type=Path)
    parser.add_argument("images", type=Path)
    parser.add_argument("--output", type=Path, default=Path("data/docvqa-subset"))
    parser.add_argument("--limit", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be positive")
    payload = json.loads(args.annotations.read_text())
    rows = payload["data"] if isinstance(payload, dict) else payload
    labeled = [r for r in rows if r.get("answers")]
    random.Random(args.seed).shuffle(labeled)
    selected = labeled[: args.limit]
    if not selected:
        raise ValueError(
            "No labeled questions found. Use train/validation annotations, not an unlabeled test split."
        )
    image_root = args.images.resolve()
    # Validate every source before copying, including common documents/ prefixes.
    sources = []
    for row in selected:
        image_name = row["image"]
        candidates = [
            (image_root / image_name).resolve(),
            (image_root / Path(image_name).name).resolve(),
        ]
        path = next(
            (p for p in candidates if p.is_relative_to(image_root) and p.is_file()),
            None,
        )
        if path is None:
            raise FileNotFoundError(f"Cannot find {image_name} below {image_root}")
        sources.append(path)
    output = args.output
    (output / "images").mkdir(parents=True, exist_ok=True)
    records = []
    for row, path in zip(selected, sources):
        name = (
            hashlib.sha256(str(path.relative_to(image_root)).encode()).hexdigest()[:16]
            + path.suffix
        )
        destination = output / "images" / name
        if not destination.exists():
            shutil.copyfile(path, destination)
        records.append(
            {
                "id": row.get("questionId"),
                "document_id": row.get("docId", path.stem),
                "document": f"images/{name}",
                "page": 1,
                "question": row["question"],
                "answers": row["answers"],
            }
        )
    (output / "questions.jsonl").write_text(
        "\n".join(json.dumps(r) for r in records) + "\n"
    )
    print(
        f"Prepared {len(records)} labeled questions at {output / 'questions.jsonl'} (seed={args.seed})."
    )


if __name__ == "__main__":
    main()
