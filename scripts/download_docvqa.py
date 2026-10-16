"""Stream a pinned, reproducible DocVQA subset without downloading all shards.

The sample uses seeded shard order and source order within each shard; it is not
an independent uniform random sample of the full corpus. No Groq calls are made.
"""

import argparse
import hashlib
import io
import json
from pathlib import Path
import random
import tarfile
import time
import urllib.parse
import urllib.request

from PIL import Image

REPOSITORY = "pixparse/docvqa-wds"
BASE = "https://huggingface.co"


def read_json(url):
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.load(response)


def iter_samples(response):
    """Read adjacent WebDataset JSON/image pairs without extracting tar paths."""
    pending = {}
    with tarfile.open(fileobj=response, mode="r|") as archive:
        for member in archive:
            if not member.isfile():
                continue
            path = Path(member.name)
            suffix = path.suffix.lower()
            if suffix not in {".json", ".png", ".jpg", ".jpeg"}:
                continue
            if member.size > 30 * 1024 * 1024:
                raise ValueError(f"Unexpectedly large member: {member.name}")
            entry = pending.setdefault(path.stem, {})
            handle = archive.extractfile(member)
            if suffix == ".json":
                entry["annotation"] = json.load(handle)
            else:
                entry["image"] = handle.read()
                entry["suffix"] = suffix
            if "annotation" in entry and "image" in entry:
                yield path.stem, pending.pop(path.stem)
            if len(pending) > 100:
                raise ValueError("Too many unpaired image/annotation members.")


def source_id(annotation):
    return str(
        annotation.get("ucsf_document_id")
        or annotation.get("docId")
        or annotation["image"]
    )


def download_split(output, split, target, shards, revision, seed, excluded_sources):
    folder = output / split
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "images").mkdir(exist_ok=True)
    (folder / "annotations").mkdir(exist_ok=True)
    manifest = folder / "questions.jsonl"
    records = (
        [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
        if manifest.exists()
        else []
    )
    seen = {str(r["id"]) for r in records}
    sources = {r["source_document_id"] for r in records}
    if sources & excluded_sources:
        raise ValueError(
            "Existing subset overlaps source documents across splits; use a fresh output directory."
        )
    if len(records) > target:
        raise ValueError(
            "Existing split exceeds target; use a new output directory for a smaller subset."
        )
    ordered = sorted(
        s for s in shards if s.startswith(f"docvqa-{split}-") and s.endswith(".tar")
    )
    random.Random(seed).shuffle(ordered)
    used_shards = set(r["shard"] for r in records)
    with manifest.open("a") as sink:
        for shard in ordered:
            if len(records) >= target:
                break
            print(f"{split}: streaming {shard} ({len(records)}/{target})", flush=True)
            url = f"{BASE}/datasets/{REPOSITORY}/resolve/{revision}/{urllib.parse.quote(shard)}"
            for attempt in range(3):
                try:
                    with urllib.request.urlopen(url, timeout=60) as response:
                        for key, sample in iter_samples(response):
                            annotation = sample["annotation"]
                            question_id = str(annotation["questionId"])
                            answers = annotation.get("answers")
                            origin = source_id(annotation)
                            if (
                                question_id in seen
                                or origin in excluded_sources
                                or not answers
                            ):
                                continue
                            if not isinstance(answers, list) or not all(
                                isinstance(a, str) for a in answers
                            ):
                                raise ValueError("Unexpected answer format")
                            expected_split = "val" if split == "validation" else "train"
                            if annotation.get("data_split") != expected_split:
                                raise ValueError("Shard annotation split mismatch")
                            image = sample["image"]
                            with Image.open(io.BytesIO(image)) as check:
                                check.verify()
                            digest = hashlib.sha256(image).hexdigest()
                            image_name = digest + sample["suffix"]
                            (folder / "images" / image_name).write_bytes(image)
                            safe_key = hashlib.sha256(question_id.encode()).hexdigest()[
                                :20
                            ]
                            annotation_name = safe_key + ".json"
                            (folder / "annotations" / annotation_name).write_text(
                                json.dumps(annotation)
                            )
                            record = {
                                "id": annotation["questionId"],
                                "document_id": annotation.get("docId"),
                                "source_document_id": origin,
                                "source_page": annotation.get("ucsf_document_page_no"),
                                "document": f"images/{image_name}",
                                "annotation": f"annotations/{annotation_name}",
                                "page": 1,
                                "question": annotation["question"],
                                "answers": answers,
                                "split": split,
                                "shard": shard,
                                "image_sha256": digest,
                            }
                            sink.write(json.dumps(record) + "\n")
                            sink.flush()
                            records.append(record)
                            seen.add(question_id)
                            sources.add(origin)
                            used_shards.add(shard)
                            if len(records) % 100 == 0:
                                print(f"{split}: {len(records)}/{target}", flush=True)
                            if len(records) >= target:
                                break
                    break
                except (OSError, EOFError, tarfile.ReadError) as error:
                    if attempt == 2:
                        raise
                    print(
                        f"Retry {attempt + 1} for {shard}: {type(error).__name__}",
                        flush=True,
                    )
                    time.sleep(1)
    if len(records) != target:
        raise ValueError(
            f"Only found {len(records)} eligible {split} questions; expected {target}."
        )
    return records, sorted(used_shards)


def validate(output):
    summary = {}
    split_sources = []
    split_ids = []
    split_images = []
    for split in ("train", "validation"):
        folder = output / split
        records = [
            json.loads(line)
            for line in (folder / "questions.jsonl").read_text().splitlines()
        ]
        ids = [str(r["id"]) for r in records]
        if len(set(ids)) != len(ids):
            raise ValueError(f"Duplicate question IDs in {split}")
        for row in records:
            image = (folder / row["document"]).read_bytes()
            if hashlib.sha256(image).hexdigest() != row["image_sha256"]:
                raise ValueError("Image hash mismatch")
            if not row["answers"] or not row["question"].strip():
                raise ValueError("Unlabeled/empty question")
            annotation = json.loads((folder / row["annotation"]).read_text())
            if (
                annotation["questionId"] != row["id"]
                or annotation["answers"] != row["answers"]
                or annotation["question"] != row["question"]
                or source_id(annotation) != row["source_document_id"]
            ):
                raise ValueError("Image/annotation manifest mismatch")
        summary[split] = {
            "questions": len(records),
            "unique_images": len({r["image_sha256"] for r in records}),
            "source_documents": len({r["source_document_id"] for r in records}),
        }
        split_sources.append({r["source_document_id"] for r in records})
        split_ids.append(set(ids))
        split_images.append({r["image_sha256"] for r in records})
    if split_sources[0] & split_sources[1] or split_ids[0] & split_ids[1]:
        raise ValueError("Train/validation source or question overlap")
    summary["source_document_overlap"] = 0
    summary["image_hash_overlap"] = len(split_images[0] & split_images[1])
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/docvqa"))
    parser.add_argument("--train", type=int, default=1600)
    parser.add_argument("--validation", type=int, default=400)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.validate_only:
        print(json.dumps(validate(args.output), indent=2))
        return
    if min(args.train, args.validation) < 1:
        parser.error("Both split sizes must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    provenance_file = args.output / "provenance.json"
    if provenance_file.exists():
        provenance = json.loads(provenance_file.read_text())
        if provenance["seed"] != args.seed:
            raise ValueError(
                "Existing subset uses another seed; choose a new output folder."
            )
    else:
        info = read_json(f"{BASE}/api/datasets/{REPOSITORY}")
        provenance = {
            "repository": REPOSITORY,
            "revision": info["sha"],
            "seed": args.seed,
            "sampling": "seeded shard order, source order within shards; not globally uniform",
            "license_note": "Mirror lists MIT; review original image conditions before redistribution.",
        }
        provenance_file.write_text(json.dumps(provenance, indent=2))
    revision = provenance["revision"]
    tree = read_json(f"{BASE}/api/datasets/{REPOSITORY}/tree/{revision}?limit=1000")
    shards = [entry["path"] for entry in tree if entry["type"] == "file"]
    train, train_shards = download_split(
        args.output, "train", args.train, shards, revision, args.seed, set()
    )
    _, validation_shards = download_split(
        args.output,
        "validation",
        args.validation,
        shards,
        revision,
        args.seed,
        {r["source_document_id"] for r in train},
    )
    provenance.update(
        shards={"train": train_shards, "validation": validation_shards},
        summary=validate(args.output),
    )
    provenance_file.write_text(json.dumps(provenance, indent=2))
    print(json.dumps(provenance["summary"], indent=2))


if __name__ == "__main__":
    main()
