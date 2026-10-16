"""Evaluate local Tesseract and heuristic OCR-only QA without API calls.

Optional manifest fields: transcript (relative .txt path), transcript_kind
('human_verified' or 'machine'). Otherwise the supplied DocVQA OCR is used as
a machine-reference agreement comparison, not ground-truth OCR accuracy.
"""

import argparse
import io
import json
from pathlib import Path
from time import perf_counter

from documind.documents import prepare_document
from documind.evaluation import answer_scores
from documind.ocr_evaluation import (
    dataset_ocr_text,
    ocr_only_answer,
    reference_answer_present,
    transcript_scores,
)


def evaluate_ocr(manifest, limit, output):
    records, cache = [], {}
    with manifest.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            if len(records) >= limit:
                break
            row = json.loads(line)
            if not isinstance(row.get("answers"), list) or not row["answers"]:
                raise ValueError(
                    "A labeled manifest with nonempty answer lists is required."
                )
            record = {
                "id": row.get("id"),
                "question": row["question"],
                "answers": row["answers"],
                "document": row["document"],
                "page": int(row.get("page", 1)),
            }
            try:
                path = (manifest.parent / row["document"]).resolve()
                key = (str(path), int(row.get("page", 1)))
                if key not in cache:
                    upload = io.BytesIO(path.read_bytes())
                    upload.name = path.name
                    start = perf_counter()
                    page = prepare_document(upload, [key[1]])[0]
                    cache[key] = (
                        page.text,
                        page.method,
                        page.warning,
                        perf_counter() - start,
                    )
                text, method, warning, seconds = cache[key]
                if method != "Tesseract OCR":
                    raise ValueError(
                        "OCR evaluation requires an image or scanned PDF page that uses Tesseract, not native PDF text."
                    )
                if warning:
                    raise ValueError(warning)
                predicted = ocr_only_answer(text, row["question"])
                record.update(
                    text=text,
                    extraction_method=method,
                    ocr_seconds=seconds,
                    ocr_only={
                        "answer": predicted,
                        **answer_scores(predicted, row["answers"]),
                    },
                    reference_answer_present=reference_answer_present(
                        text, row["answers"]
                    ),
                )
                kind, reference = None, ""
                if row.get("transcript"):
                    kind = row.get("transcript_kind")
                    if kind not in {
                        "human_verified",
                        "synthetic_ground_truth",
                        "machine",
                    }:
                        raise ValueError(
                            "transcript_kind must be human_verified, synthetic_ground_truth, or machine."
                        )
                    reference = (manifest.parent / row["transcript"]).read_text()
                elif row.get("annotation"):
                    annotation = json.loads(
                        (manifest.parent / row["annotation"]).read_text()
                    )
                    reference = dataset_ocr_text(annotation)
                    kind = "machine"
                if reference.strip():
                    record["transcript_comparison"] = {
                        "reference_kind": kind,
                        **transcript_scores(text, reference),
                    }
                else:
                    record["transcript_comparison"] = {
                        "unavailable": "No reference transcript; CER/WER cannot be measured."
                    }
            except Exception as error:
                record["error"] = f"{type(error).__name__}: {error}"
            records.append(record)
            print(f"OCR completed {len(records)}/{limit}", flush=True)
    if not records:
        raise ValueError("No examples found.")
    valid = [r for r in records if "error" not in r]
    summary = {
        "questions": len(records),
        "succeeded": len(valid),
        "failed": len(records) - len(valid),
        "unique_pages_processed": len(cache),
        "normalization": "lowercase and collapse whitespace; punctuation retained",
        "qa_baseline": "question-keyword line selection; label/value split; no LLM and no gold-answer selection",
    }
    for metric in ("exact_match", "anls"):
        summary[f"ocr_only_{metric}"] = (
            sum(r["ocr_only"][metric] for r in valid) / len(valid) if valid else None
        )
    summary["reference_answer_presence_rate"] = (
        sum(r["reference_answer_present"] for r in valid) / len(valid)
        if valid
        else None
    )
    for kind in ("human_verified", "synthetic_ground_truth", "machine"):
        # One transcript comparison per document page, even with multiple questions.
        comparisons = {}
        for record in valid:
            comparison = record.get("transcript_comparison", {})
            if comparison.get("reference_kind") == kind:
                comparisons[(record["document"], record["page"])] = comparison
        values = list(comparisons.values())
        chars = sum(v["reference_characters"] for v in values)
        words = sum(v["reference_words"] for v in values)
        summary[kind + "_transcript_comparison"] = {
            "pages": len(values),
            "micro_cer": (
                sum(v["character_errors"] for v in values) / chars if chars else None
            ),
            "micro_wer": (
                sum(v["word_errors"] for v in values) / words if words else None
            ),
            "interpretation": (
                "OCR accuracy against labeled human transcripts"
                if kind == "human_verified"
                else (
                    "Accuracy against known rendered synthetic text; not real-world document accuracy"
                    if kind == "synthetic_ground_truth"
                    else "Agreement with supplied machine OCR; not ground-truth accuracy"
                )
            ),
        }
    summary["unique_page_ocr_seconds"] = sum(value[3] for value in cache.values())
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"summary": summary, "results": records}, indent=2))
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument(
        "--output", type=Path, default=Path("reports/ocr-validation.json")
    )
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be positive")
    evaluate_ocr(args.manifest, args.limit, args.output)
