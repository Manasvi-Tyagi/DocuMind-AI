"""Compare OCR-only, text + LLM, and vision + LLM using the selected backend on a JSONL subset.

Run from the project root: python -m scripts.evaluate data/validation.jsonl --limit 50
Each row: {"document": "images/doc.png", "question": "...", "answers": ["..."], "page": 1}
Document paths are relative to the manifest. Ollama is local; selecting Groq sends selected images/text to its API.
"""

import argparse
import io
import json
from pathlib import Path
from time import perf_counter
from groq import RateLimitError

from documind.ai import answer_question, extract_structured_data
from documind.documents import prepare_document, pages_text
from documind.evaluation import answer_scores
from documind.ocr_evaluation import ocr_only_answer
from documind.reporting import write_report
from documind.config import get_settings, model_name, use_provider


def evaluate(
    manifest, limit, output, modes=("text", "vision"), reuse=None, retry_failed=False
):
    records = []
    cache = {}
    settings = get_settings()
    prior = {}
    rate_limited_modes = set()
    if reuse:
        previous = json.loads(reuse.read_text())

        current = get_settings()
        old_models = previous.get("summary", {}).get("models", {})
        for mode in ("text", "vision"):
            if (
                (mode not in modes or retry_failed)
                and old_models.get(mode)
                and old_models[mode] != model_name(current, vision=mode == "vision")
            ):
                raise ValueError(
                    "Cannot reuse answers from a different model/backend. Use a new report without --reuse."
                )
        prior = {str(r["id"]): r for r in previous["results"]}

    with manifest.open() as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            if len(records) >= limit:
                break
            row = json.loads(line)
            if not isinstance(row.get("answers"), list) or not row["answers"]:
                raise ValueError(
                    f"Line {line_number} needs nonempty reference answers (use a labeled split)."
                )
            record = {
                "id": row.get("id", line_number),
                "document_id": row.get("document_id", row["document"]),
                "question": row["question"],
                "answers": row["answers"],
            }
            old = prior.get(str(record["id"]), {})
            if (
                old.get("question") == record["question"]
                and old.get("document_id") == record["document_id"]
                and old.get("answers") == record["answers"]
            ):
                for mode in ("text", "vision"):
                    if mode in old and (
                        mode not in modes
                        or (
                            retry_failed
                            and "error" not in old[mode]
                            and "not_run" not in old[mode]
                        )
                    ):
                        record[mode] = old[mode]
            try:
                path = (manifest.parent / row["document"]).resolve()
                page_number = int(row.get("page", 1))
                key = (str(path), page_number)
                if key not in cache:
                    upload = io.BytesIO(path.read_bytes())
                    upload.name = path.name
                    started = perf_counter()
                    pages = prepare_document(upload, [page_number])
                    cache[key] = (pages, perf_counter() - started)
                pages, preprocessing = cache[key]
                text = pages_text(pages)
                record.update(
                    page=page_number,
                    preprocessing_seconds=preprocessing,
                    extraction_method=pages[0].method,
                )
                baseline_started = perf_counter()
                baseline = ocr_only_answer(pages[0].text, row["question"])
                record["ocr_only"] = {
                    "answer": baseline,
                    **answer_scores(baseline, row["answers"]),
                    "seconds": perf_counter() - baseline_started,
                }
                for mode in modes:
                    if (
                        retry_failed
                        and mode in record
                        and "error" not in record[mode]
                        and "not_run" not in record[mode]
                    ):
                        continue
                    if mode in rate_limited_modes:
                        record[mode] = {
                            "not_run": "An earlier request hit the Groq rate limit. Resume after quota is available."
                        }
                        continue
                    started = perf_counter()
                    try:
                        if mode == "text" and not pages[0].text.strip():
                            raise ValueError(
                                "No local text available for the text baseline."
                            )
                        answer = answer_question(
                            text,
                            row["question"],
                            pages=pages if mode == "vision" else None,
                            concise=True,
                        )
                        record[mode] = {
                            "answer": answer,
                            **answer_scores(answer, row["answers"]),
                            "seconds": perf_counter() - started,
                        }
                    except Exception as error:
                        if isinstance(error, RateLimitError):
                            rate_limited_modes.add(mode)
                        record[mode] = {
                            "error": f"{type(error).__name__}: {error}",
                            "seconds": perf_counter() - started,
                        }
                # Optional field labels enable a third structured-extraction comparison.
                if row.get("fields") and row.get("category"):
                    started = perf_counter()
                    try:
                        document = extract_structured_data(
                            text, row["category"], pages=pages
                        )
                        scores = {}
                        for name, references in row["fields"].items():
                            value = document.fields.get(name)
                            if hasattr(value, "value"):
                                value = value.value
                            elif isinstance(value, dict):
                                value = value.get("value")
                            scores[name] = answer_scores(
                                "" if value is None else str(value), references
                            )
                        record["structured"] = {
                            "fields": scores,
                            "seconds": perf_counter() - started,
                        }
                    except Exception as error:
                        record["structured"] = {
                            "error": f"{type(error).__name__}: {error}"
                        }
            except Exception as error:
                record["error"] = f"{type(error).__name__}: {error}"
            records.append(record)
            write_report(
                output,
                {
                    "status": "running",
                    "summary": {
                        "provider": settings.ai_provider,
                        "models": {
                            "text": model_name(settings),
                            "vision": model_name(settings, vision=True),
                        },
                        "completed": len(records),
                        "requested": limit,
                    },
                    "results": records,
                },
            )
            print(f"Completed {len(records)} / {limit}", flush=True)
    if not records:
        raise ValueError("Manifest contains no evaluation examples.")

    settings = get_settings()
    summary = {
        "examples": len(records),
        "manifest": str(manifest),
        "provider": settings.ai_provider,
        "models": {
            "text": model_name(settings),
            "vision": model_name(settings, vision=True),
        },
        "rerun_modes": list(modes),
        "reused_report": str(reuse) if reuse else None,
    }
    for mode in ("ocr_only", "text", "vision"):
        valid = [
            r[mode]
            for r in records
            if mode in r and "error" not in r[mode] and "not_run" not in r[mode]
        ]
        failed = sum(
            "error" in r or (mode in r and "error" in r[mode]) for r in records
        )
        summary[mode] = {
            "succeeded": len(valid),
            "failed": failed,
            "not_run": len(records) - len(valid) - failed,
        }
        for metric in ("exact_match", "anls", "seconds"):
            summary[mode][metric] = (
                sum(v[metric] for v in valid) / len(valid) if valid else None
            )
    write_report(output, {"summary": summary, "results": records})
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--output", type=Path, default=Path("reports/evaluation.json"))
    parser.add_argument(
        "--modes", nargs="+", choices=["text", "vision"], default=["text", "vision"]
    )
    parser.add_argument(
        "--reuse",
        type=Path,
        help="Reuse unselected modes from a prior report with matching IDs/questions/answers.",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="With --reuse, keep successful results and retry only missing/failed requests. Stop each mode after a rate limit.",
    )
    parser.add_argument(
        "--provider",
        choices=["ollama", "groq"],
        help="Override the .env backend for this run only.",
    )
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be positive")
    with use_provider(args.provider or get_settings().ai_provider):
        evaluate(
            args.manifest,
            args.limit,
            args.output,
            modes=args.modes,
            reuse=args.reuse,
            retry_failed=args.retry_failed,
        )
