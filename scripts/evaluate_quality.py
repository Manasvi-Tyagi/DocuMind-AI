"""Measure labeled fields, missing fields, consistency, and groundedness.

This report labels its data source explicitly. Synthetic results do not establish
real-world accuracy, and keyword-based refusal scoring needs human review.
"""

import argparse
import json
from pathlib import Path
from time import perf_counter

from documind.ai import answer_question, extract_structured_data
from documind.documents import prepare_document, pages_text
from documind.evaluation import answer_scores
from documind.evidence import locate_quote
from documind.validation import detect_missing_fields, check_document_consistency
import io
from documind.reporting import write_report
from documind.config import get_settings, model_name, use_provider


def set_counts(predicted, expected):
    return {
        "tp": len(predicted & expected),
        "fp": len(predicted - expected),
        "fn": len(expected - predicted),
    }


def rates(counts):
    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
    return {
        **counts,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
    }


def evaluate_quality(manifest, output):
    rows = [
        json.loads(line) for line in manifest.read_text().splitlines() if line.strip()
    ]
    records = []
    for row in rows:
        started = perf_counter()
        record = {
            "id": row["id"],
            "data_kind": row.get("transcript_kind", "user_labeled"),
        }
        try:
            file = io.BytesIO((manifest.parent / row["document"]).read_bytes())
            file.name = row["document"]
            pages = prepare_document(file, [row.get("page", 1)])
            text = pages_text(pages)
            document = extract_structured_data(
                text,
                row["category"],
                pages=pages,
                required_fields=row["required_fields"],
            )
            record["extraction"] = document.model_dump()
            record["field_scores"] = {
                name: answer_scores(
                    (
                        ""
                        if document.fields.get(name) is None
                        or document.fields[name].value is None
                        else str(document.fields[name].value)
                    ),
                    answers,
                )
                for name, answers in row["fields"].items()
            }
            missing = {
                r["field"].casefold().replace(" ", "_")
                for r in detect_missing_fields(
                    document.model_dump(), row["category"], row["required_fields"]
                )
                if r["status"] == "Not found"
            }
            record["missing_fields"] = set_counts(missing, set(row["missing_fields"]))
            warning = {
                r["rule"]
                for r in check_document_consistency(
                    document.model_dump(), row["category"]
                )
                if r["status"] == "Warning"
            }
            record["consistency_warnings"] = set_counts(
                warning, set(row["warning_rules"])
            )
            record["quote_checks"] = {
                name: locate_quote(pages, field.evidence or "", field.page)
                for name, field in document.fields.items()
                if field.value is not None
            }
            answer = answer_question(
                text, row["unsupported_question"], pages=pages, concise=False
            )
            # This is an automated heuristic, not a human hallucination judgment.
            record["unsupported_answer"] = answer
            record["refusal_heuristic"] = any(
                term in answer.casefold()
                for term in [
                    "not provided",
                    "not found",
                    "cannot find",
                    "could not find",
                    "not mentioned",
                    "does not contain",
                    "doesn't contain",
                    "not available",
                    "no information",
                    "not listed",
                    "does not include",
                    "doesn't include",
                    "not specified",
                    "not stated",
                ]
            )
        except Exception as error:
            record["error"] = f"{type(error).__name__}: {error}"
        record["seconds"] = perf_counter() - started
        records.append(record)
        write_report(output, {"status": "running", "results": records})
        print(f"Quality completed {len(records)}/{len(rows)}", flush=True)
    valid = [r for r in records if "error" not in r]
    settings = get_settings()
    summary = {
        "provider": settings.ai_provider,
        "model": model_name(settings, vision=True),
        "examples": len(rows),
        "completed": len(valid),
        "failed": len(rows) - len(valid),
        "data_kind": sorted({r["data_kind"] for r in records}),
    }
    fields = [s for r in valid for s in r["field_scores"].values()]
    summary["field_exact_match"] = (
        sum(s["exact_match"] for s in fields) / len(fields) if fields else None
    )
    for label in ["missing_fields", "consistency_warnings"]:
        summary[label] = rates(
            {key: sum(r[label][key] for r in valid) for key in ["tp", "fp", "fn"]}
        )
    quotes = [check for r in valid for check in r["quote_checks"].values()]
    summary["quote_location_rate"] = (
        sum(q["status"] == "located" for q in quotes) / len(quotes) if quotes else None
    )
    summary["unsupported_refusal_heuristic_rate"] = (
        sum(r["refusal_heuristic"] for r in valid) / len(valid) if valid else None
    )
    summary["caveat"] = (
        "Small labeled suite; synthetic data and automated refusal scoring require separate real-document/human evaluation."
    )
    write_report(output, {"summary": summary, "results": records})
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, default=Path("reports/quality.json"))
    parser.add_argument("--provider", choices=["ollama", "groq"], default="ollama")
    args = parser.parse_args()
    with use_provider(args.provider):
        evaluate_quality(args.manifest, args.output)
