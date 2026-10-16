"""Summarize saved measurements without inventing unavailable metrics."""

import json
from pathlib import Path


def read(name):
    path = Path("reports") / name
    return json.loads(path.read_text()) if path.exists() else None


def percent(value):
    return f"{value*100:.1f}%" if value is not None else "Not measured"


def main():
    lines = [
        "# Evaluation status",
        "",
        "These are measured smoke-test results, not full-corpus or production accuracy. No human usability study or cloud deployment is claimed.",
        "",
    ]
    smoke = read("ollama-smoke.json")
    if smoke:
        lines += [
            "## Local model verification",
            "",
            f"Model: `{smoke['model']}`",
            "",
            "| Check | Passed | Seconds |",
            "|---|---|---:|",
        ]
        for name, stage in smoke.get("stages", {}).items():
            lines.append(f"| {name} | {stage['passed']} | {stage['seconds']:.2f} |")
    benchmark = read("ollama-validation-20.json")
    if benchmark and "examples" in benchmark.get("summary", {}):
        summary = benchmark["summary"]
        lines += [
            "",
            "## DocVQA validation sample",
            "",
            f"Questions: {summary['examples']}; same `{summary['models']['text']}` model for text and vision. Dataset source revision is recorded in `data/docvqa/provenance.json`.",
            "",
            "| Pipeline | Succeeded | Failed | Exact match | ANLS-style score | Mean request seconds |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for name in ["ocr_only", "text", "vision"]:
            row = summary[name]
            score = f"{row['anls']:.3f}" if row["anls"] is not None else "Not measured"
            seconds = (
                f"{row['seconds']:.2f}"
                if row["seconds"] is not None
                else "Not measured"
            )
            lines.append(
                f"| {name} | {row['succeeded']} | {row['failed']} | {percent(row['exact_match'])} | {score} | {seconds} |"
            )
        lines += [
            "",
            "Accuracy is over successful requests; failed requests are reported separately. Timing excludes preprocessing and includes backend/model loading or queueing when present. A small seeded-shard sample is not a statistically representative full DocVQA benchmark.",
        ]
    else:
        lines += [
            "",
            "## DocVQA validation sample",
            "",
            f"Run incomplete: {len(benchmark.get('results',[])) if benchmark else 0} checkpointed rows; no final metrics claimed.",
        ]
    ocr = read("synthetic-ocr.json")
    if ocr:
        summary = ocr["summary"]
        comparison = summary.get("synthetic_ground_truth_transcript_comparison", {})
        lines += [
            "",
            "## Synthetic OCR ground truth",
            "",
            f"Pages: {comparison.get('pages',0)}. CER: {percent(comparison.get('micro_cer'))}; WER: {percent(comparison.get('micro_wer'))}. References are the exact generated text used to render these clean pages. This does not replace human transcription of real documents.",
        ]
    quality = read("ollama-quality.json")
    if quality and quality.get("summary"):
        summary = quality["summary"]
        lines += [
            "",
            "## Synthetic field and quality suite",
            "",
            f"Completed: {summary['completed']}/{summary['examples']}; failures: {summary['failed']}. Fixtures: clean invoice, inconsistent invoice total, and form with blank address/signature.",
            "",
            "| Metric | Result |",
            "|---|---:|",
            f"| Field exact match | {percent(summary['field_exact_match'])} |",
            f"| Missing-field precision | {percent(summary['missing_fields']['precision'])} |",
            f"| Missing-field recall | {percent(summary['missing_fields']['recall'])} |",
            f"| Consistency warning precision | {percent(summary['consistency_warnings']['precision'])} |",
            f"| Consistency warning recall | {percent(summary['consistency_warnings']['recall'])} |",
            f"| Evidence quote-location rate | {percent(summary['quote_location_rate'])} |",
            f"| Unsupported-question refusal heuristic | {percent(summary['unsupported_refusal_heuristic_rate'])} |",
            "",
            "Quote location confirms literal source text, not correctness of an extracted value. Refusal scoring is an automated phrase heuristic and requires review. This three-case suite is deliberately small and synthetic.",
        ]
    lines += [
        "",
        "## Remaining external validation",
        "",
        "- Real-document CER/WER requires verified transcripts; machine-reference agreement is reported separately.",
        "- Real field, missing-field, and semantic-contradiction accuracy needs a larger independently labeled dataset.",
        "- Use `docs/USABILITY.md` to record actual participant results; automated interface tests are not a human study.",
        "- `Dockerfile` and `docs/DEPLOYMENT.md` prepare optional deployment. No cloud project, service, or persistent cloud database has been provisioned.",
        "",
    ]
    Path("docs").mkdir(exist_ok=True)
    output = Path("docs/EVALUATION.md")
    output.write_text("\n".join(lines))
    print(output)


if __name__ == "__main__":
    main()
