"""Step 10 — Evaluate field extraction, missing-field detection and
consistency detection on the labelled synthetic documents.

Usage:
    python scripts/make_synthetic_forms.py
    python scripts/evaluate_forms.py --data data/synthetic_forms
"""

import argparse
import json
import re
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import settings  # noqa: E402
from src.document_processor import preprocess_image  # noqa: E402
from src.llm import get_backend  # noqa: E402
from src.llm.factory import BACKENDS, model_name  # noqa: E402
from src.ocr import run_ocr  # noqa: E402
from src.pipeline import PageInput, check_consistency, detect_missing_fields, extract_fields  # noqa: E402


def values_match(predicted: str | None, expected: str) -> bool:
    if predicted is None:
        return False
    a, b = predicted.strip().lower(), expected.strip().lower()
    if a == b:
        return True
    num = lambda s: re.sub(r"[^\d.]", "", s)  # noqa: E731  compare amounts numerically
    if num(b) and re.fullmatch(r"[\d,.\s$€£₹]+", b):
        try:
            return abs(float(num(a)) - float(num(b))) < 0.005
        except ValueError:
            return False
    return re.sub(r"\s+", " ", a) == re.sub(r"\s+", " ", b)


def prf(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if tp + fp else None
    r = tp / (tp + fn) if tp + fn else None
    f = 2 * p * r / (p + r) if p and r else (0.0 if p is not None and r is not None else None)
    return {"precision": p, "recall": r, "f1": f, "tp": tp, "fp": fp, "fn": fn}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=Path("data/synthetic_forms"))
    parser.add_argument("--backend", choices=BACKENDS, default=None, help="default: LLM_BACKEND from .env")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", type=Path, default=Path("results/forms"))
    args = parser.parse_args()

    labels = json.loads((args.data / "labels.json").read_text())[: args.limit]
    backend = get_backend(args.backend)
    args.out.mkdir(parents=True, exist_ok=True)

    records = []
    field_total = field_correct = 0
    hallucinated_fills = blank_fields = 0
    tp = fp = fn = 0
    cons_total = cons_correct = cons_undetermined = 0
    latencies: dict[str, list[float]] = {"extraction": [], "missing": [], "consistency": []}

    for i, doc in enumerate(labels, 1):
        image = preprocess_image(Image.open(args.data / doc["png"]), settings.target_long_side, False)
        page = PageInput(1, image, run_ocr(image))
        required = list(doc["fields"].keys())
        record = {"id": doc["id"], "template": doc["template"]}

        try:
            extraction = extract_fields(backend, [page], required)
            latencies["extraction"].append(extraction.call.latency_s)
            got = {f.field.strip().lower(): f.value for f in extraction.fields}
            record["extraction"] = {}
            for name, expected in doc["fields"].items():
                predicted = got.get(name.lower())
                if expected is None:
                    blank_fields += 1
                    hallucinated_fills += predicted is not None
                    record["extraction"][name] = {"expected": None, "predicted": predicted}
                elif name == "Signature":
                    continue  # a signature has no text value to compare
                else:
                    field_total += 1
                    ok = values_match(predicted, expected)
                    field_correct += ok
                    record["extraction"][name] = {"expected": expected, "predicted": predicted, "correct": ok}

            report, info = detect_missing_fields(backend, [page], doc["template"], required)
            latencies["missing"].append(info.latency_s)
            predicted_missing = set(report.missing)
            true_missing = set(doc["missing"])
            tp += len(predicted_missing & true_missing)
            fp += len(predicted_missing - true_missing)
            fn += len(true_missing - predicted_missing)
            record["missing"] = {"expected": sorted(true_missing), "predicted": sorted(predicted_missing)}

            if doc["consistent"] is not None:
                _, checks, _, info = check_consistency(backend, [page])
                latencies["consistency"].append(info.latency_s)
                decided = [c for c in checks if c.passed is not None]
                cons_total += 1
                if not decided:
                    cons_undetermined += 1
                    predicted_consistent = None
                else:
                    predicted_consistent = all(c.passed for c in decided)
                    cons_correct += predicted_consistent == doc["consistent"]
                record["consistency"] = {"expected": doc["consistent"], "predicted": predicted_consistent,
                                         "checks": [c.model_dump() for c in checks]}
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"[:500]

        records.append(record)
        print(f"[{i}/{len(labels)}] {doc['id']} {'ERROR ' + record['error'] if 'error' in record else 'ok'}")

    mean = lambda xs: round(sum(xs) / len(xs), 3) if xs else None  # noqa: E731
    summary = {
        "model": model_name(args.backend),
        "n_documents": len(labels),
        "errors": sum(1 for r in records if "error" in r),
        "field_extraction_accuracy": round(field_correct / field_total, 4) if field_total else None,
        "fields_evaluated": field_total,
        "blank_field_hallucination_rate": round(hallucinated_fills / blank_fields, 4) if blank_fields else None,
        "missing_field_detection": prf(tp, fp, fn),
        "consistency_detection_accuracy": round(cons_correct / (cons_total - cons_undetermined), 4)
        if cons_total - cons_undetermined else None,
        "consistency_undetermined": cons_undetermined,
        "consistency_documents": cons_total,
        "latency_mean_s": {k: mean(v) for k, v in latencies.items()},
    }
    (args.out / "details.json").write_text(json.dumps(records, indent=2))
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
