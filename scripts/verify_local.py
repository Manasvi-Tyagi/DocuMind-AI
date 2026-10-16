"""Run real local Gemma checks and checkpoint each stage for review."""

import io
import json
from pathlib import Path
from time import perf_counter

from documind.ai import answer_question, classify_document, extract_structured_data
from documind.config import get_settings, use_provider
from documind.documents import prepare_document, pages_text


def main():
    settings = get_settings(provider="ollama")
    output = Path("reports/ollama-smoke.json")
    output.parent.mkdir(exist_ok=True)
    report = {"provider": "ollama", "model": settings.ollama_model, "stages": {}}
    upload = io.BytesIO(Path("examples/demo-invoice.pdf").read_bytes())
    upload.name = "demo-invoice.pdf"
    pages = prepare_document(upload, [1])
    text = pages_text(pages)

    def stage(name, operation, validate):
        started = perf_counter()
        print(f"Starting {name}", flush=True)
        try:
            result = operation()
            passed = validate(result)
            report["stages"][name] = {
                "passed": passed,
                "result": result,
                "seconds": round(perf_counter() - started, 2),
            }
        except Exception as error:
            report["stages"][name] = {
                "passed": False,
                "error": f"{type(error).__name__}: {error}",
                "seconds": round(perf_counter() - started, 2),
            }
        output.write_text(json.dumps(report, indent=2))
        print(name, json.dumps(report["stages"][name]), flush=True)

    with use_provider("ollama"):
        stage(
            "text_qa",
            lambda: answer_question(
                text, "What is the total? Return only the number.", concise=True
            ),
            lambda result: result.strip() == "220",
        )
        # No OCR text is supplied here: this verifies actual image understanding.
        stage(
            "image_only_qa",
            lambda: answer_question(
                "",
                "What is the invoice number? Return only the invoice number.",
                pages=pages,
                concise=True,
            ),
            lambda result: "INV-1045" in result,
        )
        stage(
            "classification",
            lambda: classify_document(text, pages=pages),
            lambda result: result == "Invoice",
        )
        stage(
            "structured_extraction",
            lambda: extract_structured_data(text, "Invoice", pages=pages).model_dump(),
            lambda result: result["category"] == "Invoice"
            and bool(result["fields"])
            and any(
                str(v.get("value") if isinstance(v, dict) else v).strip()
                in {"220", "220.0", "220.00"}
                for k, v in result["fields"].items()
                if "total" in k.lower() and "subtotal" not in k.lower()
            ),
        )
    print("Report saved:", output, flush=True)
    if not all(s["passed"] for s in report["stages"].values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
