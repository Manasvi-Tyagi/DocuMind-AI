"""Render synthetic, explicitly labeled cases. Not DocVQA or human annotations."""

import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


def main():
    folder = Path("examples/validation")
    folder.mkdir(parents=True, exist_ok=True)
    cases = []
    for name, total in [("invoice_valid", 220), ("invoice_inconsistent", 250)]:
        text = f"INVOICE\nInvoice Number: INV-2001\nSeller: Example Supplies\nSubtotal: 200\nTax: 20\nTotal: {total}\nInvoice Date: 2026-10-16"
        cases.append(
            {
                "id": name,
                "category": "Invoice",
                "text": text,
                "fields": {
                    "invoice_number": ["INV-2001"],
                    "subtotal": ["200"],
                    "tax": ["20"],
                    "total": [str(total)],
                },
                "required_fields": [
                    "invoice_number",
                    "subtotal",
                    "tax",
                    "total",
                    "client_name",
                ],
                "missing_fields": ["client_name"],
                "warning_rules": ["Subtotal + tax vs. total"] if total == 250 else [],
                "question": "What is the total?",
                "answers": [str(total)],
                "unsupported_question": "What is the seller bank account number?",
            }
        )
    cases.append(
        {
            "id": "application_missing",
            "category": "Application Form",
            "text": "APPLICATION FORM\nApplicant Name: Ada Example\nDate: 2026-10-16\nIdentification Number: ID-123\nAddress:\nSignature:",
            "fields": {
                "applicant_name": ["Ada Example"],
                "date": ["2026-10-16"],
                "identification_number": ["ID-123"],
            },
            "required_fields": [
                "applicant_name",
                "date",
                "identification_number",
                "address",
                "signature_status",
            ],
            "missing_fields": ["address", "signature_status"],
            "warning_rules": [],
            "question": "What is the applicant name?",
            "answers": ["Ada Example"],
            "unsupported_question": "What is the applicant phone number?",
        }
    )
    for row in cases:
        image = Image.new("RGB", (1300, 850), "white")
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default(size=34)
        draw.multiline_text(
            (65, 60), row.pop("text"), fill="black", font=font, spacing=20
        )
        # Reconstruct the exact rendered transcript from known labels, not OCR.
        if row["id"].startswith("invoice"):
            transcript = f'INVOICE\nInvoice Number: INV-2001\nSeller: Example Supplies\nSubtotal: 200\nTax: 20\nTotal: {row["answers"][0]}\nInvoice Date: 2026-10-16'
        else:
            transcript = "APPLICATION FORM\nApplicant Name: Ada Example\nDate: 2026-10-16\nIdentification Number: ID-123\nAddress:\nSignature:"
        image.save(folder / (row["id"] + ".png"))
        (folder / (row["id"] + ".txt")).write_text(transcript)
        row.update(
            document=row["id"] + ".png",
            page=1,
            document_id=row["id"],
            transcript=row["id"] + ".txt",
            transcript_kind="synthetic_ground_truth",
        )
    (folder / "cases.jsonl").write_text(
        "\n".join(json.dumps(row) for row in cases) + "\n"
    )
    print(
        "Created 3 synthetic cases with field, missing-field, consistency, and transcript labels."
    )


if __name__ == "__main__":
    main()
