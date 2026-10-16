"""Create a synthetic invoice and a tiny labeled evaluation manifest."""

import json
from pathlib import Path
import pymupdf


def main():
    folder = Path("examples")
    folder.mkdir(exist_ok=True)
    with pymupdf.open() as pdf:
        page = pdf.new_page()
        page.insert_text((70, 70), "DEMO INVOICE", fontsize=22)
        page.insert_text(
            (70, 115),
            "Invoice Number: INV-1045\nInvoice Date: 2026-10-15\nSeller: Example Supplies\nClient: Example College\nItem: Printer paper, quantity 2, unit price 100, amount 200\nSubtotal: 200\nTax: 20\nTotal: 220\nPayment Status: Unpaid",
            fontsize=12,
        )
        pdf.save(folder / "demo-invoice.pdf")
    rows = [
        {
            "id": 1,
            "document_id": "demo-invoice",
            "document": "demo-invoice.pdf",
            "page": 1,
            "question": "What is the invoice number?",
            "answers": ["INV-1045"],
        },
        {
            "id": 2,
            "document_id": "demo-invoice",
            "document": "demo-invoice.pdf",
            "page": 1,
            "question": "What is the total?",
            "answers": ["220"],
            "category": "Invoice",
            "fields": {"invoice_number": ["INV-1045"], "total": ["220"]},
        },
    ]
    (folder / "demo.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n"
    )
    print(
        "Created examples/demo-invoice.pdf and examples/demo.jsonl (synthetic, not DocVQA)."
    )


if __name__ == "__main__":
    main()
