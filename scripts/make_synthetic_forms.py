"""Generate labelled synthetic documents for Steps 6, 7 and 10.

DocVQA has no labels for field extraction, missing fields or arithmetic
consistency, so this script renders fictional invoices and application
forms whose ground truth is known exactly:

* invoices — some have a wrong total or wrong subtotal (inconsistent)
* application forms — some have 1-2 required fields left blank (missing)

Every document is written as PDF and PNG, with labels in labels.json.

Usage:
    python scripts/make_synthetic_forms.py --invoices 25 --forms 25
"""

import argparse
import json
import random
from pathlib import Path

import pymupdf

FIRST = ["Aarav", "Maria", "Chen", "Fatima", "Lukas", "Priya", "Diego", "Amara", "Kenji", "Sofia"]
LAST = ["Sharma", "Garcia", "Wei", "Khan", "Muller", "Iyer", "Lopez", "Okafor", "Sato", "Rossi"]
STREETS = ["Maple Street", "Station Road", "Lake View", "Park Avenue", "Hill Lane", "Market Road"]
CITIES = ["Pune 411001", "Austin TX 78701", "Leeds LS1 4AP", "Lyon 69002", "Osaka 530-0001"]
COMPANIES = ["Northwind Supplies", "Bluepeak Traders", "Crescent Office Co.", "Vertex Parts Ltd", "Lumen Print House"]
ITEMS = ["A4 paper (box)", "Toner cartridge", "Desk lamp", "USB-C cable", "Notebook pack", "Stapler", "Monitor stand"]

PAGE_W, PAGE_H = 595, 842  # A4 in points


def person(rng):
    return f"{rng.choice(FIRST)} {rng.choice(LAST)}"


def address(rng):
    return f"{rng.randint(1, 250)} {rng.choice(STREETS)}, {rng.choice(CITIES)}"


def date(rng):
    return f"{rng.randint(2024, 2026)}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"


def money(x: float) -> str:
    return f"{x:,.2f}"


def save(doc: pymupdf.Document, out_dir: Path, stem: str) -> tuple[str, str]:
    pdf_path = out_dir / f"{stem}.pdf"
    png_path = out_dir / f"{stem}.png"
    doc.save(pdf_path)
    doc[0].get_pixmap(matrix=pymupdf.Matrix(200 / 72, 200 / 72), alpha=False).save(png_path)
    doc.close()
    return pdf_path.name, png_path.name


def make_invoice(rng: random.Random, index: int, out_dir: Path, inconsistency: str | None) -> dict:
    seller, buyer = rng.sample(COMPANIES, 2)
    number = f"INV-{rng.randint(1000, 9999)}"
    issued = date(rng)

    items = []
    for desc in rng.sample(ITEMS, rng.randint(2, 4)):
        qty = rng.randint(1, 6)
        price = round(rng.uniform(3, 150), 2)
        items.append((desc, qty, price, round(qty * price, 2)))

    subtotal = round(sum(i[3] for i in items), 2)
    tax = round(subtotal * 0.10, 2)
    total = round(subtotal + tax, 2)

    printed_subtotal, printed_total = subtotal, total
    if inconsistency == "total":
        printed_total = round(total + rng.choice([-1, 1]) * rng.randint(5, 60), 2)
    elif inconsistency == "subtotal":
        printed_subtotal = round(subtotal + rng.choice([-1, 1]) * rng.randint(5, 60), 2)
        printed_total = round(printed_subtotal + tax, 2)  # total still = subtotal + tax

    doc = pymupdf.open()
    page = doc.new_page(width=PAGE_W, height=PAGE_H)
    page.insert_text((50, 70), "INVOICE", fontsize=26, fontname="hebo")
    page.insert_text((50, 100), seller, fontsize=12, fontname="hebo")
    page.insert_text((50, 116), address(rng), fontsize=10)
    page.insert_text((370, 70), f"Invoice Number: {number}", fontsize=11)
    page.insert_text((370, 88), f"Invoice Date: {issued}", fontsize=11)
    page.insert_text((50, 160), "Bill To:", fontsize=11, fontname="hebo")
    page.insert_text((50, 176), buyer, fontsize=11)
    page.insert_text((50, 192), address(rng), fontsize=10)

    y = 240
    for x, head in ((50, "Description"), (300, "Qty"), (360, "Unit Price"), (470, "Amount")):
        page.insert_text((x, y), head, fontsize=11, fontname="hebo")
    page.draw_line((50, y + 6), (545, y + 6))
    for desc, qty, price, amount in items:
        y += 22
        page.insert_text((50, y), desc, fontsize=10)
        page.insert_text((300, y), str(qty), fontsize=10)
        page.insert_text((360, y), money(price), fontsize=10)
        page.insert_text((470, y), money(amount), fontsize=10)
    page.draw_line((50, y + 10), (545, y + 10))

    y += 36
    for label, value in (("Subtotal", printed_subtotal), ("Tax (10%)", tax), ("Total", printed_total)):
        page.insert_text((360, y), f"{label}:", fontsize=11, fontname="hebo" if label == "Total" else "helv")
        page.insert_text((470, y), money(value), fontsize=11, fontname="hebo" if label == "Total" else "helv")
        y += 20

    stem = f"invoice_{index:03d}"
    pdf_name, png_name = save(doc, out_dir, stem)
    return {
        "id": stem,
        "pdf": pdf_name,
        "png": png_name,
        "template": "Invoice",
        "fields": {
            "Invoice Number": number,
            "Invoice Date": issued,
            "Seller Name": seller,
            "Buyer Name": buyer,
            "Subtotal": money(printed_subtotal),
            "Tax": money(tax),
            "Total Amount": money(printed_total),
        },
        "missing": [],
        "consistent": inconsistency is None,
        "inconsistency": inconsistency,
    }


def draw_signature(page: pymupdf.Page, x: float, y: float, rng: random.Random) -> None:
    points = [pymupdf.Point(x + i * 9, y - 8 + rng.uniform(-9, 9)) for i in range(14)]
    page.draw_polyline(points, color=(0.05, 0.1, 0.45), width=1.4)


def make_application_form(rng: random.Random, index: int, out_dir: Path, missing: list[str]) -> dict:
    values = {
        "Name": person(rng),
        "Address": address(rng),
        "Date": date(rng),
        "ID Number": f"ID-{rng.randint(100000, 999999)}",
        "Signature": "signed",
    }

    doc = pymupdf.open()
    page = doc.new_page(width=PAGE_W, height=PAGE_H)
    page.insert_text((50, 70), "APPLICATION FORM", fontsize=22, fontname="hebo")
    page.insert_text((50, 92), "Please complete all fields in block letters.", fontsize=10)

    y = 150
    for label in ("Name", "Address", "Date", "ID Number"):
        page.insert_text((50, y), f"{label}:", fontsize=12, fontname="hebo")
        page.draw_line((150, y + 4), (545, y + 4), color=(0.5, 0.5, 0.5))
        if label not in missing:
            page.insert_text((155, y), values[label], fontsize=12)
        y += 50

    y += 30
    page.insert_text((50, y), "Signature:", fontsize=12, fontname="hebo")
    page.draw_line((150, y + 4), (380, y + 4), color=(0.5, 0.5, 0.5))
    if "Signature" not in missing:
        draw_signature(page, 160, y, rng)

    stem = f"application_{index:03d}"
    pdf_name, png_name = save(doc, out_dir, stem)
    return {
        "id": stem,
        "pdf": pdf_name,
        "png": png_name,
        "template": "Application form",
        "fields": {k: (None if k in missing else v) for k, v in values.items()},
        "missing": missing,
        "consistent": None,
        "inconsistency": None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--invoices", type=int, default=25)
    parser.add_argument("--forms", type=int, default=25)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--out", type=Path, default=Path("data/synthetic_forms"))
    args = parser.parse_args()

    rng = random.Random(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    labels = []

    for i in range(args.invoices):
        # About 40% of invoices are inconsistent.
        kind = rng.choices([None, "total", "subtotal"], weights=[6, 2, 2])[0]
        labels.append(make_invoice(rng, i, args.out, kind))

    required = ["Name", "Address", "Date", "Signature", "ID Number"]
    for i in range(args.forms):
        n_missing = rng.choices([0, 1, 2], weights=[4, 4, 2])[0]
        labels.append(make_application_form(rng, i, args.out, rng.sample(required, n_missing)))

    (args.out / "labels.json").write_text(json.dumps(labels, indent=2))
    print(f"Wrote {len(labels)} documents and labels.json to {args.out}")


if __name__ == "__main__":
    main()
