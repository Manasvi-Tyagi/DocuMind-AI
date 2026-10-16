"""Step 6 — Predefined forms and their required fields.

"Application form" uses exactly the example fields from the project brief.
The other templates are editable defaults; the UI also accepts custom fields.
"""

FORM_TEMPLATES: dict[str, list[str]] = {
    "Application form": ["Name", "Address", "Date", "Signature", "ID Number"],
    "Invoice": [
        "Invoice Number",
        "Invoice Date",
        "Seller Name",
        "Buyer Name",
        "Subtotal",
        "Tax",
        "Total Amount",
    ],
    "Receipt": ["Merchant Name", "Date", "Total Amount", "Payment Method"],
}
