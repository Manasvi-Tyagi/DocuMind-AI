from .categories import get_required_template

EXPECTED_FIELDS = {
    "Resume": [
        "name",
        "email",
        "phone",
        "education",
        "skills",
        "experience",
        "projects",
        "certifications",
    ],
    "Invoice": [
        "invoice_number",
        "invoice_date",
        "due_date",
        "seller_name",
        "client_name",
        "items",
        "subtotal",
        "tax",
        "total",
        "payment_status",
    ],
    "Application Form": [
        "applicant_name",
        "address",
        "date",
        "application_type",
        "identification_number",
        "signature_status",
    ],
    "Report": ["title", "author", "date", "purpose", "key_findings", "conclusion"],
    "Statement": [
        "statement_type",
        "account_holder",
        "statement_period",
        "opening_balance",
        "closing_balance",
        "transactions",
    ],
    "Scanned Form": [
        "form_title",
        "name",
        "address",
        "date",
        "identification_number",
        "filled_fields",
        "signature_status",
    ],
    "Certificate": [
        "certificate_type",
        "recipient_name",
        "issuing_organization",
        "issue_date",
        "completion_date",
        "certificate_number",
    ],
    "Receipt": [
        "receipt_number",
        "purchase_date",
        "merchant",
        "items",
        "subtotal",
        "tax",
        "total",
        "payment_method",
    ],
}


def detect_missing_fields(structured_data, category, required_fields=None):
    """

    Compare category-specific expected fields with extracted values.

    A field is reported as not found when it is absent or has an empty value.

    """
    extracted_fields = structured_data.get("fields", {})
    saved = get_required_template(category) if required_fields is None else None
    expected_fields = (
        required_fields
        if required_fields is not None
        else saved if saved is not None else EXPECTED_FIELDS.get(category, [])
    )
    results = []
    for field_name in expected_fields:
        field_data = extracted_fields.get(field_name)
        if isinstance(field_data, dict) and "value" in field_data:
            value = field_data.get("value")
        else:
            value = field_data
        is_empty = (
            field_data is None
            or value is None
            or (
                isinstance(value, str)
                and (
                    not value.strip()
                    or value.strip().casefold()
                    in {"not found", "missing", "not provided", "blank", "none", "null"}
                )
            )
            or (
                "signature" in field_name.casefold()
                and isinstance(value, str)
                and value.strip().casefold()
                in {"unsigned", "not signed", "no signature", "absent"}
            )
            or (isinstance(value, (list, dict)) and len(value) == 0)
        )
        results.append(
            {
                "field": field_name.replace("_", " ").title(),
                "status": "Not found" if is_empty else "Found",
                "value": value,
            }
        )
    return results


def _unwrap_field(field_data):
    """Return the extracted value from the app's {value, confidence, evidence} shape."""
    if isinstance(field_data, dict) and "value" in field_data:
        return field_data.get("value")
    return field_data


def _to_number(value):
    """Convert common currency/number strings to float; return None if unclear."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = value.strip().replace(",", "")
        for symbol in ("₹", "$", "€", "£", "INR", "Rs.", "Rs", "USD"):
            cleaned = cleaned.replace(symbol, "")
        cleaned = cleaned.strip()
        try:
            return float(cleaned)
        except ValueError:
            return None
    return None


def _find_value(fields, *possible_names):
    """Find a field by one of its accepted names, ignoring case and underscores."""
    normalized = {
        str(key).lower().replace("_", "").replace(" ", ""): _unwrap_field(value)
        for key, value in fields.items()
    }
    for name in possible_names:
        key = name.lower().replace("_", "").replace(" ", "")
        if key in normalized:
            return normalized[key]
    return None


def _check_result(rule, status, details, evidence=""):
    return {
        "rule": rule,
        "status": status,
        "details": details,
        "evidence": evidence or "No supporting evidence was supplied for this check.",
    }


def _check_invoice(fields):
    checks = []
    items = _find_value(fields, "items", "line_items", "invoice_items")
    subtotal = _to_number(_find_value(fields, "subtotal", "sub_total"))
    tax = _to_number(_find_value(fields, "tax", "gst", "tax_amount"))
    total = _to_number(_find_value(fields, "total", "grand_total", "total_amount"))
    if isinstance(items, list) and items:
        item_sum = 0.0
        all_amounts_known = True
        quantity_checks = 0
        for item in items:
            if not isinstance(item, dict):
                all_amounts_known = False
                continue
            quantity = _to_number(item.get("quantity", item.get("qty")))
            unit_price = _to_number(
                item.get(
                    "unit_price",
                    item.get("unit price", item.get("rate", item.get("price"))),
                )
            )
            amount = _to_number(
                item.get("amount", item.get("total", item.get("line_total")))
            )
            if quantity is not None and unit_price is not None and (amount is not None):
                quantity_checks += 1
                calculated = quantity * unit_price
                if abs(calculated - amount) > 0.02:
                    checks.append(
                        _check_result(
                            "Line-item calculation",
                            "Warning",
                            f"Quantity × unit price is {calculated:.2f}, but the extracted line amount is {amount:.2f}.",
                            str(item),
                        )
                    )
                else:
                    checks.append(
                        _check_result(
                            "Line-item calculation",
                            "Passed",
                            f"Quantity × unit price matches the line amount ({amount:.2f}).",
                            str(item),
                        )
                    )
            if amount is None:
                all_amounts_known = False
            else:
                item_sum += amount
        if quantity_checks == 0:
            checks.append(
                _check_result(
                    "Line-item calculation",
                    "Unable to verify",
                    "The items did not contain enough numeric quantity, unit-price and amount values.",
                )
            )
        if all_amounts_known and subtotal is not None:
            if abs(item_sum - subtotal) <= 0.02:
                checks.append(
                    _check_result(
                        "Subtotal vs. line items",
                        "Passed",
                        f"Line-item amounts sum to {item_sum:.2f}, matching the subtotal.",
                    )
                )
            else:
                checks.append(
                    _check_result(
                        "Subtotal vs. line items",
                        "Warning",
                        f"Line-item amounts sum to {item_sum:.2f}, but the extracted subtotal is {subtotal:.2f}.",
                    )
                )
        else:
            checks.append(
                _check_result(
                    "Subtotal vs. line items",
                    "Unable to verify",
                    "A subtotal or one or more line-item amounts could not be read reliably.",
                )
            )
    else:
        checks.append(
            _check_result(
                "Subtotal vs. line items",
                "Unable to verify",
                "No structured list of invoice items was available.",
            )
        )
    if subtotal is not None and tax is not None and (total is not None):
        calculated_total = subtotal + tax
        if abs(calculated_total - total) <= 0.02:
            checks.append(
                _check_result(
                    "Subtotal + tax vs. total",
                    "Passed",
                    f"Subtotal + tax = {calculated_total:.2f}, matching the total.",
                )
            )
        else:
            checks.append(
                _check_result(
                    "Subtotal + tax vs. total",
                    "Warning",
                    f"Subtotal + tax = {calculated_total:.2f}, but the extracted total is {total:.2f}.",
                )
            )
    else:
        checks.append(
            _check_result(
                "Subtotal + tax vs. total",
                "Unable to verify",
                "Subtotal, tax or total was not available as a numeric value.",
            )
        )
    return checks


def _parse_date(value):
    """Parse common date formats without guessing ambiguous dates."""
    from datetime import datetime

    if not isinstance(value, str) or not value.strip():
        return None
    value = value.strip()
    formats = (
        "%Y-%m-%d",
        "%d %B %Y",
        "%d %b %Y",
        "%B %d, %Y",
        "%b %d, %Y",
        "%d/%m/%Y",
        "%m/%d/%Y",
        "%d-%m-%Y",
        "%d.%m.%Y",
    )
    parsed = set()
    for fmt in formats:
        try:
            parsed.add(datetime.strptime(value, fmt).date())
        except ValueError:
            continue
    return next(iter(parsed)) if len(parsed) == 1 else None


def _check_date_order(fields, earlier_names, later_names, rule):
    earlier_raw = _find_value(fields, *earlier_names)
    later_raw = _find_value(fields, *later_names)
    earlier = _parse_date(earlier_raw)
    later = _parse_date(later_raw)
    if earlier is None or later is None:
        return _check_result(
            rule,
            "Unable to verify",
            "Both dates were not available in a supported, unambiguous format.",
            f"Earlier date: {earlier_raw}; later date: {later_raw}",
        )
    if earlier <= later:
        return _check_result(
            rule,
            "Passed",
            f"{earlier_raw} is not later than {later_raw}.",
            f"{earlier_raw}; {later_raw}",
        )
    return _check_result(
        rule,
        "Warning",
        f"The date order may be inconsistent: {earlier_raw} occurs after {later_raw}.",
        f"{earlier_raw}; {later_raw}",
    )


def _check_resume(fields):
    checks = []
    education = _find_value(fields, "education")
    experience = _find_value(fields, "experience")
    if isinstance(education, list) and education:
        parsed_periods = []
        for entry in education:
            if isinstance(entry, dict):
                period = (
                    entry.get("period") or entry.get("date") or entry.get("duration")
                )
                if period:
                    parsed_periods.append(str(period))
        checks.append(
            _check_result(
                "Education timeline",
                "Unable to verify",
                "Education entries were found. Automatic overlap detection is not reliable for all period formats; review the listed periods.",
                "; ".join(parsed_periods),
            )
        )
    else:
        checks.append(
            _check_result(
                "Education timeline",
                "Unable to verify",
                "No structured education list was available.",
            )
        )
    if isinstance(experience, list) and experience:
        checks.append(
            _check_result(
                "Experience timeline",
                "Unable to verify",
                "Work experience entries were found, but their date formats need reliable normalization before comparing them.",
            )
        )
    else:
        checks.append(
            _check_result(
                "Experience timeline",
                "Unable to verify",
                "No structured work-experience list was available.",
            )
        )
    return checks


def _check_statement(fields):
    checks = []
    opening = _to_number(_find_value(fields, "opening_balance", "opening"))
    closing = _to_number(_find_value(fields, "closing_balance", "closing"))
    transactions = _find_value(fields, "transactions")
    if isinstance(transactions, list) and opening is not None and (closing is not None):
        credits = 0.0
        debits = 0.0
        known = True
        for txn in transactions:
            if not isinstance(txn, dict):
                known = False
                break
            amount = _to_number(txn.get("amount", txn.get("value")))
            if amount is None:
                known = False
                break
            kind = str(txn.get("type", txn.get("transaction_type", ""))).lower()
            if any((word in kind for word in ("credit", "deposit", "received"))):
                credits += abs(amount)
            elif any(
                (word in kind for word in ("debit", "withdrawal", "payment", "spent"))
            ):
                debits += abs(amount)
            else:
                known = False
                break
        if known:
            calculated = opening + credits - debits
            if abs(calculated - closing) <= 0.02:
                checks.append(
                    _check_result(
                        "Balance reconciliation",
                        "Passed",
                        f"Opening balance + credits − debits = {calculated:.2f}, matching closing balance.",
                    )
                )
            else:
                checks.append(
                    _check_result(
                        "Balance reconciliation",
                        "Warning",
                        f"Calculated closing balance is {calculated:.2f}, but extracted closing balance is {closing:.2f}.",
                    )
                )
        else:
            checks.append(
                _check_result(
                    "Balance reconciliation",
                    "Unable to verify",
                    "Some transactions lack a clear amount or credit/debit type.",
                )
            )
    else:
        checks.append(
            _check_result(
                "Balance reconciliation",
                "Unable to verify",
                "Opening balance, closing balance or a structured transaction list is missing.",
            )
        )
    return checks


def _check_receipt(fields):
    return _check_invoice(fields)


def _check_application_or_form(fields):
    checks = []
    for names, label in (
        (("date_of_birth", "dob"), "Date of birth"),
        (("application_date", "submission_date"), "Application/submission date"),
    ):
        raw = _find_value(fields, *names)
        parsed = _parse_date(raw)
        if raw is None:
            checks.append(
                _check_result(
                    f"{label} format",
                    "Unable to verify",
                    "The field was not extracted.",
                )
            )
        elif parsed is None:
            checks.append(
                _check_result(
                    f"{label} format",
                    "Unable to verify",
                    "The date was extracted but could not be parsed unambiguously.",
                    str(raw),
                )
            )
        else:
            checks.append(
                _check_result(
                    f"{label} format",
                    "Passed",
                    f"The date could be parsed as {parsed.isoformat()}.",
                    str(raw),
                )
            )
    return checks


def _check_certificate(fields):
    return [
        _check_date_order(
            fields,
            ("completion_date", "date_of_completion"),
            ("issue_date", "issued_date"),
            "Completion date vs. issue date",
        )
    ]


def _check_report(fields):
    return [
        _check_result(
            "Findings vs. conclusion",
            "Unable to verify",
            "The report's key findings and conclusion require semantic review. This version does not automatically assert that they are consistent.",
            str(_find_value(fields, "key_findings", "findings"))
            + " | "
            + str(_find_value(fields, "conclusion", "conclusions")),
        )
    ]


def _check_category_fallback(fields):
    return [
        _check_result(
            "Category-specific consistency",
            "Unable to verify",
            "No deterministic validation rules are configured for this category yet.",
        )
    ]


def check_document_consistency(structured_data, category):
    """
    Shared entry point: dispatch to validation rules based on document category.
    Returns a list of checks with Passed, Warning, or Unable to verify status.
    """
    fields = (
        structured_data.get("fields", {}) if isinstance(structured_data, dict) else {}
    )
    if not isinstance(fields, dict):
        fields = {}
    if category == "Invoice":
        checks = _check_invoice(fields)
    elif category == "Receipt":
        checks = _check_receipt(fields)
    elif category == "Statement":
        checks = _check_statement(fields)
    elif category == "Certificate":
        checks = _check_certificate(fields)
    elif category == "Resume":
        checks = _check_resume(fields)
    elif category in ("Application Form", "Scanned Form"):
        checks = _check_application_or_form(fields)
    elif category == "Report":
        checks = _check_report(fields)
    else:
        checks = _check_category_fallback(fields)
    return checks
