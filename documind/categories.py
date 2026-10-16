"""Persistent, case-insensitive registry of user-defined document types."""

import sqlite3
import json
import re
from contextlib import contextmanager
from pathlib import Path

from .models import ALLOWED_CATEGORIES

DATABASE = Path(__file__).resolve().parent.parent / "data" / "categories.sqlite3"


@contextmanager
def _connect():
    DATABASE.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DATABASE)
    connection.execute(
        "CREATE TABLE IF NOT EXISTS categories (key TEXT PRIMARY KEY, name TEXT NOT NULL)"
    )
    connection.execute(
        "CREATE TABLE IF NOT EXISTS required_templates (category TEXT PRIMARY KEY, fields_json TEXT NOT NULL)"
    )
    connection.execute(
        "CREATE TABLE IF NOT EXISTS category_profiles (category TEXT PRIMARY KEY, description TEXT NOT NULL)"
    )
    try:
        with connection:
            yield connection
    finally:
        connection.close()


def get_categories():
    with _connect() as connection:
        custom = [
            row[0]
            for row in connection.execute("SELECT name FROM categories ORDER BY name")
        ]
    return sorted(set(ALLOWED_CATEGORIES) | set(custom), key=str.casefold)


def add_category(name):
    name = " ".join(name.split())
    if not name or len(name) > 80 or any(not c.isprintable() for c in name):
        raise ValueError("Enter a document type between 1 and 80 printable characters.")
    for existing in get_categories():
        if existing.casefold() == name.casefold():
            return existing
    with _connect() as connection:
        connection.execute(
            "INSERT OR IGNORE INTO categories (key, name) VALUES (?, ?)",
            (name.casefold(), name),
        )
        return connection.execute(
            "SELECT name FROM categories WHERE key = ?", (name.casefold(),)
        ).fetchone()[0]


def get_required_template(category):
    with _connect() as connection:
        row = connection.execute(
            "SELECT fields_json FROM required_templates WHERE category = ?",
            (category.casefold(),),
        ).fetchone()
    return json.loads(row[0]) if row else None


def save_required_template(category, fields):
    normalized = list(
        dict.fromkeys(
            name.strip().lower().replace(" ", "_") for name in fields if name.strip()
        )
    )
    if not normalized or any(
        not re.fullmatch(r"[a-z][a-z0-9_]{0,79}", name) for name in normalized
    ):
        raise ValueError(
            "Use field names such as name, address, date, or signature_status."
        )
    with _connect() as connection:
        connection.execute(
            "INSERT INTO required_templates (category, fields_json) VALUES (?, ?) ON CONFLICT(category) DO UPDATE SET fields_json=excluded.fields_json",
            (category.casefold(), json.dumps(normalized)),
        )
    return normalized


def save_category_profile(name, description=""):
    canonical = add_category(name)
    description = " ".join(description.split())[:400]
    if description:
        with _connect() as connection:
            connection.execute(
                "INSERT INTO category_profiles (category, description) VALUES (?, ?) ON CONFLICT(category) DO UPDATE SET description=excluded.description",
                (canonical, description),
            )
    return canonical


def get_category_profiles():
    with _connect() as connection:
        return dict(
            connection.execute("SELECT category, description FROM category_profiles")
        )
