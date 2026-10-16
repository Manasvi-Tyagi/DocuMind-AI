"""Bounded page batching and transparent lexical retrieval for long documents."""

import re
from .models import StructuredDocument


def page_batches(pages, size=3):
    if size < 1:
        raise ValueError("Batch size must be positive.")
    return [pages[i : i + size] for i in range(0, len(pages), size)]


def retrieve_pages(pages, question, limit=3):
    """Rank local text, not hidden model claims; explicitly requested pages win."""
    requested = {int(n) for n in re.findall(r"\bpage\s+(\d+)\b", question, re.I)}
    ignored = set(
        "what which where is are the a an in of to and document please".split()
    )
    terms = set(re.findall(r"\w+", question.casefold())) - ignored
    ranked = sorted(
        enumerate(pages),
        key=lambda item: (
            item[1].number not in requested,
            -len(terms & set(re.findall(r"\w+", item[1].text.casefold()))),
            item[0],
        ),
    )
    return sorted([page for _, page in ranked[:limit]], key=lambda p: p.number)


def merge_documents(documents, batches, category):
    fields, summaries = {}, []
    for document, pages in zip(documents, batches):
        scope = ", ".join(str(p.number) for p in pages)
        summaries.append(f"Pages {scope}: {document.summary}")
        for name, value in document.fields.items():
            key = (
                name
                if name not in fields
                else f'{name}__pages_{"_".join(str(p.number) for p in pages)}'
            )
            fields[key] = value
    return StructuredDocument(
        category=category,
        summary="\n\n".join(summaries),
        fields=fields,
        batches=documents,
    )
