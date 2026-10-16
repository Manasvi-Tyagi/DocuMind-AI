"""Apply the same batching scope to text-only and vision document analysis."""

from .ai import classify_document, extract_structured_data
from .batching import page_batches, merge_documents
from .documents import pages_text


def classify_pages(text, pages, vision=True):
    if len(pages) > 3 and not vision:
        categories = [
            classify_document(pages_text(batch)) for batch in page_batches(pages)
        ]
        return categories[0] if len(set(categories)) == 1 else "Other / Uncertain"
    return classify_document(text, pages=pages if vision else None)


def extract_pages(text, category, pages, vision=True):
    if len(pages) > 3 and not vision:
        batches = page_batches(pages)
        documents = [
            extract_structured_data(pages_text(batch), category) for batch in batches
        ]
        return merge_documents(documents, batches, category)
    return extract_structured_data(text, category, pages=pages if vision else None)


def review_pages(text, pages, vision=True):
    from .semantic import review_semantics

    if len(pages) > 3 and not vision:
        return [
            issue
            for batch in page_batches(pages)
            for issue in review_semantics(pages_text(batch))
        ]
    return review_semantics(text, pages=pages if vision else None)
