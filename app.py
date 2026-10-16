"""Streamlit workspace for document analysis."""

import hashlib
import importlib
import json
import sqlite3
from time import perf_counter

import streamlit as st

from documind import ai as ai_service, config as app_config, prompts as ai_prompts

# Refresh a pre-upgrade Streamlit process once so imported services agree.
if getattr(ai_service, "SERVICE_REVISION", None) != 5:
    if not hasattr(app_config, "use_provider"):
        importlib.reload(app_config)
    for module_name in (
        "models",
        "categories",
        "prompts",
        "documents",
        "batching",
        "evidence",
        "ollama_backend",
        "ai",
        "analysis",
        "validation",
        "semantic",
        "ui",
    ):
        importlib.reload(importlib.import_module(f"documind.{module_name}"))

from documind.categories import (
    get_categories,
    add_category,
    get_required_template,
    save_required_template,
    save_category_profile,
)
from documind.ai import answer_question
from documind.config import get_settings, backend_ready, use_provider
from documind.documents import page_count, prepare_document, pages_text
from documind.ui import apply_styles, render_fields
from documind.analysis import classify_pages, extract_pages, review_pages
from documind.validation import check_document_consistency, detect_missing_fields

st.set_page_config(
    page_title="DocuMind · Document workspace", page_icon="📄", layout="wide"
)
apply_styles()
settings = get_settings()

with st.sidebar:
    st.markdown("## ◈ DocuMind")
    st.caption("DOCUMENT ANALYSIS STUDIO")
    provider = st.radio(
        "AI backend",
        ["ollama", "groq"],
        index=0 if settings.ai_provider == "ollama" else 1,
        format_func=lambda value: (
            "Ollama · Gemma 3 4B (local)" if value == "ollama" else "Groq (cloud)"
        ),
        key="selected_ai_provider",
        help="Ollama uses your local server. Groq sends document content to its API and uses your account quota.",
    )
    settings = get_settings(provider=provider)
    ai_enabled = backend_ready(settings)
    if st.session_state.get("last_ai_provider", provider) != provider:
        st.session_state.pop("structured_data", None)
        st.session_state.pop("messages", None)
        st.session_state.pop("semantic_review", None)
    st.session_state["last_ai_provider"] = provider

    with st.expander("Recognized document types"):
        category_list = st.empty()
        category_list.write(", ".join(get_categories()))
    st.divider()
    st.markdown("### Start with a document")
    st.caption("Upload a PDF, scan, or photo to get started.")
    uploaded_file = st.file_uploader(
        "Choose your document", type=["pdf", "png", "jpg", "jpeg"]
    )
    mode = st.radio(
        "Analysis mode",
        ["Vision + OCR", "Text only"],
        help="Vision reads selected page images as well as local OCR. Text only sends extracted text.",
    )
    selected_pages = []
    if uploaded_file is not None:
        try:
            total_pages = page_count(uploaded_file)
            analyze_all = (
                st.checkbox(
                    "Analyze entire PDF in batches",
                    help="Up to 30 pages. Each AI request receives at most 3 images; longer files take more time.",
                )
                if total_pages > 3
                else False
            )
            if analyze_all:
                selected_pages = (
                    list(range(1, total_pages + 1)) if total_pages <= 30 else []
                )
                if total_pages > 30:
                    st.warning(
                        "Whole-document mode supports up to 30 pages. Split larger files first."
                    )
            else:
                selected_pages = st.multiselect(
                    "Pages to analyze (up to 3)",
                    range(1, total_pages + 1),
                    default=list(range(1, min(total_pages, 3) + 1)),
                    max_selections=3,
                )
            st.caption(
                f"{total_pages} total pages. Analysis covers only selected pages."
            )
        except Exception as error:
            st.error(f"Cannot open this document: {error}")
        signature = (
            hashlib.sha256(uploaded_file.getvalue()).hexdigest(),
            mode,
            tuple(selected_pages),
        )
        if st.session_state.get("file_signature") != signature:
            for key in (
                "document_text",
                "document_category",
                "category_confirmed",
                "manual_category",
                "document_name",
                "uploaded_file_size",
                "structured_data",
                "messages",
                "semantic_review",
                "document_pages",
                "processing_seconds",
            ):
                st.session_state.pop(key, None)
            st.session_state["file_signature"] = signature
        st.caption(
            f"{uploaded_file.size / 1024:,.1f} KB · {uploaded_file.name.rsplit('.', 1)[-1].upper()}"
        )
    process = st.button(
        "Process document",
        type="primary",
        width="stretch",
        disabled=uploaded_file is None or not selected_pages,
    )
    st.divider()
    st.markdown("### How it works")
    st.markdown(
        "**01**  Upload your document\n\n**02**  Review extracted insights\n\n**03**  Ask questions with context"
    )
    st.divider()
    if ai_enabled:
        st.caption(
            f"● Ollama · {settings.ollama_model}"
            if settings.ai_provider == "ollama"
            else "● Groq is configured"
        )
    else:
        st.warning("Add GROQ_API_KEY to .env to enable AI analysis.")
    st.caption(
        "Images and text are processed by your configured Ollama server. No Groq requests are made."
        if settings.ai_provider == "ollama"
        else "Selected page images and text are sent to Groq in vision mode; text only sends extracted text."
    )


def run_ai(operation, *args, **kwargs):
    with use_provider(settings.ai_provider):
        return operation(*args, **kwargs)


st.markdown(
    '<div class="hero"><div class="eyebrow">DocuMind / Analysis workspace</div>'
    "<h1>A clearer view of every document.</h1>"
    "<p>Extract the essentials. Check the details. Find answers in your files.</p></div>",
    unsafe_allow_html=True,
)

if process and uploaded_file is not None:
    try:
        with st.status("Reading your document…", expanded=True) as status:
            started = perf_counter()
            pages = prepare_document(uploaded_file, selected_pages, max_pages=30)
            text = pages_text(pages)
            for page in pages:
                if page.warning:
                    st.warning(page.warning)
            if not any(page.text.strip() for page in pages) and mode == "Text only":
                status.update(label="No readable text found", state="error")
                st.warning("Try a clearer scan or a document with selectable text.")
            else:
                st.session_state.update(
                    document_text=text,
                    document_pages=pages,
                    document_name=uploaded_file.name,
                    uploaded_file_size=uploaded_file.size,
                )
                st.session_state.pop("structured_data", None)
                st.session_state.pop("document_category", None)
                st.session_state.pop("category_confirmed", None)
                st.session_state.pop("manual_category", None)
                st.session_state.pop("messages", None)
                st.session_state.pop("semantic_review", None)
                st.write("Text extracted. Identifying document type…")
                if ai_enabled:
                    try:
                        st.session_state["document_category"] = run_ai(
                            classify_pages,
                            text,
                            pages,
                            vision=mode == "Vision + OCR",
                        )
                    except Exception as error:
                        st.warning(
                            "Automatic classification failed. Choose the document type below to continue."
                        )
                        st.caption(f"{type(error).__name__}: {error}")
                    status.update(
                        label="Your document is ready", state="complete", expanded=False
                    )
                else:
                    status.update(
                        label="Text ready · configure the AI backend to enable analysis",
                        state="complete",
                        expanded=False,
                    )
                st.session_state["processing_seconds"] = perf_counter() - started
    except Exception as error:
        st.error(
            "We couldn't finish processing this document. You can retry from the sidebar."
        )
        with st.expander("Error details"):
            st.write(f"{type(error).__name__}: {error}")

category_list.write(", ".join(get_categories()))

if not st.session_state.get("document_text"):
    a, b, c = st.columns(3)
    a.metric("01 · Understand", "Identify document type")
    b.metric("02 · Organize", "Extract key details")
    c.metric("03 · Explore", "Ask your document")
    st.markdown(
        '<div class="empty"><div class="symbol">▤</div><h3>Ready when your document is.</h3><p>Choose a document in the sidebar, then select Process document. Your results will appear in this workspace.</p></div>',
        unsafe_allow_html=True,
    )
    st.stop()

text = st.session_state["document_text"]
category = st.session_state.get("document_category")
if (
    category not in get_categories() or category == "Other / Uncertain"
) and not st.session_state.get("category_confirmed"):
    st.info(
        "We couldn't identify a specific document type. What kind of document did you upload?"
    )
    with st.form("choose_document_type"):
        manual_category = st.selectbox(
            "Document type",
            [name for name in get_categories() if name != "Other / Uncertain"],
            index=None,
            placeholder="Choose a document type",
            key="manual_category",
        )
        new_category = st.text_input(
            "Or add a new document type",
            placeholder="e.g. Purchase Order, Medical Record, Lease Agreement",
            max_chars=80,
        )
        category_description = st.text_input(
            "What identifies this type? (optional)",
            placeholder="e.g. purchase order number, supplier, delivery date",
        )
        st.caption(
            "New types and their descriptions are saved locally and reused for automatic classification."
        )
        confirm_category = st.form_submit_button("Save and use document type")
    if confirm_category:
        if new_category.strip() or manual_category:
            try:
                chosen = (
                    save_category_profile(new_category, category_description)
                    if new_category.strip()
                    else manual_category
                )
                st.session_state["document_category"] = chosen
                st.session_state["category_confirmed"] = True
                st.session_state.pop("structured_data", None)
                st.rerun()
            except (ValueError, OSError, sqlite3.Error) as error:
                st.error(f"Could not save document type: {error}")
        else:
            st.warning("Choose an existing document type or enter a new one.")

pages = st.session_state.get("document_pages", [])
vision_pages = pages if mode == "Vision + OCR" else None
st.markdown("### Your document")
st.write(st.session_state.get("document_name", "Uploaded document"))
a, b, c = st.columns(3)
a.metric("Document type", category or "Not classified")
b.metric("Words extracted", f"{len(text.split()):,}")
c.metric("File size", f"{st.session_state.get('uploaded_file_size', 0) / 1024:,.1f} KB")
st.caption(
    f"{mode} · Selected pages: {', '.join(str(p.number) for p in pages) or 'None'} · Processing: {st.session_state.get('processing_seconds', 0):.1f}s"
)
st.write("")
overview, checks, source, chat = st.tabs(
    ["Overview", "Document checks", "Source text", "Ask a question"]
)

with overview:
    st.markdown("### The details, without the digging")
    st.caption(
        "Extract a summary and structured fields, with confidence levels and source evidence."
    )
    if not category:
        if st.button("Identify document type", disabled=not ai_enabled):
            try:
                with st.spinner("Identifying document type…"):
                    st.session_state["document_category"] = run_ai(
                        classify_pages, text, pages, vision=mode == "Vision + OCR"
                    )
                st.rerun()
            except Exception as error:
                st.error(f"Classification failed: {error}")
    if st.button(
        "Extract insights",
        type="primary",
        disabled=not category or not ai_enabled,
    ):
        try:
            with st.spinner("Finding the important details…"):
                st.session_state["structured_data"] = run_ai(
                    extract_pages, text, category, pages, vision=mode == "Vision + OCR"
                ).model_dump()
        except Exception as error:
            st.error("Insights could not be extracted. Please try again.")
            with st.expander("Error details"):
                st.write(f"{type(error).__name__}: {error}")
    data = st.session_state.get("structured_data")
    if data:
        with st.container(border=True):
            st.markdown("#### At a glance")
            st.write(data.get("summary", "No summary returned."))
        st.markdown("#### Extracted information")
        render_fields(data.get("fields", {}), pages=pages)
        st.download_button(
            "Download insights · JSON",
            json.dumps(data, indent=2, ensure_ascii=False),
            file_name="document-insights.json",
            mime="application/json",
        )
    else:
        st.info(
            "Select Extract insights to create a summary and organize your document's key fields."
        )

with checks:
    data = st.session_state.get("structured_data")
    if not data or not category:
        st.info(
            "Extract insights in Overview to unlock missing-field and consistency checks."
        )
    else:
        if data.get("batches"):
            batch_index = st.selectbox(
                "Validation batch",
                range(len(data["batches"])),
                format_func=lambda i: f"Batch {i + 1}",
            )
            st.caption(
                "Each batch is validated separately. Values from separate batches are not combined for arithmetic checks."
            )
            data = data["batches"][batch_index]
        st.markdown("### A closer look at completeness")
        st.caption(
            "Not found means a field wasn't extracted; it may still exist in the original document."
        )
        required = st.text_input(
            "Custom required fields (optional)",
            value=", ".join(get_required_template(category) or []),
            key=f"required_fields_{category}",
            placeholder="name, address, date, signature_status, identification_number",
            help="Comma-separated extraction field names. Leave blank to use the category template.",
        )
        required_fields = [
            name.strip().lower().replace(" ", "_")
            for name in required.split(",")
            if name.strip()
        ] or None
        if st.button(
            "Save required fields for this category", disabled=not required_fields
        ):
            try:
                save_required_template(category, required_fields)
                st.success(f"Required-field template saved for {category}.")
            except (ValueError, sqlite3.Error) as error:
                st.error(str(error))
        fields = detect_missing_fields(data, category, required_fields=required_fields)
        found = sum(item["status"] == "Found" for item in fields)
        a, b = st.columns(2)
        a.metric("Fields found", found)
        b.metric("Fields not found", len(fields) - found)
        if fields:
            st.dataframe(
                [{"Field": item["field"], "Status": item["status"]} for item in fields],
                hide_index=True,
                width="stretch",
            )
        else:
            st.info("No completeness template is available for this category.")
        st.markdown("### Do the values agree?")
        st.caption(
            "Warnings flag possible inconsistencies and require review against the original."
        )
        results = check_document_consistency(data, category)
        cols = st.columns(3)
        for col, label in zip(cols, ("Passed", "Warning", "Unable to verify")):
            col.metric(label, sum(row["status"] == label for row in results))
        for result in results:
            with st.expander(f"{result['status']} · {result['rule']}"):
                st.write(result["details"])
                st.caption(f"Evidence / values checked: {result['evidence']}")

    st.markdown("### Semantic review")
    st.caption(
        "AI flags are tentative. Matching quotes confirm text presence, not that a contradiction is proven. In batched mode, review covers contradictions within each batch."
    )
    if st.button("Review semantic consistency", disabled=not ai_enabled):
        try:
            with st.spinner("Reviewing statements…"):
                st.session_state["semantic_review"] = run_ai(
                    review_pages, text, pages, vision=mode == "Vision + OCR"
                )
        except Exception as error:
            st.error(f"Semantic review failed: {error}")
    if "semantic_review" in st.session_state:
        issues = st.session_state["semantic_review"]
        if not issues:
            st.info(
                "No supported contradictions were reported. This is not proof that the document is consistent."
            )
        for issue in issues:
            with st.expander(issue["description"]):
                st.write(issue["first_quote"])
                st.write(issue["second_quote"])
                if issue["quotes_present_in_text"]:
                    st.caption(
                        "Both quotes were located in the extracted text. Review their meaning in context."
                    )
                else:
                    st.warning(
                        "One or both quotes were not found in extracted text. Treat this flag as unverified."
                    )

with source:
    st.markdown("### Straight from the source")
    st.caption("Review the text extracted from your PDF or image.")
    for page in pages:
        with st.expander(
            f"Page {page.number} · {page.method}", expanded=len(pages) == 1
        ):
            st.image(page.image, caption=f"Source page {page.number}", width="stretch")
    st.text_area("Extracted document text", text, height=420)
    st.download_button(
        "Download text", text, file_name="document-text.txt", mime="text/plain"
    )

with chat:
    st.markdown("### Let your document answer")
    st.caption(
        "Answers cover selected pages only. Page references and regions are model-generated; verify them against Source text."
    )
    st.caption(
        "Try: “What are the key dates?” or “Summarize the most important details.”"
    )
    for message in st.session_state.get("messages", []):
        with st.chat_message(message["role"]):
            st.write(message["content"])
    with st.form("document_question", clear_on_submit=True):
        question = st.text_input(
            "Your question", placeholder="What would you like to know?"
        )
        ask = st.form_submit_button(
            "Ask document", type="primary", disabled=not ai_enabled
        )
    if len(pages) > 3:
        st.caption(
            "For long documents, Q&A retrieves up to 3 pages using local text matching. Mention a page number to prioritize it; image-only pages may be missed by text retrieval."
        )
    if ask:
        if not question.strip():
            st.warning("Enter a question to continue.")
        else:
            try:
                with st.spinner("Looking through your document…"):
                    from documind.batching import retrieve_pages

                    answer_pages = (
                        retrieve_pages(pages, question) if len(pages) > 3 else pages
                    )
                    answer_text = pages_text(answer_pages) if len(pages) > 3 else text
                    answer = run_ai(
                        answer_question,
                        answer_text,
                        question,
                        pages=answer_pages if mode == "Vision + OCR" else None,
                    )
                if answer and answer.strip():
                    st.session_state.setdefault("messages", []).extend(
                        [
                            {"role": "user", "content": question},
                            {"role": "assistant", "content": answer},
                        ]
                    )
                    st.rerun()
                else:
                    st.warning("No answer was returned. Please try again.")
            except Exception as error:
                st.error(f"The question could not be answered: {error}")
