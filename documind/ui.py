"""Shared presentation styles and document field rendering."""

import streamlit as st


def apply_styles():
    st.markdown(
        """<style>
        html, body, .stApp, [data-testid="stApp"],
        [data-testid="stAppViewContainer"], [data-testid="stMain"],
        [data-testid="stMainBlockContainer"] {
            background: #f5f7fa !important;
            color: #172538 !important;
            --background-color: #f5f7fa;
            --secondary-background-color: #ffffff;
            --text-color: #172538;
            color-scheme: light;
        }
        [data-testid="stSidebar"], [data-testid="stSidebarContent"],
        [data-testid="stSidebarUserContent"], [data-testid="stHeader"] {
            background: #e8edf3 !important; color: #172538 !important;
        }
        .stApp h1, .stApp h2, .stApp h3, .stApp h4,
        .stApp p, .stApp label, .stApp small, .stApp summary,
        .stApp [data-testid="stMarkdownContainer"],
        .stApp [data-testid="stCaptionContainer"],
        .stApp [data-testid="stMetricLabel"],
        .stApp [data-testid="stMetricValue"],
        .stApp [data-baseweb="tab"] { color: #172538 !important; }
        .block-container { max-width: 1280px; padding-top: 2rem; padding-bottom: 3rem; }
        h1, h2, h3 { letter-spacing: -.025em; }
        .stApp input, .stApp textarea, .stApp [data-baseweb="input"],
        .stApp [data-baseweb="textarea"], .stApp [data-baseweb="base-input"] {
            background: #ffffff !important; color: #172538 !important;
            -webkit-text-fill-color: #172538 !important;
        }
        .stApp input::placeholder, .stApp textarea::placeholder {
            color: #526176 !important; -webkit-text-fill-color: #526176 !important; opacity: 1;
        }
        .stApp button { background: #e0eaf5 !important; color: #172538 !important; border-color: #a6b8cd !important; border-radius: 8px; }
        .stApp button p { color: #172538 !important; }
        .stApp button[kind="primary"], .stApp button[kind="primaryFormSubmit"] { background: #b9e5ce !important; color: #172538 !important; }
        .stApp button:disabled { opacity: .7; }
        .stApp [data-testid="stMetric"],
        .stApp [data-testid="stFileUploaderDropzone"],
        .stApp [data-testid="stExpander"], .stApp [data-testid="stChatMessage"],
        .stApp [data-testid="stAlert"], .stApp [data-testid="stStatusWidget"],
        .stApp [data-testid="stJson"] {
            background: #ffffff !important; color: #172538 !important; border-color: #c3cfdd !important;
        }
        [data-testid="stMetric"] { padding: 20px; border: 1px solid #c3cfdd; border-radius: 12px; }
        [data-testid="stMetricValue"] { font-size: 1.4rem; }
        .stApp [data-baseweb="tab-list"] { background: #e8edf3 !important; gap: 12px; }
        .stApp [data-baseweb="tab"][aria-selected="true"] { background: #d6e7e0 !important; }
        .eyebrow { display: inline-block; color: #24533e !important; background: #d6e7e0; padding: 5px 10px; border-radius: 6px; text-transform: uppercase; letter-spacing: .13em; font-size: .72rem; font-weight: 700; margin-bottom: 18px; }
        .hero { padding: 28px 32px; margin-bottom: 26px; border: 1px solid #c3cfdd; border-radius: 18px; background: #e6eef5 !important; }
        .hero h1 { color: #172538 !important; font-size: clamp(1.8rem, 3.5vw, 2.65rem); line-height: 1.2; margin-bottom: .5rem; }
        .hero p { color: #304359 !important; font-size: 1rem; max-width: 650px; margin-bottom: 0; }
        .empty { padding: 48px 24px; text-align: center; background: #ffffff !important; color: #172538 !important; border: 1px solid #c3cfdd; border-radius: 16px; margin-top: 24px; }
        .empty .symbol { display: inline-flex; align-items: center; justify-content: center; width: 64px; height: 64px; border-radius: 16px; background: #d6e7e0; font-size: 2rem; margin-bottom: 18px; color: #24533e; }
        .empty h3 { font-size: 1.4rem; color: #172538 !important; }
        .empty p { color: #304359 !important; max-width: 440px; margin: auto; }
        @media (max-width: 640px) { .block-container { padding-top: 1rem; } .hero { padding: 22px 18px; } }
        </style>""",
        unsafe_allow_html=True,
    )


def render_fields(fields, pages=None):
    for name, field in fields.items():
        with st.expander(name.replace("_", " ").title()):
            wrapped = isinstance(field, dict) and "value" in field
            value = field.get("value") if wrapped else field
            if isinstance(value, (dict, list)):
                st.json(value)
            else:
                st.write(value if value is not None else "Not found")
            if wrapped:
                if field.get("page"):
                    st.caption(
                        f"Source: page {field['page']} · {field.get('region') or 'Region not specified'}"
                    )
                st.caption(f"Confidence: {field.get('confidence', 'LOW')}")
                if field.get("evidence"):
                    st.write("**Source evidence**")
                    st.write(field["evidence"])
                    if pages and st.button(
                        "Locate quote on source", key=f"evidence_{name}"
                    ):
                        from .evidence import locate_quote, highlight_matches

                        with st.spinner("Locating source evidence…"):
                            result = locate_quote(
                                pages, field["evidence"], field.get("page")
                            )
                        if result["status"] == "located":
                            st.success(
                                "Quote located. Verify the field value against the highlighted text."
                            )
                            for page in pages:
                                if any(
                                    match["page"] == page.number
                                    for match in result["matches"]
                                ):
                                    st.image(
                                        highlight_matches(page, result["matches"]),
                                        caption=f"Page {page.number} · matched quote",
                                        width="stretch",
                                    )
                        else:
                            st.warning(result["reason"])
