"""Central configuration loaded from environment variables / .env."""

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


def _int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


@dataclass(frozen=True)
class Settings:
    # "gemma" (local Gemma 3 4B, default) or "gemini" (API — for 8 GB laptops / cloud).
    llm_backend: str = os.getenv("LLM_BACKEND", "gemma").strip().lower()

    # Gemma 3 4B Instruct (multimodal: image + text)
    model_id: str = os.getenv("MODEL_ID", "google/gemma-3-4b-it")
    max_new_tokens: int = _int("MAX_NEW_TOKENS", 512)

    # Gemini API
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    gemini_model: str = os.getenv("GEMINI_MODEL", "")

    # Document preprocessing
    max_pdf_pages: int = _int("MAX_PDF_PAGES", 20)
    pdf_dpi: int = _int("PDF_DPI", 200)
    target_long_side: int = _int("TARGET_LONG_SIDE", 2000)
    max_pages_per_query: int = _int("MAX_PAGES_PER_QUERY", 5)

    # OCR
    tesseract_lang: str = os.getenv("TESSERACT_LANG", "eng")
    tesseract_cmd: str = os.getenv("TESSERACT_CMD", "")

    # Optional Google Cloud Storage bucket for uploaded documents
    gcs_bucket: str = os.getenv("GCS_BUCKET", "")


settings = Settings()
