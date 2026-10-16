"""Read application settings from the project's private .env file."""

import os
import shutil
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = "openai/gpt-oss-120b"
_provider_override = ContextVar("documind_provider", default=None)


@dataclass(frozen=True)
class Settings:
    groq_api_key: str
    groq_model: str
    tesseract_cmd: str | None
    groq_vision_model: str = "qwen/qwen3.8-27b"
    ai_provider: str = "ollama"
    ollama_model: str = "gemma3:4b"
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_num_ctx: int = 16384
    ollama_timeout: float = 600


def backend_ready(settings):
    return getattr(settings, "ai_provider", "groq") == "ollama" or bool(
        settings.groq_api_key
    )


def model_name(settings, vision=False):
    if getattr(settings, "ai_provider", "groq") == "ollama":
        return settings.ollama_model
    return settings.groq_vision_model if vision else settings.groq_model


@contextmanager
def use_provider(provider):
    """Route one operation without changing other sessions or the environment."""
    if provider not in {"ollama", "groq"}:
        raise ValueError("AI provider must be ollama or groq.")
    token = _provider_override.set(provider)
    try:
        yield
    finally:
        _provider_override.reset(token)


def get_settings(provider=None) -> Settings:
    """Reload settings so Streamlit reruns pick up local configuration changes."""
    load_dotenv(PROJECT_ROOT / ".env", override=True)
    provider = (
        provider
        or _provider_override.get()
        or os.getenv("AI_PROVIDER", "ollama").strip().lower()
    )
    if provider not in {"ollama", "groq"}:
        raise ValueError("AI_PROVIDER must be ollama or groq.")
    return Settings(
        ai_provider=provider,
        ollama_model=os.getenv("OLLAMA_MODEL", "gemma3:4b").strip() or "gemma3:4b",
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").strip(),
        ollama_num_ctx=int(os.getenv("OLLAMA_NUM_CTX", "16384")),
        ollama_timeout=float(os.getenv("OLLAMA_TIMEOUT", "600")),
        groq_api_key=os.getenv("GROQ_API_KEY", "").strip(),
        groq_model=os.getenv("GROQ_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL,
        groq_vision_model=os.getenv("GROQ_VISION_MODEL", "qwen/qwen3.8-27b").strip()
        or "qwen/qwen3.8-27b",
        tesseract_cmd=os.getenv("TESSERACT_CMD") or shutil.which("tesseract"),
    )
