from functools import lru_cache

from src.config import settings
from src.llm.base import LLMBackend

BACKENDS = ("gemma", "gemini")


@lru_cache(maxsize=2)
def get_backend(name: str | None = None) -> LLMBackend:
    """Create (once) and return a backend: 'gemma' (default, local) or 'gemini' (API)."""

    name = (name or settings.llm_backend).lower()
    if name == "gemma":
        from src.llm.gemma_local import GemmaBackend

        return GemmaBackend(settings.model_id, settings.max_new_tokens)
    if name == "gemini":
        from src.llm.gemini_api import GeminiBackend

        return GeminiBackend(settings.gemini_api_key, settings.gemini_model, settings.max_new_tokens)
    raise ValueError(f"Unknown LLM_BACKEND '{name}'. Use 'gemma' or 'gemini'.")


def model_name(name: str | None = None) -> str:
    name = (name or settings.llm_backend).lower()
    return settings.gemini_model if name == "gemini" else settings.model_id
