"""Gemini API backend — the brief's option for 8 GB laptops and the cloud deployment."""

import time

from src.llm.base import LLMBackend, LLMResponse, Part


class GeminiBackend(LLMBackend):
    name = "gemini"

    def __init__(self, api_key: str, model: str, max_output_tokens: int):
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set. Add it to your .env file.")
        if not model:
            raise RuntimeError("GEMINI_MODEL is not set. Add it to your .env file.")

        from google import genai

        self._client = genai.Client(api_key=api_key)
        self.model = model
        self.max_output_tokens = max_output_tokens

    def generate(self, parts: list[Part], json_mode: bool = False) -> LLMResponse:
        from google.genai import types

        config = types.GenerateContentConfig(
            temperature=0.0,  # deterministic answers reduce hallucination
            max_output_tokens=self.max_output_tokens,
            response_mime_type="application/json" if json_mode else "text/plain",
        )

        start = time.perf_counter()
        response = self._client.models.generate_content(
            model=self.model,
            contents=list(parts),
            config=config,
        )
        return LLMResponse(
            text=(response.text or "").strip(),
            latency_s=round(time.perf_counter() - start, 3),
        )
