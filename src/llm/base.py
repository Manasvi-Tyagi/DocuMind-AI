from abc import ABC, abstractmethod
from dataclasses import dataclass

from PIL import Image

# A prompt is an ordered list of text and image parts.
Part = str | Image.Image


@dataclass
class LLMResponse:
    text: str
    latency_s: float


class LLMBackend(ABC):
    name: str = "base"

    @abstractmethod
    def generate(self, parts: list[Part], json_mode: bool = False) -> LLMResponse:
        """Generate a response. ``json_mode`` asks for a JSON-only reply."""
