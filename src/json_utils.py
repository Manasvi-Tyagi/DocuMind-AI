"""Parse and validate JSON returned by an LLM."""

import json
import re
from typing import TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)


class LLMOutputError(ValueError):
    def __init__(self, message: str, raw: str):
        super().__init__(message)
        self.raw = raw


def extract_json(text: str):
    """Return the first JSON object/array in ``text`` (handles ```json fences)."""

    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    candidate = fenced.group(1) if fenced else text

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    starts = [i for i in (candidate.find("{"), candidate.find("[")) if i != -1]
    if not starts:
        raise ValueError("No JSON found in model output.")
    start = min(starts)
    decoder = json.JSONDecoder()
    obj, _ = decoder.raw_decode(candidate[start:])
    return obj


def parse_model(text: str, model: type[T], list_key: str | None = None) -> T:
    """Parse ``text`` into ``model``.

    If the LLM returns a bare list and ``list_key`` is given, it is wrapped
    as ``{list_key: [...]}``.
    """

    try:
        data = extract_json(text)
    except (ValueError, json.JSONDecodeError) as exc:
        raise LLMOutputError(f"Model did not return valid JSON: {exc}", text) from exc

    if isinstance(data, list) and list_key:
        data = {list_key: data}
    try:
        return model.model_validate(data)
    except ValidationError as exc:
        raise LLMOutputError(f"Model JSON did not match the expected schema: {exc}", text) from exc
