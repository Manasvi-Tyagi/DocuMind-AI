"""Native Ollama chat adapter for local text, images, and JSON output."""

from types import SimpleNamespace
import httpx


def chat(settings, **kwargs):
    messages = []
    for message in kwargs["messages"]:
        content = message["content"]
        converted = {"role": message["role"], "content": content}
        if isinstance(content, list):
            text, images = [], []
            for part in content:
                if part["type"] == "text":
                    text.append(part["text"])
                elif part["type"] == "image_url":
                    url = part["image_url"]["url"]
                    if not url.startswith("data:image/") or ";base64," not in url:
                        raise ValueError(
                            "Local vision expects embedded page images, not remote URLs."
                        )
                    images.append(url.split(";base64,", 1)[1])
            converted["content"] = "\n\n".join(text)
            if images:
                converted["images"] = images
        messages.append(converted)
    payload = {
        "model": settings.ollama_model,
        "messages": messages,
        "stream": False,
        "keep_alive": "10m",
        "options": {
            "temperature": kwargs.get("temperature", 0),
            "num_predict": kwargs.get("max_completion_tokens", 2048),
            "num_ctx": settings.ollama_num_ctx,
        },
    }
    if kwargs.get("response_format"):
        payload["format"] = "json"
    try:
        with httpx.Client(
            timeout=httpx.Timeout(settings.ollama_timeout, connect=10), trust_env=False
        ) as client:
            response = client.post(
                settings.ollama_base_url.rstrip("/") + "/api/chat", json=payload
            )
    except httpx.ConnectError as error:
        raise RuntimeError(
            "Cannot connect to Ollama. Start it with `ollama serve` and check OLLAMA_BASE_URL."
        ) from error
    except httpx.TimeoutException as error:
        raise RuntimeError(
            "Ollama inference timed out. Try one page or increase OLLAMA_TIMEOUT in .env."
        ) from error
    if response.status_code == 404:
        raise RuntimeError(
            f"Ollama model not found. Run `ollama pull {settings.ollama_model}`."
        )
    if response.is_error:
        raise RuntimeError(
            f"Ollama request failed ({response.status_code}): {response.text[:500]}"
        )
    data = response.json()
    if data.get("error"):
        raise RuntimeError(f"Ollama: {data['error']}")
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=data.get("message", {}).get("content", "")
                ),
                finish_reason=(
                    "length" if data.get("done_reason") == "length" else "stop"
                ),
            )
        ]
    )
