"""Gemma 3 4B Instruct (google/gemma-3-4b-it) via Hugging Face Transformers."""

import time

from src.llm.base import LLMBackend, LLMResponse, Part


class GemmaBackend(LLMBackend):
    name = "gemma"

    def __init__(self, model_id: str, max_new_tokens: int):
        import torch
        from transformers import AutoProcessor, Gemma3ForConditionalGeneration

        self._torch = torch
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        # Gemma 3 is trained in bfloat16; float32 on CPU for compatibility.
        dtype = torch.bfloat16 if self.device.type in ("cuda", "mps") else torch.float32

        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = Gemma3ForConditionalGeneration.from_pretrained(
            model_id,
            dtype=dtype,
            low_cpu_mem_usage=True,
        )
        self.model.to(self.device)
        self.model.eval()

    def generate(self, parts: list[Part], json_mode: bool = False) -> LLMResponse:
        torch = self._torch

        content = [
            {"type": "text", "text": p} if isinstance(p, str) else {"type": "image", "image": p}
            for p in parts
        ]
        if json_mode:
            content.append({"type": "text", "text": "Respond with valid JSON only."})

        inputs = self.processor.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
        )
        inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(self.model.dtype)

        start = time.perf_counter()
        with torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        latency = time.perf_counter() - start

        input_length = inputs["input_ids"].shape[1]
        text = self.processor.batch_decode(output[:, input_length:], skip_special_tokens=True)[0]
        return LLMResponse(text=text.strip(), latency_s=round(latency, 3))
