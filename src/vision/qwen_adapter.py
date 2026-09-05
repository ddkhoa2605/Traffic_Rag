from __future__ import annotations

import json
import re

from .base import VisionResult, VisionVerifier


class QwenVisionAdapter(VisionVerifier):
    """Lazy optional adapter so the core package never imports torch/transformers."""

    def __init__(self, model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct"):
        self.model_name = model_name
        self._pipeline = None

    def _load(self):
        if self._pipeline is None:
            try:
                from transformers import pipeline
            except ImportError as exc:
                raise RuntimeError("Install the 'vision' dependency group to use QwenVisionAdapter") from exc
            self._pipeline = pipeline("image-text-to-text", model=self.model_name)
        return self._pipeline

    def inspect_page(self, image_path: str, task: str, *, context: dict | None = None) -> VisionResult:
        context = context or {}
        prompt = (
            f"{task}\nNative context: {json.dumps(context, ensure_ascii=False)}\n"
            "Return one JSON object with keys vision_result, confidence, status. "
            "status must be AGREEMENT, DISAGREEMENT, or INCONCLUSIVE. Do not rewrite source text."
        )
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "url": image_path},
                {"type": "text", "text": prompt},
            ],
        }]
        output = self._load()(messages, max_new_tokens=512, return_full_text=False)
        generated = output[0].get("generated_text", output[0]) if isinstance(output, list) else output
        if isinstance(generated, list):
            generated = generated[-1].get("content", generated[-1])
        if isinstance(generated, list):
            generated = "".join(item.get("text", "") for item in generated if isinstance(item, dict))
        match = re.search(r"\{.*\}", str(generated), re.DOTALL)
        if not match:
            return VisionResult(
                page=int(context.get("page", 0)), task=task,
                native_result=context.get("native_result"), vision_result=str(generated),
                confidence=0.0, status="INCONCLUSIVE",
            )
        payload = json.loads(match.group(0))
        return VisionResult(
            page=int(context.get("page", payload.get("page", 0))),
            task=task,
            native_result=context.get("native_result"),
            vision_result=payload.get("vision_result"),
            confidence=float(payload.get("confidence", 0)),
            status=payload.get("status", "INCONCLUSIVE"),
        )
