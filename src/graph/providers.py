from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Protocol

from .models import ArticleExtraction
from .prompt import SYSTEM_PROMPT


@dataclass(frozen=True)
class ProviderResult:
    payload: ArticleExtraction
    raw: dict[str, Any]
    metadata: dict[str, Any]


class ProviderSchemaError(ValueError):
    """A provider returned a response, but it did not satisfy the frozen schema."""

    def __init__(self, message: str, *, raw: dict[str, Any], metadata: dict[str, Any]):
        super().__init__(message)
        self.raw = raw
        self.metadata = metadata


class ExtractionProvider(Protocol):
    name: str
    model: str

    def preflight(self) -> dict[str, Any]: ...
    def extract(self, prompt: str) -> ProviderResult: ...


class OpenAIExtractionProvider:
    name = "openai"

    def __init__(
        self,
        model: str = "gpt-5.6-luna",
        *,
        reasoning_effort: str = "low",
        api_key: str | None = None,
    ):
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")

    def _client(self):
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is required for paid extraction")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError('Install graph dependencies: pip install -e ".[graph]"') from exc
        return OpenAI(api_key=self.api_key)

    def preflight(self) -> dict[str, Any]:
        client = self._client()
        model = client.models.retrieve(self.model)
        raw = model.model_dump(mode="json")
        return {
            "provider": self.name,
            "requested_model": self.model,
            "resolved_model": raw.get("id"),
            "exists": raw.get("id") == self.model,
            "metadata": raw,
        }

    def extract(self, prompt: str) -> ProviderResult:
        schema = ArticleExtraction.model_json_schema()
        started = time.perf_counter()
        client = self._client()
        response = client.responses.create(
            model=self.model,
            reasoning={"effort": self.reasoning_effort},
            input=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "legal_article_extraction",
                    "strict": True,
                    "schema": schema,
                }
            },
        )
        raw = response.model_dump(mode="json")
        usage = raw.get("usage") or {}
        metadata = {
            "provider": self.name,
            "model": raw.get("model", self.model),
            "response_id": raw.get("id"),
            "input_tokens": usage.get("input_tokens", 0),
            "output_tokens": usage.get("output_tokens", 0),
            "elapsed_seconds": time.perf_counter() - started,
        }
        try:
            payload = ArticleExtraction.model_validate(json.loads(response.output_text))
        except Exception as exc:
            raise ProviderSchemaError(str(exc), raw=raw, metadata=metadata) from exc
        return ProviderResult(payload=payload, raw=raw, metadata=metadata)


class GeminiExtractionProvider:
    name = "gemini"

    def __init__(self, model: str = "gemini-3.1-flash-lite", *, api_key: str | None = None):
        self.model = model
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")

    def _client(self):
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is required for paid extraction")
        try:
            from google import genai
        except ImportError as exc:
            raise RuntimeError('Install graph dependencies: pip install -e ".[graph]"') from exc
        return genai.Client(api_key=self.api_key)

    def preflight(self) -> dict[str, Any]:
        client = self._client()
        model = client.models.get(model=self.model)
        raw = model.model_dump(mode="json") if hasattr(model, "model_dump") else {"name": getattr(model, "name", None)}
        resolved = (raw.get("name") or "").removeprefix("models/")
        return {
            "provider": self.name,
            "requested_model": self.model,
            "resolved_model": resolved,
            "exists": resolved == self.model,
            "metadata": raw,
        }

    def extract(self, prompt: str) -> ProviderResult:
        try:
            from google.genai import types
        except ImportError as exc:
            raise RuntimeError('Install graph dependencies: pip install -e ".[graph]"') from exc
        started = time.perf_counter()
        client = self._client()
        response = client.models.generate_content(
            model=self.model,
            contents=f"{SYSTEM_PROMPT}\n\n{prompt}",
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_json_schema=ArticleExtraction.model_json_schema(),
            ),
        )
        usage = getattr(response, "usage_metadata", None)
        metadata = {
            "provider": self.name,
            "model": self.model,
            "input_tokens": getattr(usage, "prompt_token_count", 0) or 0,
            "output_tokens": getattr(usage, "candidates_token_count", 0) or 0,
            "elapsed_seconds": time.perf_counter() - started,
        }
        raw = response.model_dump(mode="json") if hasattr(response, "model_dump") else {"text": response.text}
        try:
            payload = ArticleExtraction.model_validate(json.loads(response.text))
        except Exception as exc:
            raise ProviderSchemaError(str(exc), raw=raw, metadata=metadata) from exc
        return ProviderResult(payload=payload, raw=raw, metadata=metadata)


def provider_from_name(
    name: str,
    model: str | None = None,
    *,
    reasoning_effort: str = "low",
    api_key: str | None = None,
) -> ExtractionProvider:
    if name == "openai":
        return OpenAIExtractionProvider(
            model or "gpt-5.6-luna",
            reasoning_effort=reasoning_effort,
            api_key=api_key,
        )
    if name == "gemini":
        return GeminiExtractionProvider(
            model or "gemini-3.1-flash-lite", api_key=api_key
        )
    raise ValueError(f"Unsupported extraction provider: {name}")
