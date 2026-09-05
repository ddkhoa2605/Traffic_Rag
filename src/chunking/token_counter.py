from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class EncodedText:
    token_ids: list[int]
    offsets: list[tuple[int, int]]


class TokenCounter(Protocol):
    model_name: str
    revision: str
    max_tokens: int

    def encode_with_offsets(self, text: str) -> EncodedText: ...
    def count(self, text: str) -> int: ...


class BGETokenCounter:
    model_name = "BAAI/bge-m3"
    max_tokens = 8192

    def __init__(self, revision: str = "main"):
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, revision=revision, use_fast=True)
        # Offset mapping may process a whole document. The 8192-token ceiling is
        # enforced on every resulting index passage by the integrity validator.
        self.tokenizer.model_max_length = 1_000_000_000
        resolved = getattr(self.tokenizer, "init_kwargs", {}).get("_commit_hash")
        self.revision = resolved or revision

    def encode_with_offsets(self, text: str) -> EncodedText:
        encoded = self.tokenizer(
            text, add_special_tokens=False, return_offsets_mapping=True,
            truncation=False,
        )
        return EncodedText(
            token_ids=list(encoded["input_ids"]),
            offsets=[tuple(pair) for pair in encoded["offset_mapping"]],
        )

    def count(self, text: str) -> int:
        return len(self.encode_with_offsets(text).token_ids)


class RegexTokenCounter:
    """Deterministic lightweight counter for unit tests only."""

    model_name = "test-regex"
    revision = "v1"
    max_tokens = 8192

    def encode_with_offsets(self, text: str) -> EncodedText:
        import re

        matches = list(re.finditer(r"\S+", text, re.UNICODE))
        return EncodedText(
            token_ids=list(range(len(matches))),
            offsets=[match.span() for match in matches],
        )

    def count(self, text: str) -> int:
        return len(self.encode_with_offsets(text).token_ids)
