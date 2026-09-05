from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Literal

from pydantic import BaseModel, Field


class VisionResult(BaseModel):
    page: int
    task: str
    native_result: Any = None
    vision_result: Any = None
    confidence: float = Field(ge=0, le=1)
    status: Literal["AGREEMENT", "DISAGREEMENT", "INCONCLUSIVE"]


class VisionVerifier(ABC):
    """Optional verifier. Implementations may advise, never mutate canonical nodes."""

    @abstractmethod
    def inspect_page(self, image_path: str, task: str, *, context: dict | None = None) -> VisionResult:
        raise NotImplementedError

