from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


Severity = Literal["INFO", "WARN", "ERROR", "FATAL"]


class ValidationIssue(BaseModel):
    code: str
    severity: Severity
    message: str
    page: int | None = None
    node_id: str | None = None
    block_id: str | None = None
    vision_review_required: bool = False


class ValidationReport(BaseModel):
    document_id: str
    status: Literal["PASS", "WARN", "ERROR", "FATAL"]
    metrics: dict[str, float | int | str]
    issues: list[ValidationIssue] = Field(default_factory=list)

