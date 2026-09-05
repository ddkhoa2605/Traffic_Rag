from __future__ import annotations

from pathlib import Path

import yaml
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class QueryEncoderConfig(BaseModel):
    model: str
    revision: str
    max_length: int = Field(gt=0, le=8192)
    preferred_device: str = "cpu"


class ReferenceRoutingConfig(BaseModel):
    enabled: bool = False


class MultiEvidenceConfig(BaseModel):
    enabled: bool = False


class RerankerConfig(BaseModel):
    model: str
    revision: str
    candidate_top_n: int = Field(default=30, ge=5, le=100)


class ApplicationConfig(BaseModel):
    version: str
    retrieval_backend: str
    strategy: str
    routing_policy: Literal["B6", "B7a", "B7b", "B7c", "B7d"] = "B6"
    dataset_selector: str
    runtime_dsn_env: str
    require_strategy_lock_match: bool = True
    require_b7_policy_lock_match: bool = True
    reference_routing: ReferenceRoutingConfig = Field(default_factory=ReferenceRoutingConfig)
    multi_evidence: MultiEvidenceConfig = Field(default_factory=MultiEvidenceConfig)
    reranker: RerankerConfig | None = None
    query_encoder: QueryEncoderConfig
    default_top_k: int = Field(gt=0, le=10)
    max_top_k: int = Field(gt=0, le=10)

    @model_validator(mode="after")
    def validate_frozen_runtime(self):
        if self.retrieval_backend != "postgres":
            raise ValueError("Application runtime is locked to the PostgreSQL backend")
        if self.strategy != "B6" or self.dataset_selector != "active":
            raise ValueError("Application runtime is locked to ACTIVE B6")
        expected = {
            "B6": (False, False),
            "B7a": (True, False),
            "B7b": (False, True),
            "B7c": (True, True),
            "B7d": (True, True),
        }[self.routing_policy]
        actual = (self.reference_routing.enabled, self.multi_evidence.enabled)
        if actual != expected:
            raise ValueError(
                f"routing_policy {self.routing_policy} requires "
                f"reference_routing={expected[0]} and multi_evidence={expected[1]}"
            )
        if self.routing_policy == "B7d" and self.reranker is None:
            raise ValueError("B7d requires a pinned reranker configuration")
        if self.routing_policy != "B7d" and self.reranker is not None:
            raise ValueError("reranker may only be configured for B7d")
        if self.default_top_k > self.max_top_k:
            raise ValueError("default_top_k exceeds max_top_k")
        return self


def load_application_config(root: str | Path) -> ApplicationConfig:
    path = Path(root).resolve() / "configs/application.yaml"
    return ApplicationConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
