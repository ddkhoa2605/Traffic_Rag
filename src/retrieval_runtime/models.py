from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator
from src.retrieval_eval.models import EvidenceComponent


ResolutionStatus = Literal[
    "NO_REFERENCE", "RESOLVED", "PARTIAL", "AMBIGUOUS", "NOT_FOUND", "CONFLICT",
]


class ReferenceSpan(BaseModel):
    kind: str
    text: str
    start: int = Field(ge=0)
    end: int = Field(ge=0)


class LegalReferenceIntent(BaseModel):
    article: str | None = None
    clause: str | None = None
    points: list[str] = Field(default_factory=list)
    explicit_document_alias: str | None = None
    explicit_document_ids: list[str] = Field(default_factory=list)
    matched_spans: list[ReferenceSpan] = Field(default_factory=list)


class ReferenceResolution(BaseModel):
    status: ResolutionStatus
    candidate_document_ids: list[str] = Field(default_factory=list)
    resolved_node_ids: list[str] = Field(default_factory=list)
    message: str | None = None


class RuntimeSearchResult(BaseModel):
    rank: int = Field(gt=0, le=10)
    score: float
    source_strategy: str
    passage_id: str
    primary_node_id: str
    document_id: str
    hierarchy: dict[str, str | None]
    evidence_text: str
    included_node_ids: list[str]
    context_node_ids: list[str]
    citation_node_ids: list[str]
    evidence_tokens: int = Field(gt=0)
    result_type: Literal["passage", "canonical_node", "evidence_bundle", "suggestion"] = "passage"
    member_node_ids: list[str] = Field(default_factory=list)
    evidence_components: list[EvidenceComponent] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_canonical_provenance(self):
        if not self.citation_node_ids:
            raise ValueError("citation_node_ids must not be empty")
        if not self.member_node_ids:
            self.member_node_ids = list(self.citation_node_ids)
        return self


class RuntimeSearchResponse(BaseModel):
    query: str
    backend: str = "postgres"
    strategy: str = "B6"
    dataset_id: str
    model: str
    model_revision: str
    routing_policy: Literal["B6", "B7a", "B7b", "B7c", "B7d"] = "B6"
    route: str = "B6"
    resolution_status: ResolutionStatus = "NO_REFERENCE"
    parsed_reference: LegalReferenceIntent | None = None
    ambiguity_candidates: list[str] = Field(default_factory=list)
    results: list[RuntimeSearchResult]
