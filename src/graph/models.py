from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


ReleaseStatus = Literal["BUILDING", "VALIDATING", "READY", "ACTIVE", "RETIRED", "FAILED"]
NodeKind = Literal["canonical", "semantic", "authority"]
EntityType = Literal[
    "Actor",
    "Vehicle",
    "Infrastructure",
    "Action",
    "Violation",
    "Sanction",
    "TimeConstraint",
    "LegalConcept",
    "Condition",
    "Exception",
]
StructuralRelation = Literal["CONTAINS", "NEXT", "ISSUED_BY", "REFERENCES"]
SemanticRelation = Literal[
    "DEFINES",
    "APPLIES_TO",
    "REGULATES",
    "REQUIRES",
    "PROHIBITS",
    "HAS_DEADLINE",
    "CONDITION_FOR",
    "EXCEPTION_TO",
]
ExtractionRelation = SemanticRelation | Literal["OTHER"]
GraphRelationType = StructuralRelation | SemanticRelation


class GraphSourceDocument(BaseModel):
    model_config = ConfigDict(extra="forbid")

    article_node_id: str
    canonical_document_id: str
    title: str
    text: str = Field(min_length=1)
    extraction_text: str = Field(min_length=1)
    source_node_ids: list[str] = Field(min_length=1)
    node_texts: dict[str, str]
    token_count: int = Field(ge=1)
    content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")


class GraphNode(BaseModel):
    model_config = ConfigDict(extra="forbid")

    graph_node_id: str
    node_kind: NodeKind
    node_type: str
    label: str = Field(min_length=1)
    document_id: str | None = None
    canonical_node_id: str | None = None
    properties: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def canonical_mapping(self) -> "GraphNode":
        if self.node_kind == "canonical" and self.canonical_node_id != self.graph_node_id:
            raise ValueError("Canonical graph nodes must preserve canonical node IDs")
        return self


class GraphProvenance(BaseModel):
    model_config = ConfigDict(extra="forbid")

    canonical_node_id: str
    evidence_text: str = Field(min_length=1)
    start_offset: int = Field(ge=0)
    end_offset: int = Field(gt=0)

    @model_validator(mode="after")
    def offset_order(self) -> "GraphProvenance":
        if self.end_offset <= self.start_offset:
            raise ValueError("end_offset must be greater than start_offset")
        return self


class GraphEdge(BaseModel):
    model_config = ConfigDict(extra="forbid")

    edge_id: str
    source_node_id: str
    target_node_id: str
    relation_type: GraphRelationType
    description: str = ""
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    extraction_version: str = "structural-v1"
    provenance: list[GraphProvenance] = Field(default_factory=list)
    properties: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def no_self_loop(self) -> "GraphEdge":
        if self.source_node_id == self.target_node_id:
            raise ValueError("Graph self-loops are not allowed")
        return self


class GraphRelease(BaseModel):
    model_config = ConfigDict(extra="forbid")

    release_id: str
    status: ReleaseStatus = "BUILDING"
    workspace: str
    canonical_digest: str
    b7_dataset_id: str
    source_manifest_hash: str
    config_hash: str
    extraction_provider: str | None = None
    extraction_model: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime | None = None


class ExtractionEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    canonical_node_id: str
    evidence_text: str = Field(min_length=1)


class ExtractedEntity(BaseModel):
    model_config = ConfigDict(extra="forbid")

    local_key: str = Field(min_length=1)
    label: str = Field(min_length=1)
    entity_type: EntityType
    description: str = Field(min_length=1)
    aliases: list[str]
    provenance: list[ExtractionEvidence] = Field(min_length=1)


class ExtractedRelationship(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_ref: str = Field(min_length=1)
    target_ref: str = Field(min_length=1)
    relation_type: ExtractionRelation
    description: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    provenance: list[ExtractionEvidence] = Field(min_length=1)


class ArticleExtraction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    entities: list[ExtractedEntity]
    relationships: list[ExtractedRelationship]
