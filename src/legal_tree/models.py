from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


BBox = tuple[float, float, float, float]
NodeType = Literal[
    "document", "preamble", "part", "chapter", "section", "subsection",
    "article", "clause", "point", "subpoint", "paragraph", "table",
    "table_row", "appendix", "form", "footnote", "amendment_instruction",
    "signature", "unknown_block",
]
NodeStatus = Literal["active", "amended", "repealed", "replaced", "unknown"]
RelationType = Literal[
    "REFERENCES", "AMENDS", "REPEALS", "REPLACES", "EXCEPTS",
    "APPLIES_TO", "IMPLEMENTS", "GUIDES", "CONSOLIDATES",
]
HIERARCHY_KEYS = (
    "part", "chapter", "section", "subsection", "article", "clause",
    "point", "subpoint",
)


class SourceBlockRef(BaseModel):
    block_id: str
    page: int
    bbox: BBox


class NodeSource(BaseModel):
    source_file_id: str
    page_start: int
    page_end: int
    blocks: list[SourceBlockRef]


class LegalNode(BaseModel):
    """Backward-compatible canonical node model.

    Defaults are injected only in memory. Loading this model never rewrites the
    validated corpus and never participates in canonical content hashing.
    """

    id: str
    document_id: str
    type: NodeType
    hierarchy: dict[str, str | None]
    title: str | None = None
    text: str = ""
    parent_id: str | None = None
    children_ids: list[str] = Field(default_factory=list)
    source: NodeSource | None = None
    content_hash: str
    node_status: NodeStatus = "active"
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("hierarchy", mode="before")
    @classmethod
    def normalize_hierarchy(cls, value):
        source = dict(value or {})
        return {key: source.get(key) for key in HIERARCHY_KEYS}


class LegalRelation(BaseModel):
    relation_id: str
    relation_type: RelationType
    source_node_id: str
    target_node_id: str
    metadata: dict[str, Any] = Field(default_factory=dict)


@dataclass(frozen=True)
class LegalDocument:
    document_id: str
    nodes: tuple[LegalNode, ...]

