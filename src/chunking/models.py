from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


StrategyName = Literal[
    "B0_fixed_window", "B1_article", "B2_clause", "B3_point",
    "B4_point_clause_context", "B5_child_parent", "B4a_point",
    "B4b_article_point", "B4c_clause_point", "B4d_article_clause_point",
    "B4e_document_article_clause_point",
]


class RetrievalPassage(BaseModel):
    passage_id: str
    strategy: StrategyName
    document_id: str
    primary_node_id: str
    source_node_ids: list[str]
    index_node_ids: list[str]
    context_node_ids: list[str]
    citation_node_ids: list[str]
    hierarchy: dict[str, str | None]
    index_text: str
    evidence_text: str
    display_text: str
    token_count_index: int = Field(gt=0)
    token_count_evidence: int = Field(gt=0)
    content_hash: str

    @model_validator(mode="after")
    def validate_lists(self):
        if self.primary_node_id not in self.source_node_ids:
            raise ValueError("primary_node_id must be a source node")
        if self.primary_node_id not in self.index_node_ids:
            raise ValueError("primary_node_id must be an index node")
        if not self.citation_node_ids:
            raise ValueError("citation_node_ids must not be empty")
        for name in ("source_node_ids", "index_node_ids", "context_node_ids", "citation_node_ids"):
            values = getattr(self, name)
            if len(values) != len(dict.fromkeys(values)):
                raise ValueError(f"{name} contains duplicates")
        return self


class ExpansionPolicy(BaseModel):
    include_article_heading: bool = True
    include_clause_intro: bool = True


class EvidenceBundle(BaseModel):
    primary_node_id: str
    included_node_ids: list[str]
    context_node_ids: list[str]
    citation_node_ids: list[str]
    evidence_text: str
    token_count: int = Field(gt=0)


class PassageManifest(BaseModel):
    strategy: StrategyName
    corpus_version: str
    chunk_builder_version: str
    tokenizer_model: str
    tokenizer_revision: str
    documents: list[str]
    document_digests: dict[str, str]
    passage_count: int
    avg_index_tokens: float
    avg_evidence_tokens: float
    passage_digest: str
    code_digest: str
    git_commit: str | None = None
    git_dirty: bool | None = None
    generated_at: str
