from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class SearchHit(BaseModel):
    passage_id: str
    score: float
    rank: int
    source_strategy: str | None = None
    evidence_passage_ids: list[str] = Field(default_factory=list)


class EvidenceComponent(BaseModel):
    node_id: str
    role: str
    token_sequence_hash: str
    token_count: int


class PerRankResult(BaseModel):
    run_id: str
    query_id: str
    split: str
    category: str
    rank: int
    retrieved_passage_id: str
    retrieved_primary_node_id: str
    score: float
    exact_hit: bool
    structural_hit: bool
    index_tokens: int
    evidence_tokens: int
    included_node_ids: list[str]
    context_node_ids: list[str]
    citation_node_ids: list[str]
    structurally_covered_gold_ids: list[str]
    evidence_components: list[EvidenceComponent] = Field(default_factory=list)


class RunSummary(BaseModel):
    run_id: str
    strategy: str
    retriever: Literal["bm25", "dense"]
    split: Literal["dev", "test"]
    provisional: bool
    query_count: int
    metrics: dict[str, float]
    category_metrics: dict[str, dict[str, float]]
    retriever_config: dict
    corpus_digests: dict[str, str]
    passage_manifest_hash: str
    dataset_hash: str
    code_digest: str
    git_commit: str | None = None
    metrics_schema_version: str = "0.1.0"
    ranking_digest: str | None = None
