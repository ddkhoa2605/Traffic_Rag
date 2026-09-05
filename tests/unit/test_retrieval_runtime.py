from __future__ import annotations

from pathlib import Path

import pytest

from src.retrieval_runtime.config import ApplicationConfig, load_application_config
from src.retrieval_runtime.models import RuntimeSearchResult


def test_application_default_uses_locked_b7c_over_frozen_postgres_b6():
    root = Path(__file__).resolve().parents[2]
    config = load_application_config(root)
    assert config.retrieval_backend == "postgres"
    assert config.strategy == "B6"
    assert config.routing_policy == "B7c"
    assert config.reference_routing.enabled
    assert config.multi_evidence.enabled
    assert config.dataset_selector == "active"
    assert config.query_encoder.model == "BAAI/bge-m3"
    assert config.default_top_k == 5 and config.max_top_k == 10


def test_application_config_rejects_numpy_default():
    root = Path(__file__).resolve().parents[2]
    value = load_application_config(root).model_dump()
    value["retrieval_backend"] = "numpy"
    with pytest.raises(ValueError, match="PostgreSQL"):
        ApplicationConfig.model_validate(value)


def test_runtime_result_requires_canonical_citations():
    result = RuntimeSearchResult(
        rank=1, score=.5, source_strategy="B4e", passage_id="p",
        primary_node_id="n", document_id="d", hierarchy={},
        evidence_text="evidence", included_node_ids=["n"],
        context_node_ids=[], citation_node_ids=["n"], evidence_tokens=1,
    )
    assert result.citation_node_ids == ["n"]
    assert result.member_node_ids == ["n"]
