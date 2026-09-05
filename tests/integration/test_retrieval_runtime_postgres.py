from __future__ import annotations

import os
from pathlib import Path

import pytest

from src.retrieval_runtime import application_health


@pytest.mark.postgres
def test_application_postgres_backend_is_ready():
    if not os.environ.get("TRAFFIC_RAG_TEST_POSTGRES_DSN"):
        pytest.skip("Set TRAFFIC_RAG_TEST_POSTGRES_DSN to run PostgreSQL runtime integration")
    root = Path(__file__).resolve().parents[2]
    health = application_health(root)
    assert health["ready"]
    assert health["backend"] == "postgres"
    assert health["strategy"] == "B6"
    assert health["routing_policy"] == "B6"
    assert health["passage_count"] == health["embedding_count"] == 1635
    assert health["b7_policy_lock_match"]
