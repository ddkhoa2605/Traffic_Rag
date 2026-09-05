from __future__ import annotations

import os
from pathlib import Path

import pytest

from src.postgres_store.artifacts import load_frozen_release
from src.postgres_store.config import PostgresSettings
from src.postgres_store.repository import validate_database


@pytest.mark.postgres
def test_frozen_release_in_real_pgvector_database():
    dsn = os.environ.get("TRAFFIC_RAG_TEST_POSTGRES_DSN")
    if not dsn:
        pytest.skip("Set TRAFFIC_RAG_TEST_POSTGRES_DSN after migrate + ingest-frozen")
    root = Path(__file__).resolve().parents[2]
    release = load_frozen_release(root)
    settings = PostgresSettings.load(root, dsn=dsn)
    result = validate_database(settings, release)
    assert result["valid"], result["errors"]
    assert result["status"] in {"BUILDING", "ACTIVE"}
