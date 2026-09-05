from __future__ import annotations

import os

import pytest

from src.graph.config import GraphSettings
from src.graph.repository import connect_graph, validate_database


def _settings() -> GraphSettings:
    dsn = os.environ.get("LIGHTRAG_TEST_DSN")
    if not dsn:
        pytest.skip("Set LIGHTRAG_TEST_DSN after graph db-init/build-structural")
    return GraphSettings(
        admin_dsn=None,
        app_dsn=dsn,
        app_password=None,
    )


def test_graph_database_is_isolated_and_structural_release_is_complete():
    settings = _settings()
    with connect_graph(settings) as connection:
        database = connection.execute("SELECT current_database()").fetchone()[0]
        assert database == "traffic_rag_lightrag"
        assert connection.execute("SELECT to_regclass('traffic_rag.retrieval_dataset')").fetchone()[0] is None
        result = validate_database(connection, "kg-v001")
    assert result["valid"], result["errors"]


def test_graph_database_keeps_directed_multi_edges():
    settings = _settings()
    with connect_graph(settings) as connection:
        rows = connection.execute(
            """
            SELECT source_node_id, target_node_id, count(DISTINCT relation_type)
            FROM legal_graph_edge
            WHERE release_id='kg-v001'
            GROUP BY source_node_id, target_node_id
            HAVING count(DISTINCT relation_type) > 1
            """
        ).fetchall()
        edge_count = connection.execute(
            "SELECT count(*) FROM legal_graph_edge WHERE release_id='kg-v001'"
        ).fetchone()[0]
    assert edge_count >= 3324
    assert isinstance(rows, list)
