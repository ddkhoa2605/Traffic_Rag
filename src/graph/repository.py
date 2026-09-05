from __future__ import annotations

from pathlib import Path

from src.postgres_store.config import require_postgres_packages

from .config import GraphSettings
from .identity import normalized_label
from .models import GraphEdge, GraphNode, GraphRelease, GraphSourceDocument


RELEASE_TRANSITIONS = {
    "BUILDING": {"VALIDATING", "FAILED"},
    "VALIDATING": {"READY", "FAILED"},
    "READY": {"ACTIVE", "FAILED"},
    "ACTIVE": {"RETIRED"},
    "FAILED": {"BUILDING"},
    "RETIRED": {"ACTIVE"},
}


def _executemany(connection, query: str, params: list[tuple]) -> None:
    """Execute a batch with Psycopg 3's cursor-level API."""
    if not params:
        return
    with connection.cursor() as cursor:
        cursor.executemany(query, params)


def validate_release_transition(current: str, target: str) -> None:
    if current not in RELEASE_TRANSITIONS:
        raise ValueError(f"Unknown graph release status: {current}")
    if target not in RELEASE_TRANSITIONS[current]:
        raise ValueError(f"Invalid graph release transition: {current} -> {target}")


def connect_graph(settings: GraphSettings, *, autocommit: bool = False):
    psycopg, _ = require_postgres_packages()
    return psycopg.connect(
        settings.require("app_dsn"), autocommit=autocommit, connect_timeout=5
    )


def upsert_release(connection, release: GraphRelease) -> None:
    from psycopg.types.json import Jsonb

    connection.execute(
        """
        INSERT INTO legal_graph_release(
            release_id, status, workspace, canonical_digest, b7_dataset_id,
            source_manifest_hash, config_hash, extraction_provider,
            extraction_model, metadata
        ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        ON CONFLICT (release_id) DO UPDATE SET
            status=CASE
                WHEN legal_graph_release.status='FAILED' THEN 'BUILDING'
                ELSE legal_graph_release.status
            END,
            workspace=EXCLUDED.workspace,
            canonical_digest=EXCLUDED.canonical_digest,
            b7_dataset_id=EXCLUDED.b7_dataset_id,
            source_manifest_hash=EXCLUDED.source_manifest_hash,
            config_hash=EXCLUDED.config_hash,
            extraction_provider=EXCLUDED.extraction_provider,
            extraction_model=EXCLUDED.extraction_model,
            metadata=EXCLUDED.metadata,
            updated_at=now()
        WHERE legal_graph_release.status IN ('BUILDING','FAILED')
        """,
        (
            release.release_id,
            release.status,
            release.workspace,
            release.canonical_digest,
            release.b7_dataset_id,
            release.source_manifest_hash,
            release.config_hash,
            release.extraction_provider,
            release.extraction_model,
            Jsonb(release.metadata),
        ),
    )


def reset_building_release(connection, release_id: str) -> None:
    row = connection.execute(
        "SELECT status FROM legal_graph_release WHERE release_id=%s FOR UPDATE",
        (release_id,),
    ).fetchone()
    if not row:
        raise KeyError(release_id)
    if row[0] != "BUILDING":
        raise ValueError(
            f"Cannot rebuild immutable graph release {release_id} in status {row[0]}"
        )
    connection.execute("DELETE FROM legal_graph_mirror_map WHERE release_id=%s", (release_id,))
    connection.execute("DELETE FROM legal_graph_alias WHERE release_id=%s", (release_id,))
    connection.execute("DELETE FROM legal_graph_node WHERE release_id=%s", (release_id,))
    connection.execute("DELETE FROM legal_graph_source_document WHERE release_id=%s", (release_id,))


def store_sources(connection, release_id: str, sources: list[GraphSourceDocument]) -> None:
    from psycopg.types.json import Jsonb

    _executemany(
        connection,
        """
        INSERT INTO legal_graph_source_document(
            release_id, article_node_id, canonical_document_id, title, text,
            extraction_text, source_node_ids, node_texts, token_count, content_hash
        ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        ON CONFLICT (release_id, article_node_id) DO UPDATE SET
            title=EXCLUDED.title, text=EXCLUDED.text,
            extraction_text=EXCLUDED.extraction_text,
            source_node_ids=EXCLUDED.source_node_ids, node_texts=EXCLUDED.node_texts,
            token_count=EXCLUDED.token_count, content_hash=EXCLUDED.content_hash
        """,
        [
            (
                release_id,
                item.article_node_id,
                item.canonical_document_id,
                item.title,
                item.text,
                item.extraction_text,
                Jsonb(item.source_node_ids),
                Jsonb(item.node_texts),
                item.token_count,
                item.content_hash,
            )
            for item in sources
        ],
    )


def store_graph(
    connection,
    release_id: str,
    nodes: list[GraphNode],
    edges: list[GraphEdge],
) -> None:
    from psycopg.types.json import Jsonb

    _executemany(
        connection,
        """
        INSERT INTO legal_graph_node(
            release_id, graph_node_id, node_kind, node_type, label,
            document_id, canonical_node_id, properties
        ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
        ON CONFLICT (release_id, graph_node_id) DO UPDATE SET
            node_kind=EXCLUDED.node_kind, node_type=EXCLUDED.node_type,
            label=EXCLUDED.label, document_id=EXCLUDED.document_id,
            canonical_node_id=EXCLUDED.canonical_node_id,
            properties=legal_graph_node.properties || EXCLUDED.properties
        """,
        [
            (
                release_id,
                node.graph_node_id,
                node.node_kind,
                node.node_type,
                node.label,
                node.document_id,
                node.canonical_node_id,
                Jsonb(node.properties),
            )
            for node in nodes
        ],
    )
    _executemany(
        connection,
        """
        INSERT INTO legal_graph_edge(
            release_id, edge_id, source_node_id, target_node_id, relation_type,
            description, confidence, extraction_version, properties
        ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
        ON CONFLICT (release_id, edge_id) DO UPDATE SET
            description=EXCLUDED.description, confidence=EXCLUDED.confidence,
            extraction_version=EXCLUDED.extraction_version,
            properties=EXCLUDED.properties
        """,
        [
            (
                release_id,
                edge.edge_id,
                edge.source_node_id,
                edge.target_node_id,
                edge.relation_type,
                edge.description,
                edge.confidence,
                edge.extraction_version,
                Jsonb(edge.properties),
            )
            for edge in edges
        ],
    )
    for edge in edges:
        connection.execute(
            "DELETE FROM legal_graph_provenance WHERE release_id=%s AND edge_id=%s",
            (release_id, edge.edge_id),
        )
        _executemany(
            connection,
            """
            INSERT INTO legal_graph_provenance(
                release_id, edge_id, ordinal, canonical_node_id,
                evidence_text, start_offset, end_offset
            ) VALUES (%s,%s,%s,%s,%s,%s,%s)
            """,
            [
                (
                    release_id,
                    edge.edge_id,
                    ordinal,
                    item.canonical_node_id,
                    item.evidence_text,
                    item.start_offset,
                    item.end_offset,
                )
                for ordinal, item in enumerate(edge.provenance)
            ],
        )


def validate_database(connection, release_id: str) -> dict:
    row = connection.execute(
        """
        SELECT
          (SELECT count(*) FROM legal_graph_source_document WHERE release_id=%s),
          (SELECT count(*) FROM legal_graph_node WHERE release_id=%s AND node_kind='canonical'),
          (SELECT count(*) FROM legal_graph_edge WHERE release_id=%s AND relation_type='CONTAINS'),
          (SELECT count(*) FROM legal_graph_edge WHERE release_id=%s AND relation_type='NEXT'),
          (SELECT count(*) FROM legal_graph_edge WHERE release_id=%s AND relation_type='ISSUED_BY'),
          (SELECT count(*) FROM legal_graph_edge e
             LEFT JOIN legal_graph_node s ON s.release_id=e.release_id AND s.graph_node_id=e.source_node_id
             LEFT JOIN legal_graph_node t ON t.release_id=e.release_id AND t.graph_node_id=e.target_node_id
           WHERE e.release_id=%s AND (s.graph_node_id IS NULL OR t.graph_node_id IS NULL)),
          (SELECT count(*) FROM legal_graph_edge e
           WHERE e.release_id=%s
             AND e.relation_type NOT IN ('CONTAINS','NEXT','ISSUED_BY','REFERENCES')
             AND NOT EXISTS (
               SELECT 1 FROM legal_graph_provenance p
               WHERE p.release_id=e.release_id AND p.edge_id=e.edge_id
             ))
        """,
        (release_id,) * 7,
    ).fetchone()
    labels = (
        "source_articles",
        "canonical_nodes",
        "contains_edges",
        "next_edges",
        "issued_by_edges",
        "dangling_edges",
        "semantic_edges_without_provenance",
    )
    counts = dict(zip(labels, row))
    errors: list[str] = []
    expected = {
        "source_articles": 175,
        "canonical_nodes": 1850,
        "contains_edges": 1848,
        "next_edges": 1474,
        "issued_by_edges": 2,
        "dangling_edges": 0,
        "semantic_edges_without_provenance": 0,
    }
    for key, value in expected.items():
        if counts[key] != value:
            errors.append(f"{key}: expected {value}, found {counts[key]}")
    return {"release_id": release_id, "valid": not errors, "counts": counts, "errors": errors}


def transition_release(connection, release_id: str, target: str) -> None:
    row = connection.execute(
        "SELECT status FROM legal_graph_release WHERE release_id=%s FOR UPDATE", (release_id,)
    ).fetchone()
    if not row:
        raise KeyError(release_id)
    current = row[0]
    validate_release_transition(current, target)
    if target == "ACTIVE":
        connection.execute(
            "UPDATE legal_graph_release SET status='RETIRED', updated_at=now() WHERE status='ACTIVE'"
        )
    connection.execute(
        "UPDATE legal_graph_release SET status=%s, updated_at=now() WHERE release_id=%s",
        (target, release_id),
    )


def store_mirror_map(
    connection,
    release_id: str,
    workspace: str,
    rows: list[tuple[str, str, str, dict]],
) -> None:
    from psycopg.types.json import Jsonb

    _executemany(
        connection,
        """
        INSERT INTO legal_graph_mirror_map(
            release_id, object_kind, legal_object_id, lightrag_object_id,
            workspace, mirror_status, metadata
        ) VALUES (%s,%s,%s,%s,%s,'READY',%s)
        ON CONFLICT (release_id, object_kind, legal_object_id) DO UPDATE SET
            lightrag_object_id=EXCLUDED.lightrag_object_id,
            workspace=EXCLUDED.workspace, mirror_status='READY', metadata=EXCLUDED.metadata
        """,
        [
            (release_id, kind, legal_id, lightrag_id, workspace, Jsonb(metadata))
            for kind, legal_id, lightrag_id, metadata in rows
        ],
    )


def rebuild_approved_aliases(connection, release_id: str) -> int:
    rows = connection.execute(
        """
        SELECT graph_node_id, properties
        FROM legal_graph_node
        WHERE release_id=%s AND node_kind='semantic'
        ORDER BY graph_node_id
        """,
        (release_id,),
    ).fetchall()
    values: list[tuple[str, str, str, str]] = []
    for graph_node_id, properties in rows:
        seen: set[str] = set()
        for alias in properties.get("aliases", []):
            normalized = normalized_label(alias)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            values.append((release_id, graph_node_id, alias, normalized))
    connection.execute("DELETE FROM legal_graph_alias WHERE release_id=%s", (release_id,))
    _executemany(
        connection,
        """
        INSERT INTO legal_graph_alias(
            release_id, graph_node_id, alias, normalized_alias, review_status
        ) VALUES (%s,%s,%s,%s,'approved')
        """,
        values,
    )
    return len(values)


def fetch_release_graph(connection, release_id: str) -> tuple[list[dict], list[dict], list[dict]]:
    nodes = connection.execute(
        """
        SELECT graph_node_id, node_kind, node_type, label, document_id,
               canonical_node_id, properties
        FROM legal_graph_node WHERE release_id=%s ORDER BY graph_node_id
        """,
        (release_id,),
    ).fetchall()
    edges = connection.execute(
        """
        SELECT edge_id, source_node_id, target_node_id, relation_type,
               description, confidence, extraction_version, properties
        FROM legal_graph_edge WHERE release_id=%s ORDER BY edge_id
        """,
        (release_id,),
    ).fetchall()
    provenance = connection.execute(
        """
        SELECT edge_id, ordinal, canonical_node_id, evidence_text, start_offset, end_offset
        FROM legal_graph_provenance WHERE release_id=%s ORDER BY edge_id, ordinal
        """,
        (release_id,),
    ).fetchall()
    return nodes, edges, provenance
