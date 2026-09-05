from __future__ import annotations

import importlib.metadata
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.parser.io import write_json

from .artifacts import DIMENSIONS, NORM_TOLERANCE, FrozenRelease, sha256_text
from .config import PostgresSettings, connect, require_postgres_packages
from .migrations import migration_checksums


REPORT_DIR = Path("reports/chunk_ablation/postgres")
POSTGRES_IMAGE = "pgvector/pgvector:0.8.6-pg18-trixie"


def _resolved_image_digest() -> str | None:
    configured = os.environ.get("TRAFFIC_RAG_POSTGRES_IMAGE_DIGEST")
    if configured:
        return configured
    try:
        completed = subprocess.run(
            ["docker", "image", "inspect", POSTGRES_IMAGE, "--format", "{{index .RepoDigests 0}}"],
            check=True, capture_output=True, text=True, timeout=15,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip() or None


def preflight(settings: PostgresSettings) -> dict:
    dsn = settings.admin_dsn or settings.runtime_dsn or settings.ingest_dsn
    if not dsn:
        raise ValueError("No PostgreSQL DSN configured")
    with connect(dsn, register_types=False) as connection:
        server = connection.execute("SHOW server_version").fetchone()[0]
        extension = connection.execute(
            "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
        ).fetchone()
        schema_exists = connection.execute(
            "SELECT to_regnamespace('traffic_rag') IS NOT NULL"
        ).fetchone()[0]
        roles = {
            row[0] for row in connection.execute(
                "SELECT rolname FROM pg_roles WHERE rolname = ANY(%s)",
                (["traffic_rag_owner", "traffic_rag_ingest", "traffic_rag_runtime"],),
            )
        }
    return {
        "server_version": server,
        "postgres_major_ok": int(server.split(".", 1)[0]) == 18,
        "vector_extension_version": extension[0] if extension else None,
        "schema_exists": schema_exists,
        "roles": sorted(roles),
        "ready": bool(extension and schema_exists and len(roles) == 3),
    }


def _jsonb(value):
    psycopg, _ = require_postgres_packages()
    return psycopg.types.json.Jsonb(value)


def _dataset_row(release: FrozenRelease) -> tuple:
    manifest = release.release_manifest
    return (
        release.dataset_id,
        manifest["release_name"],
        "BUILDING",
        manifest["corpus_version"],
        _jsonb(manifest["document_digests"]),
        manifest["article_passage_digest"],
        manifest["fine_passage_digest"],
        release.strategy_lock_sha256,
        manifest["embedding_model"],
        manifest["embedding_revision"],
        manifest["embedding_dimensions"],
        manifest["normalized_embeddings"],
        _jsonb(manifest),
    )


def ingest_frozen_release(
    root: str | Path,
    settings: PostgresSettings,
    release: FrozenRelease,
) -> dict:
    root_path = Path(root).resolve()
    dsn = settings.require("ingest")
    with connect(dsn) as connection:
        cursor = connection.cursor()
        connection.execute("SELECT pg_advisory_xact_lock(hashtext('traffic_rag_ingest'))")
        existing = connection.execute(
            "SELECT status FROM traffic_rag.retrieval_dataset WHERE dataset_id = %s",
            (release.dataset_id,),
        ).fetchone()
        if existing:
            raise ValueError(f"Retrieval release already exists with status {existing[0]}")
        connection.execute(
            """
            INSERT INTO traffic_rag.retrieval_dataset(
                dataset_id, release_name, status, corpus_version, document_digests,
                article_manifest_digest, fine_manifest_digest, strategy_lock_sha256,
                embedding_model, embedding_revision, embedding_dimensions,
                normalized_embeddings, release_manifest
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            _dataset_row(release),
        )

        node_rows = []
        child_rows = []
        for document in release.documents:
            for ordinal, node in enumerate(document.nodes):
                node_rows.append((
                    release.dataset_id, node.id, node.document_id, node.type,
                    _jsonb(node.hierarchy), node.title, node.text, node.parent_id,
                    _jsonb(node.source.model_dump(mode="json")) if node.source else None,
                    node.content_hash, node.node_status, _jsonb(node.metadata), ordinal,
                ))
                child_rows.extend(
                    (release.dataset_id, node.id, child_id, child_ordinal)
                    for child_ordinal, child_id in enumerate(node.children_ids)
                )
        cursor.executemany(
            """
            INSERT INTO traffic_rag.legal_node(
                dataset_id, node_id, document_id, node_type, hierarchy, title,
                text_content, parent_node_id, source, content_hash, node_status,
                metadata, source_ordinal
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            node_rows,
        )
        cursor.executemany(
            """
            INSERT INTO traffic_rag.legal_node_child(
                dataset_id, parent_node_id, child_node_id, ordinal
            ) VALUES (%s, %s, %s, %s)
            """,
            child_rows,
        )

        passage_rows = []
        link_rows = []
        for projection in release.projections:
            for passage, article_id in zip(projection.passages, projection.article_node_ids):
                text_sha = sha256_text(passage.index_text)
                passage_rows.append((
                    release.dataset_id, passage.passage_id, passage.strategy,
                    passage.document_id, passage.primary_node_id, article_id,
                    _jsonb(passage.hierarchy), passage.index_text, passage.evidence_text,
                    passage.display_text, passage.token_count_index,
                    passage.token_count_evidence, passage.content_hash, text_sha, text_sha,
                ))
                for relation_type, node_ids in (
                    ("source", passage.source_node_ids),
                    ("index", passage.index_node_ids),
                    ("context", passage.context_node_ids),
                    ("citation", passage.citation_node_ids),
                ):
                    link_rows.extend(
                        (release.dataset_id, passage.passage_id, relation_type, node_id, ordinal)
                        for ordinal, node_id in enumerate(node_ids)
                    )
        cursor.executemany(
            """
            INSERT INTO traffic_rag.retrieval_passage(
                dataset_id, passage_id, strategy, document_id, primary_node_id,
                article_node_id, hierarchy, index_text, evidence_text, display_text,
                token_count_index, token_count_evidence, passage_content_hash,
                index_text_sha256, embedding_input_sha256
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            passage_rows,
        )
        cursor.executemany(
            """
            INSERT INTO traffic_rag.passage_node_link(
                dataset_id, passage_id, relation_type, node_id, ordinal
            ) VALUES (%s, %s, %s, %s, %s)
            """,
            link_rows,
        )

        model = release.release_manifest["embedding_model"]
        revision = release.release_manifest["embedding_revision"]
        with cursor.copy("""
            COPY traffic_rag.passage_embedding(
                dataset_id, passage_id, model_name, model_revision, dimensions,
                normalized, embedding_input_sha256, embedding, status
            ) FROM STDIN WITH (FORMAT BINARY)
        """) as copy:
            copy.set_types(["text", "text", "text", "text", "int4", "bool", "text", "vector", "text"])
            for projection in release.projections:
                for passage, vector in zip(projection.passages, projection.matrix):
                    copy.write_row((
                        release.dataset_id, passage.passage_id, model, revision,
                        DIMENSIONS, True, sha256_text(passage.index_text),
                        np.asarray(vector, dtype=np.float32), "READY",
                    ))
        errors = validate_release(connection, release)
        if errors:
            raise ValueError("PostgreSQL release validation failed: " + "; ".join(errors))
        database_runtime = {
            "server_version": connection.execute("SHOW server_version").fetchone()[0],
            "vector_extension_version": connection.execute(
                "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
            ).fetchone()[0],
        }

    manifest = {
        **release.release_manifest,
        "dataset_id": release.dataset_id,
        "database_status": "BUILDING",
        "postgres_image": POSTGRES_IMAGE,
        "postgres_image_digest": _resolved_image_digest(),
        "database_runtime": database_runtime,
        "migration_checksums": migration_checksums(root_path),
        "package_versions": {
            package: importlib.metadata.version(package)
            for package in ("psycopg", "pgvector", "numpy")
        },
        "python": platform.python_version(),
        "ingested_at": datetime.now(timezone.utc).isoformat(),
    }
    output_dir = root_path / REPORT_DIR
    if release.release_manifest.get("schema_version") != "0.1.0":
        output_dir = output_dir / "releases" / release.release_manifest["release_name"]
    output = output_dir / "postgres_migration_manifest.json"
    write_json(output, manifest)
    return manifest


def validate_release(connection, release: FrozenRelease) -> list[str]:
    errors: list[str] = []
    expected_nodes = sum(len(document.nodes) for document in release.documents)
    expected_passages = len(release.passages)
    expected_links = sum(
        len(passage.source_node_ids) + len(passage.index_node_ids)
        + len(passage.context_node_ids) + len(passage.citation_node_ids)
        for passage in release.passages
    )
    counts = connection.execute("""
        SELECT
          (SELECT count(*) FROM traffic_rag.legal_node WHERE dataset_id = %s),
          (SELECT count(*) FROM traffic_rag.retrieval_passage WHERE dataset_id = %s),
          (SELECT count(*) FROM traffic_rag.passage_node_link WHERE dataset_id = %s),
          (SELECT count(*) FROM traffic_rag.passage_embedding WHERE dataset_id = %s AND status = 'READY')
    """, (release.dataset_id,) * 4).fetchone()
    for label, actual, expected in zip(
        ("nodes", "passages", "links", "embeddings"),
        counts,
        (expected_nodes, expected_passages, expected_links, expected_passages),
    ):
        if actual != expected:
            errors.append(f"{label}: expected {expected}, found {actual}")
    strategy_counts = dict(connection.execute("""
        SELECT strategy, count(*) FROM traffic_rag.retrieval_passage
        WHERE dataset_id = %s GROUP BY strategy
    """, (release.dataset_id,)).fetchall())
    if strategy_counts != {"B1_article": 175, "B4e_document_article_clause_point": 1460}:
        errors.append(f"unexpected strategy counts: {strategy_counts}")
    vector_row = connection.execute("""
        SELECT min(vector_dims(embedding)), max(vector_dims(embedding)),
               max(abs(vector_norm(embedding) - 1.0))
        FROM traffic_rag.passage_embedding WHERE dataset_id = %s
    """, (release.dataset_id,)).fetchone()
    if vector_row[0] != DIMENSIONS or vector_row[1] != DIMENSIONS:
        errors.append(f"unexpected vector dimensions: {vector_row[:2]}")
    if vector_row[2] is None or float(vector_row[2]) > NORM_TOLERANCE:
        errors.append(f"vectors not normalized: max error {vector_row[2]}")
    return errors


def validate_database(settings: PostgresSettings, release: FrozenRelease) -> dict:
    dsn = settings.runtime_dsn or settings.ingest_dsn
    if not dsn:
        raise ValueError("Missing runtime or ingest DSN")
    with connect(dsn) as connection:
        errors = validate_release(connection, release)
        status = connection.execute(
            "SELECT status FROM traffic_rag.retrieval_dataset WHERE dataset_id = %s",
            (release.dataset_id,),
        ).fetchone()
    if not status:
        errors.append("release is not present")
    return {"dataset_id": release.dataset_id, "status": status[0] if status else None, "errors": errors, "valid": not errors}


def activate_release(settings: PostgresSettings, dataset_id: str) -> dict:
    if len(dataset_id) != 64:
        raise ValueError("dataset_id must be a 64-character release digest")
    with connect(settings.require("ingest")) as connection:
        connection.execute("SELECT pg_advisory_xact_lock(hashtext('traffic_rag_activate'))")
        row = connection.execute(
            "SELECT status FROM traffic_rag.retrieval_dataset WHERE dataset_id = %s FOR UPDATE",
            (dataset_id,),
        ).fetchone()
        if not row:
            raise KeyError(f"Unknown retrieval release: {dataset_id}")
        if row[0] == "ACTIVE":
            return {"dataset_id": dataset_id, "status": "ACTIVE", "changed": False}
        if row[0] != "BUILDING":
            raise ValueError(f"Only BUILDING releases can be activated, found {row[0]}")
        connection.execute(
            "UPDATE traffic_rag.retrieval_dataset SET status = 'RETIRED' WHERE status = 'ACTIVE'"
        )
        connection.execute(
            "UPDATE traffic_rag.retrieval_dataset SET status = 'ACTIVE', activated_at = now() WHERE dataset_id = %s",
            (dataset_id,),
        )
    return {"dataset_id": dataset_id, "status": "ACTIVE", "changed": True}
