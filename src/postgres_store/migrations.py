from __future__ import annotations

import hashlib
from pathlib import Path

from .config import PostgresSettings, require_postgres_packages


ROLE_PASSWORD_FIELDS = {
    "traffic_rag_owner": "owner_password",
    "traffic_rag_ingest": "ingest_password",
    "traffic_rag_runtime": "runtime_password",
}


def _migration_files(root: Path) -> list[Path]:
    paths = sorted((root / "sql/migrations").glob("[0-9][0-9][0-9][0-9]_*.sql"))
    if not paths:
        raise FileNotFoundError("No SQL migrations found")
    return paths


def _bootstrap_roles(connection, settings: PostgresSettings) -> None:
    psycopg, _ = require_postgres_packages()
    sql = psycopg.sql
    for role, field in ROLE_PASSWORD_FIELDS.items():
        password = getattr(settings, field)
        if not password:
            raise ValueError(f"Missing {field.upper()} for PostgreSQL role bootstrap")
        exists = connection.execute("SELECT 1 FROM pg_roles WHERE rolname = %s", (role,)).fetchone()
        if not exists:
            connection.execute(
                sql.SQL("CREATE ROLE {} LOGIN PASSWORD {}").format(
                    sql.Identifier(role), sql.Literal(password)
                ),
            )
        else:
            connection.execute(
                sql.SQL("ALTER ROLE {} LOGIN PASSWORD {}").format(
                    sql.Identifier(role), sql.Literal(password)
                ),
            )
    database = connection.info.dbname
    for role in ROLE_PASSWORD_FIELDS:
        connection.execute(
            sql.SQL("GRANT CONNECT ON DATABASE {} TO {}").format(
                sql.Identifier(database), sql.Identifier(role)
            )
        )
    connection.execute("REVOKE CREATE ON SCHEMA public FROM PUBLIC")


def migrate(root: str | Path, settings: PostgresSettings) -> list[dict]:
    root_path = Path(root).resolve()
    psycopg, _ = require_postgres_packages()
    dsn = settings.require("admin")
    applied: list[dict] = []
    with psycopg.connect(dsn) as connection:
        connection.execute("SELECT pg_advisory_xact_lock(hashtext('traffic_rag_migrate'))")
        connection.execute("CREATE EXTENSION IF NOT EXISTS vector")
        _bootstrap_roles(connection, settings)
        connection.execute("CREATE SCHEMA IF NOT EXISTS traffic_rag AUTHORIZATION traffic_rag_owner")
        connection.execute("GRANT USAGE ON SCHEMA traffic_rag TO traffic_rag_ingest, traffic_rag_runtime")
        connection.execute("SET ROLE traffic_rag_owner")
        connection.execute("""
            CREATE TABLE IF NOT EXISTS traffic_rag.schema_migration (
                version text PRIMARY KEY,
                checksum char(64) NOT NULL,
                applied_at timestamptz NOT NULL DEFAULT now()
            )
        """)
        for path in _migration_files(root_path):
            version = path.stem.split("_", 1)[0]
            body = path.read_text(encoding="utf-8")
            checksum = hashlib.sha256(body.encode("utf-8")).hexdigest()
            row = connection.execute(
                "SELECT checksum FROM traffic_rag.schema_migration WHERE version = %s", (version,)
            ).fetchone()
            if row:
                if row[0].strip() != checksum:
                    raise ValueError(f"Migration checksum changed after apply: {path.name}")
                applied.append({"version": version, "checksum": checksum, "status": "already_applied"})
                continue
            connection.execute(body)
            connection.execute(
                "INSERT INTO traffic_rag.schema_migration(version, checksum) VALUES (%s, %s)",
                (version, checksum),
            )
            applied.append({"version": version, "checksum": checksum, "status": "applied"})
        connection.execute("RESET ROLE")
    return applied


def migration_checksums(root: str | Path) -> dict[str, str]:
    root_path = Path(root).resolve()
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in _migration_files(root_path)
    }
