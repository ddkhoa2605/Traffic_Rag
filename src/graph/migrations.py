from __future__ import annotations

import hashlib
from pathlib import Path

from src.postgres_store.config import require_postgres_packages

from .config import GraphSettings


def _files(root: Path) -> list[Path]:
    paths = sorted((root / "sql/graph_migrations").glob("[0-9][0-9][0-9][0-9]_*.sql"))
    if not paths:
        raise FileNotFoundError("No legal graph migrations found")
    return paths


def init_database(root: str | Path, settings: GraphSettings) -> list[dict]:
    root_path = Path(root).resolve()
    psycopg, _ = require_postgres_packages()
    sql = psycopg.sql
    password = settings.require("app_password")
    with psycopg.connect(settings.require("admin_dsn"), autocommit=True, connect_timeout=5) as connection:
        role = connection.execute(
            "SELECT 1 FROM pg_roles WHERE rolname = %s", (settings.app_role,)
        ).fetchone()
        if role:
            connection.execute(
                sql.SQL("ALTER ROLE {} LOGIN PASSWORD {}").format(
                    sql.Identifier(settings.app_role), sql.Literal(password)
                )
            )
        else:
            connection.execute(
                sql.SQL("CREATE ROLE {} LOGIN PASSWORD {}").format(
                    sql.Identifier(settings.app_role), sql.Literal(password)
                )
            )
        database = connection.execute(
            "SELECT 1 FROM pg_database WHERE datname = %s", (settings.database,)
        ).fetchone()
        if not database:
            connection.execute(
                sql.SQL("CREATE DATABASE {} OWNER {} TEMPLATE template0 ENCODING 'UTF8'").format(
                    sql.Identifier(settings.database), sql.Identifier(settings.app_role)
                )
            )
        else:
            connection.execute(
                sql.SQL("ALTER DATABASE {} OWNER TO {}").format(
                    sql.Identifier(settings.database), sql.Identifier(settings.app_role)
                )
            )

    admin_values = psycopg.conninfo.conninfo_to_dict(settings.require("admin_dsn"))
    admin_values["dbname"] = settings.database
    graph_admin_dsn = psycopg.conninfo.make_conninfo(**admin_values)
    with psycopg.connect(graph_admin_dsn, autocommit=True, connect_timeout=5) as connection:
        connection.execute("CREATE EXTENSION IF NOT EXISTS vector")

    applied: list[dict] = []
    with psycopg.connect(settings.require("app_dsn"), connect_timeout=5) as connection:
        connection.execute("SELECT pg_advisory_xact_lock(hashtext('traffic_rag_lightrag_migrate'))")
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS legal_graph_schema_migration (
                version text PRIMARY KEY,
                checksum char(64) NOT NULL,
                applied_at timestamptz NOT NULL DEFAULT now()
            )
            """
        )
        for path in _files(root_path):
            version = path.stem.split("_", 1)[0]
            body = path.read_text(encoding="utf-8")
            checksum = hashlib.sha256(body.encode("utf-8")).hexdigest()
            row = connection.execute(
                "SELECT checksum FROM legal_graph_schema_migration WHERE version = %s",
                (version,),
            ).fetchone()
            if row:
                if row[0].strip() != checksum:
                    raise ValueError(f"Graph migration checksum changed: {path.name}")
                status = "already_applied"
            else:
                connection.execute(body)
                connection.execute(
                    "INSERT INTO legal_graph_schema_migration(version, checksum) VALUES (%s, %s)",
                    (version, checksum),
                )
                status = "applied"
            applied.append({"version": version, "checksum": checksum, "status": status})
    return applied


def migration_checksums(root: str | Path) -> dict[str, str]:
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in _files(Path(root).resolve())}
