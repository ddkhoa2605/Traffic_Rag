from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _dotenv(root: Path) -> dict[str, str]:
    path = root / ".env"
    if not path.is_file():
        return {}
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


@dataclass(frozen=True)
class PostgresSettings:
    admin_dsn: str | None
    owner_dsn: str | None
    ingest_dsn: str | None
    runtime_dsn: str | None
    owner_password: str | None
    ingest_password: str | None
    runtime_password: str | None

    @classmethod
    def load(cls, root: str | Path, *, dsn: str | None = None) -> "PostgresSettings":
        root_path = Path(root).resolve()
        local = _dotenv(root_path)

        def get(name: str) -> str | None:
            return os.environ.get(name) or local.get(name)

        return cls(
            admin_dsn=dsn or get("TRAFFIC_RAG_ADMIN_DSN"),
            owner_dsn=get("TRAFFIC_RAG_OWNER_DSN"),
            ingest_dsn=dsn or get("TRAFFIC_RAG_INGEST_DSN"),
            runtime_dsn=dsn or get("TRAFFIC_RAG_RUNTIME_DSN"),
            owner_password=get("POSTGRES_OWNER_PASSWORD"),
            ingest_password=get("POSTGRES_INGEST_PASSWORD"),
            runtime_password=get("POSTGRES_RUNTIME_PASSWORD"),
        )

    def require(self, purpose: str) -> str:
        value = {
            "admin": self.admin_dsn,
            "owner": self.owner_dsn,
            "ingest": self.ingest_dsn,
            "runtime": self.runtime_dsn,
        }.get(purpose)
        if not value:
            raise ValueError(f"Missing PostgreSQL {purpose} DSN; configure .env or pass --dsn")
        return value


def require_postgres_packages():
    try:
        import psycopg
        from pgvector.psycopg import register_vector
    except ImportError as exc:
        raise RuntimeError('PostgreSQL support requires: python -m pip install -e ".[postgres]"') from exc
    return psycopg, register_vector


def connect(dsn: str, *, autocommit: bool = False, register_types: bool = True):
    psycopg, register_vector = require_postgres_packages()
    connection = psycopg.connect(dsn, autocommit=autocommit)
    if register_types:
        register_vector(connection)
    return connection
