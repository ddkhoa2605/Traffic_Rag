from __future__ import annotations

import os
import shutil
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path

from src.parser.io import write_json

from .config import connect, require_postgres_packages
from .repository import REPORT_DIR


def _postgres_binary(name: str) -> str:
    found = shutil.which(name)
    if found:
        return found
    windows = Path("C:/Program Files/PostgreSQL/18/bin") / f"{name}.exe"
    if windows.is_file():
        return str(windows)
    raise FileNotFoundError(f"PostgreSQL client binary not found: {name}")


def _connection_args(dsn: str) -> tuple[list[str], dict[str, str]]:
    psycopg, _ = require_postgres_packages()
    values = psycopg.conninfo.conninfo_to_dict(dsn)
    args = []
    for option, key in (("--host", "host"), ("--port", "port"), ("--username", "user")):
        if values.get(key):
            args += [option, values[key]]
    environment = dict(os.environ)
    if values.get("password"):
        environment["PGPASSWORD"] = values["password"]
    return args, environment


def backup_database(root: str | Path, dsn: str, output: str | Path | None = None) -> dict:
    root_path = Path(root).resolve()
    psycopg, _ = require_postgres_packages()
    values = psycopg.conninfo.conninfo_to_dict(dsn)
    database = values.get("dbname")
    if not database:
        raise ValueError("Backup DSN must include a database name")
    if output is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_path = root_path / REPORT_DIR / "backups" / f"traffic_rag_{stamp}.dump"
    else:
        output_path = Path(output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    args, environment = _connection_args(dsn)
    command = [_postgres_binary("pg_dump"), *args, "--format=custom", "--file", str(output_path), database]
    subprocess.run(command, check=True, env=environment, capture_output=True, text=True)
    result = {
        "database": database,
        "output": str(output_path),
        "size_bytes": output_path.stat().st_size,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(root_path / REPORT_DIR / "backup_report.json", result)
    return result


def restore_check(root: str | Path, admin_dsn: str, backup_path: str | Path) -> dict:
    root_path = Path(root).resolve()
    source = Path(backup_path).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    psycopg, _ = require_postgres_packages()
    sql = psycopg.sql
    values = psycopg.conninfo.conninfo_to_dict(admin_dsn)
    temporary_database = f"traffic_rag_restore_check_{uuid.uuid4().hex[:12]}"
    if not temporary_database.startswith("traffic_rag_restore_check_"):
        raise ValueError("Unsafe restore-check database name")
    args, environment = _connection_args(admin_dsn)
    created = False
    try:
        with connect(admin_dsn, autocommit=True, register_types=False) as connection:
            connection.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(temporary_database)))
            created = True
        subprocess.run(
            [_postgres_binary("pg_restore"), *args, "--no-owner", "--no-privileges",
             "--dbname", temporary_database, str(source)],
            check=True, env=environment, capture_output=True, text=True,
        )
        restored_values = dict(values)
        restored_values["dbname"] = temporary_database
        restored_dsn = psycopg.conninfo.make_conninfo(**restored_values)
        with connect(restored_dsn, register_types=False) as connection:
            counts = connection.execute("""
                SELECT
                  (SELECT count(*) FROM traffic_rag.retrieval_dataset),
                  (SELECT count(*) FROM traffic_rag.legal_node),
                  (SELECT count(*) FROM traffic_rag.retrieval_passage),
                  (SELECT count(*) FROM traffic_rag.passage_embedding)
            """).fetchone()
        result = {
            "backup": str(source),
            "restore_database": temporary_database,
            "dataset_count": counts[0], "node_count": counts[1],
            "passage_count": counts[2], "embedding_count": counts[3],
            "passed": counts[0] > 0 and counts[1] > 0 and counts[2] == counts[3],
            "checked_at": datetime.now(timezone.utc).isoformat(),
        }
        write_json(root_path / REPORT_DIR / "restore_check_report.json", result)
        if not result["passed"]:
            raise ValueError("Restored database failed count validation")
        return result
    finally:
        if created:
            with connect(admin_dsn, autocommit=True, register_types=False) as connection:
                connection.execute(
                    sql.SQL("DROP DATABASE {} WITH (FORCE)").format(sql.Identifier(temporary_database))
                )
