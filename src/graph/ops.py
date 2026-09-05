from __future__ import annotations

import os
import shutil
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path

from src.parser.io import write_json
from src.postgres_store.config import require_postgres_packages

from .config import GraphSettings
from .repository import connect_graph, validate_database


def _binary(name: str) -> str:
    found = shutil.which(name)
    if found:
        return found
    windows = Path("C:/Program Files/PostgreSQL/18/bin") / f"{name}.exe"
    if windows.is_file():
        return str(windows)
    raise FileNotFoundError(f"PostgreSQL client binary not found: {name}")


def _args(dsn: str) -> tuple[list[str], dict[str, str], str]:
    psycopg, _ = require_postgres_packages()
    values = psycopg.conninfo.conninfo_to_dict(dsn)
    database = values.get("dbname")
    if not database:
        raise ValueError("DSN must include dbname")
    args: list[str] = []
    for option, key in (("--host", "host"), ("--port", "port"), ("--username", "user")):
        if values.get(key):
            args.extend((option, values[key]))
    environment = dict(os.environ)
    if values.get("password"):
        environment["PGPASSWORD"] = values["password"]
    return args, environment, database


def backup_graph(root: str | Path, settings: GraphSettings, output: str | Path | None = None) -> dict:
    root_path = Path(root).resolve()
    args, environment, database = _args(settings.require("app_dsn"))
    if output is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_path = root_path / "reports/graph/backups" / f"{database}_{stamp}.dump"
    else:
        output_path = Path(output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [_binary("pg_dump"), *args, "--format=custom", "--file", str(output_path), database],
        check=True,
        env=environment,
        capture_output=True,
        text=True,
    )
    result = {
        "database": database,
        "output": str(output_path),
        "size_bytes": output_path.stat().st_size,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(root_path / "reports/graph/backup_report.json", result)
    return result


def restore_graph_check(
    root: str | Path,
    settings: GraphSettings,
    backup_path: str | Path,
    release_id: str,
) -> dict:
    root_path = Path(root).resolve()
    source = Path(backup_path).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    psycopg, _ = require_postgres_packages()
    sql = psycopg.sql
    admin_dsn = settings.require("admin_dsn")
    args, environment, _ = _args(admin_dsn)
    temporary = f"traffic_rag_lightrag_restore_{uuid.uuid4().hex[:10]}"
    created = False
    try:
        with psycopg.connect(admin_dsn, autocommit=True, connect_timeout=5) as connection:
            connection.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(temporary)))
            created = True
        subprocess.run(
            [
                _binary("pg_restore"),
                *args,
                "--no-owner",
                "--no-privileges",
                "--dbname",
                temporary,
                str(source),
            ],
            check=True,
            env=environment,
            capture_output=True,
            text=True,
        )
        values = psycopg.conninfo.conninfo_to_dict(admin_dsn)
        values["dbname"] = temporary
        restored_dsn = psycopg.conninfo.make_conninfo(**values)
        restored = GraphSettings(
            admin_dsn=admin_dsn,
            app_dsn=restored_dsn,
            app_password=settings.app_password,
        )
        with connect_graph(restored) as connection:
            validation = validate_database(connection, release_id)
        result = {
            "backup": str(source),
            "restore_database": temporary,
            "release_id": release_id,
            "passed": validation["valid"],
            "validation": validation,
            "checked_at": datetime.now(timezone.utc).isoformat(),
        }
        write_json(root_path / "reports/graph/restore_check_report.json", result)
        if not result["passed"]:
            raise ValueError("Restored graph database failed validation")
        return result
    finally:
        if created:
            with psycopg.connect(admin_dsn, autocommit=True, connect_timeout=5) as connection:
                connection.execute(
                    sql.SQL("DROP DATABASE {} WITH (FORCE)").format(sql.Identifier(temporary))
                )
