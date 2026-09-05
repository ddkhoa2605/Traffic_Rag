from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.postgres_store.artifacts import load_frozen_release
from src.postgres_store.config import PostgresSettings
from src.postgres_store.migrations import migrate
from src.postgres_store.ops import backup_database, restore_check
from src.postgres_store.repository import (
    activate_release,
    ingest_frozen_release,
    preflight,
    validate_database,
)
from src.postgres_store.shadow import shadow_compare


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PostgreSQL/pgvector shadow store for frozen B6")
    parser.add_argument("--root", default=None, help="Repository root; defaults to current directory")
    parser.add_argument("--dsn", default=None, help="Override the command's PostgreSQL DSN")
    parser.add_argument("--release-lock", default="reports/chunk_ablation/strategy_lock.json")
    parser.add_argument("--artifact-root", default="data/07_retrieval_ablation")
    parser.add_argument("--release-name", default=None)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("preflight", "migrate", "ingest-frozen", "validate", "shadow-compare", "activate", "backup"):
        commands.add_parser(name)
    restore = commands.add_parser("restore-check")
    restore.add_argument("backup_path")
    backup = commands.choices["backup"]
    backup.add_argument("--output", default=None)
    commands.choices["activate"].add_argument("--dataset-id", default=None)
    return parser


def main() -> int:
    args = _parser().parse_args()
    root = Path(args.root or Path.cwd()).resolve()
    settings = PostgresSettings.load(root, dsn=args.dsn)
    release = None
    if args.command in {"ingest-frozen", "validate", "shadow-compare", "activate"}:
        release = load_frozen_release(
            root,
            lock_path=args.release_lock,
            artifact_root=args.artifact_root,
            release_name=args.release_name,
        )

    if args.command == "preflight":
        result = preflight(settings)
    elif args.command == "migrate":
        result = {"migrations": migrate(root, settings), "preflight": preflight(settings)}
    elif args.command == "ingest-frozen":
        result = ingest_frozen_release(root, settings, release)
    elif args.command == "validate":
        result = validate_database(settings, release)
        if not result["valid"]:
            raise ValueError("; ".join(result["errors"]))
    elif args.command == "shadow-compare":
        result = shadow_compare(root, settings, release)
    elif args.command == "activate":
        result = activate_release(settings, args.dataset_id or release.dataset_id)
    elif args.command == "backup":
        result = backup_database(root, args.dsn or settings.require("admin"), args.output)
    elif args.command == "restore-check":
        result = restore_check(root, args.dsn or settings.require("admin"), args.backup_path)
    else:
        raise AssertionError(args.command)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
