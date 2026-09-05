from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.retrieval_runtime import PostgresB6Application, application_health


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Query the PostgreSQL B6/B7 application backend")
    parser.add_argument("query", nargs="?", help="Raw Vietnamese legal query")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument(
        "--document-id", action="append", dest="document_ids",
        help="Optional caller document scope; may be repeated",
    )
    parser.add_argument("--health", action="store_true", help="Check backend without loading BGE-M3")
    parser.add_argument("--compact", action="store_true", help="Omit evidence text from JSON output")
    parser.add_argument("--root", default=None)
    args = parser.parse_args()
    root = Path(args.root or Path.cwd()).resolve()
    if args.health:
        payload = application_health(root)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0 if payload.get("ready") else 1
    if not args.query:
        parser.error("query is required unless --health is used")
    with PostgresB6Application(root) as application:
        payload = application.search(
            args.query, top_k=args.top_k, document_ids=args.document_ids,
        ).model_dump(mode="json")
    if args.compact:
        for result in payload["results"]:
            result.pop("evidence_text", None)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
