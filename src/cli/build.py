from __future__ import annotations

import argparse
import json

from src.build import build_document
from src.registry.loader import load_registry


def main() -> int:
    parser = argparse.ArgumentParser(description="Build canonical traffic-law corpus")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("document_id", nargs="?", choices=["LAW_35_2024", "LAW_36_2024"])
    target.add_argument("--all", action="store_true")
    parser.add_argument("--root", default=None)
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--approve-warnings", action="store_true", help="Publish WARN builds after explicit manual approval")
    args = parser.parse_args()
    registry = load_registry(args.root)
    ids = list(registry.documents) if args.all else [args.document_id]
    results = {}
    exit_code = 0
    for document_id in ids:
        result = build_document(registry, document_id, render_pages=not args.no_render, approve_warnings=args.approve_warnings)
        results[document_id] = result["manifest"]
        if not result["manifest"]["published"]:
            exit_code = 1
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
