from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.retrieval_eval.dataset import (
    apply_review_sheet,
    apply_benchmark_revisions,
    create_benchmark_draft,
    export_review_context,
    validate_dataset,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Create or validate the retrieval benchmark dataset")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--create-draft", action="store_true")
    action.add_argument("--validate", action="store_true")
    action.add_argument("--apply-review", metavar="CSV")
    action.add_argument("--export-review-context", action="store_true")
    action.add_argument("--apply-revisions", metavar="YAML")
    parser.add_argument("--require-approved", action="store_true")
    parser.add_argument("--root", default=None)
    args = parser.parse_args()
    root = Path(args.root or Path.cwd())
    if args.create_draft:
        print(json.dumps(create_benchmark_draft(root), ensure_ascii=False, indent=2))
        return 0
    if args.apply_review:
        print(json.dumps(apply_review_sheet(root, args.apply_review), ensure_ascii=False, indent=2))
        return 0
    if args.export_review_context:
        print(json.dumps(export_review_context(root), ensure_ascii=False, indent=2))
        return 0
    if args.apply_revisions:
        print(json.dumps(apply_benchmark_revisions(root, args.apply_revisions), ensure_ascii=False, indent=2))
        return 0
    errors = validate_dataset(root, require_approved=args.require_approved)
    print(json.dumps({"status": "PASS" if not errors else "ERROR", "errors": errors}, ensure_ascii=False, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
