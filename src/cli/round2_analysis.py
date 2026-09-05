from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.retrieval_eval.round2 import (
    apply_round2_review,
    export_round2_review,
    finalize_round2,
    recompute_round2_metrics,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Round 2 B4a-e metric migration and review gate")
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--recompute-metrics", action="store_true")
    actions.add_argument("--export-review", action="store_true")
    actions.add_argument("--apply-review", metavar="REVIEW_CSV")
    actions.add_argument("--finalize", action="store_true")
    parser.add_argument("--split", default="dev", choices=("dev", "test"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()

    if args.recompute_metrics:
        result = recompute_round2_metrics(args.root, split=args.split)
    elif args.export_review:
        result = export_round2_review(args.root, split=args.split)
    elif args.apply_review:
        result = apply_round2_review(args.root, args.apply_review)
    else:
        result = finalize_round2(args.root, split=args.split)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
