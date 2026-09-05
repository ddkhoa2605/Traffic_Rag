from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.retrieval_eval.b6_analysis import analyze_b6_dev, apply_b6_review, export_b6_review, finalize_b6


def main() -> None:
    parser = argparse.ArgumentParser(description="B6 one-shot analysis, review gate and winner freeze")
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--analyze", action="store_true")
    actions.add_argument("--export-review", action="store_true")
    actions.add_argument("--apply-review", metavar="REVIEW_CSV")
    actions.add_argument("--finalize", action="store_true")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    if args.analyze: result = analyze_b6_dev(args.root)
    elif args.export_review: result = export_b6_review(args.root)
    elif args.apply_review: result = apply_b6_review(args.root, args.apply_review)
    else: result = finalize_b6(args.root)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
