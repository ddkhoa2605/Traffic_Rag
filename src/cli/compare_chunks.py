from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.retrieval_eval.reports import compare_runs


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare completed chunking runs")
    parser.add_argument("--split", default="dev", choices=["dev", "test"])
    parser.add_argument("--root", default=None)
    args = parser.parse_args()
    try:
        result = compare_runs(Path(args.root or Path.cwd()), split=args.split)
    except FileNotFoundError as exc:
        print(json.dumps({"status": "ERROR", "error": str(exc)}, ensure_ascii=False, indent=2))
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
