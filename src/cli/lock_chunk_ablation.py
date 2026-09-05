from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.chunking.builder import STRATEGIES
from src.retrieval_eval.locking import create_strategy_lock


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze approved dev candidates before held-out test")
    parser.add_argument("--strategy", action="append", required=True, choices=[*STRATEGIES, "B6"])
    parser.add_argument("--root", default=None)
    args = parser.parse_args()
    payload = create_strategy_lock(Path(args.root or Path.cwd()), list(dict.fromkeys(args.strategy)))
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
