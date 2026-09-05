from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.chunking.builder import ROUND1_STRATEGIES, ROUND2_STRATEGIES, STRATEGIES, build_passages


def main() -> int:
    parser = argparse.ArgumentParser(description="Build deterministic retrieval passages")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--strategy", choices=list(STRATEGIES))
    target.add_argument("--all", action="store_true")
    target.add_argument("--round2-all", action="store_true")
    parser.add_argument("--root", default=None)
    parser.add_argument(
        "--artifact-root",
        default="data/07_retrieval_ablation",
        help="Versioned passage artifact root, relative to project root unless absolute",
    )
    args = parser.parse_args()
    root = Path(args.root or Path.cwd()).resolve()
    selected = list(ROUND2_STRATEGIES) if args.round2_all else (list(ROUND1_STRATEGIES) if args.all else [args.strategy])
    manifests = build_passages(root, selected, artifact_root=args.artifact_root)
    print(json.dumps({key: value.model_dump(mode="json") for key, value in manifests.items()}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
