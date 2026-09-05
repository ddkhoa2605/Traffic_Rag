from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.retrieval_eval.release import (
    create_release_candidate_lock,
    export_release_v2_review,
    validate_release_v2_review,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a versioned retrieval release candidate lock")
    parser.add_argument("--root", default=None)
    parser.add_argument("--export-review", action="store_true")
    parser.add_argument("--validate-review")
    parser.add_argument("--artifact-root")
    parser.add_argument("--report-root")
    parser.add_argument("--run-version")
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.validate_review:
        payload = validate_release_v2_review(
            Path(args.root or Path.cwd()), args.validate_review
        )
    elif args.export_review:
        payload = export_release_v2_review(Path(args.root or Path.cwd()))
    else:
        missing = [name for name in ("artifact_root", "report_root", "run_version", "output") if getattr(args, name) is None]
        if missing:
            parser.error("required unless --export-review: " + ", ".join("--" + name.replace("_", "-") for name in missing))
        payload = create_release_candidate_lock(
            Path(args.root or Path.cwd()),
            artifact_root=args.artifact_root,
            report_root=args.report_root,
            run_version=args.run_version,
            output_path=args.output,
        )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
