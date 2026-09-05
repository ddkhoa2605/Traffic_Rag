from __future__ import annotations

import json

from src.parser.inspector import inspect_pdf

from .common import document_parser, registry_from


def main() -> int:
    args = document_parser("Inspect a registered PDF").parse_args()
    report = inspect_pdf(registry_from(args), args.document_id)
    print(json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2))
    return 0 if report.pdf_type != "BROKEN" else 1


if __name__ == "__main__":
    raise SystemExit(main())

