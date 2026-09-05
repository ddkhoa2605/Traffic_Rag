from __future__ import annotations

import json

from src.parser.io import write_json
from src.parser.pipeline import load_extracted_blocks, load_parsed_nodes
from src.validation.report import validate_document

from .common import document_parser, registry_from


def main() -> int:
    args = document_parser("Validate a parsed legal document").parse_args()
    registry = registry_from(args)
    report = validate_document(registry, args.document_id, load_extracted_blocks(registry, args.document_id), load_parsed_nodes(registry, args.document_id))
    write_json(registry.root / "reports" / args.document_id / "validation_report.json", report.model_dump(mode="json"))
    print(json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2))
    return 0 if report.status in {"PASS", "WARN"} else 1


if __name__ == "__main__":
    raise SystemExit(main())

