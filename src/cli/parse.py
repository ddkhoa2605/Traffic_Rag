from __future__ import annotations

import json
from collections import Counter

from src.parser.pipeline import parse_document

from .common import document_parser, registry_from


def main() -> int:
    args = document_parser("Parse extracted blocks into a legal tree").parse_args()
    result = parse_document(registry_from(args), args.document_id)
    print(json.dumps({"nodes": len(result.nodes), "counts": Counter(node.type for node in result.nodes), "issues": result.issues}, ensure_ascii=False, indent=2))
    return 0 if not result.issues else 1


if __name__ == "__main__":
    raise SystemExit(main())

