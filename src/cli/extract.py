from __future__ import annotations

import json

from src.parser.pipeline import extract_document

from .common import document_parser, registry_from


def main() -> int:
    parser = document_parser("Render and extract a registered PDF")
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()
    _, _, report = extract_document(registry_from(args), args.document_id, render_pages=not args.no_render, dpi=args.dpi)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

