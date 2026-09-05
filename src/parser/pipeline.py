from __future__ import annotations

from collections import Counter

from src.registry.loader import Registry

from .inspector import inspect_pdf
from .io import read_jsonl, write_json, write_jsonl
from .models import ExtractedBlock, LegalNode
from .pdf_native import extract_native
from .state_machine import ParseResult, parse_blocks


def extract_document(registry: Registry, document_id: str, *, render_pages: bool = True, dpi: int = 150):
    report = inspect_pdf(registry, document_id)
    if report.pdf_type == "BROKEN":
        raise ValueError(f"Broken PDF: {document_id}")
    pages, blocks = extract_native(registry, document_id, render_pages=render_pages, dpi=dpi)
    output = registry.root / "data" / "03_extracted" / document_id
    write_jsonl(output / "pages.jsonl", pages)
    write_jsonl(output / "blocks.jsonl", (block.model_dump(mode="json") for block in blocks))
    extraction_report = report.model_dump(mode="json") | {
        "extracted_blocks": len(blocks),
        "included_blocks": sum(block.include_in_legal_text for block in blocks),
        "excluded_by_classification": dict(Counter(block.classification for block in blocks if not block.include_in_legal_text)),
    }
    write_json(output / "extraction_report.json", extraction_report)
    return pages, blocks, extraction_report


def load_extracted_blocks(registry: Registry, document_id: str) -> list[ExtractedBlock]:
    path = registry.root / "data" / "03_extracted" / document_id / "blocks.jsonl"
    return [ExtractedBlock.model_validate(item) for item in read_jsonl(path)]


def parse_document(registry: Registry, document_id: str, blocks: list[ExtractedBlock] | None = None) -> ParseResult:
    blocks = blocks or load_extracted_blocks(registry, document_id)
    source = registry.source_for(document_id)
    result = parse_blocks(document_id, source.source_file_id, blocks)
    output = registry.root / "data" / "04_parsed" / document_id
    write_jsonl(output / "nodes.jsonl", (node.model_dump(mode="json") for node in result.nodes))
    counts = Counter(node.type for node in result.nodes)
    report = {
        "document_id": document_id,
        "node_count": len(result.nodes),
        "counts": dict(counts),
        "issues": result.issues,
    }
    write_json(output / "parse_report.json", report)
    write_json(output / "document.json", {
        "document_id": document_id,
        "root_id": document_id,
        "node_ids": [node.id for node in result.nodes],
    })
    return result


def load_parsed_nodes(registry: Registry, document_id: str) -> list[LegalNode]:
    path = registry.root / "data" / "04_parsed" / document_id / "nodes.jsonl"
    return [LegalNode.model_validate(item) for item in read_jsonl(path)]
