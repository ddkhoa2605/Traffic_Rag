from __future__ import annotations

import hashlib
import shutil
from datetime import datetime, timezone

from src import __version__
from src.parser.io import write_json, write_jsonl
from src.parser.pipeline import extract_document, parse_document
from src.registry.loader import Registry
from src.registry.validator import validate_registry
from src.validation.report import validate_document


def build_document(
    registry: Registry,
    document_id: str,
    *,
    render_pages: bool = True,
    approve_warnings: bool = False,
) -> dict:
    registry_issues = validate_registry(registry)
    if registry_issues:
        raise ValueError("Registry validation failed: " + "; ".join(f"{item.code}: {item.message}" for item in registry_issues))
    _, blocks, extraction_report = extract_document(registry, document_id, render_pages=render_pages)
    parsed = parse_document(registry, document_id, blocks)
    report = validate_document(registry, document_id, blocks, parsed.nodes)
    report_dir = registry.root / "reports" / document_id
    write_json(report_dir / "validation_report.json", report.model_dump(mode="json"))

    digest_input = "\n".join(f"{node.id}:{node.content_hash}" for node in parsed.nodes)
    canonical_digest = hashlib.sha256(digest_input.encode("utf-8")).hexdigest()
    publish = report.status == "PASS" or (report.status == "WARN" and approve_warnings)
    effective_status = "PASS" if report.status == "WARN" and approve_warnings else report.status
    manifest = {
        "document_id": document_id,
        "source_sha256": registry.source_for(document_id).sha256,
        "parser_version": __version__,
        "registry_version": registry.version,
        "build_timestamp": datetime.now(timezone.utc).isoformat(),
        "validation_status": effective_status,
        "raw_validation_status": report.status,
        "manual_warning_approval": bool(report.status == "WARN" and approve_warnings),
        "canonical_digest": canonical_digest,
        "published": publish,
    }
    write_json(report_dir / "build_manifest.json", manifest)
    write_jsonl(report_dir / "regression_snapshot.jsonl", (
        {
            "node_id": node.id,
            "node_type": node.type,
            "parent_id": node.parent_id,
            "page_start": node.source.page_start if node.source else None,
            "page_end": node.source.page_end if node.source else None,
            "content_hash": node.content_hash,
        }
        for node in parsed.nodes
    ))

    parsed_dir = registry.root / "data" / "04_parsed" / document_id
    target_root = registry.root / ("data/05_validated" if publish else "data/99_quarantine") / document_id
    target_root.mkdir(parents=True, exist_ok=True)
    for filename in ("document.json", "nodes.jsonl", "parse_report.json"):
        shutil.copy2(parsed_dir / filename, target_root / filename)
    write_json(target_root / "validation_report.json", report.model_dump(mode="json"))
    write_json(target_root / "build_manifest.json", manifest)
    return {"inspection": extraction_report, "validation": report.model_dump(mode="json"), "manifest": manifest}
