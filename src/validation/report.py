from __future__ import annotations

from collections import Counter

import yaml

from src.parser.models import ExtractedBlock, LegalNode
from src.registry.loader import Registry

from .content import validate_content
from .identity import validate_identity
from .models import ValidationIssue, ValidationReport
from .provenance import validate_provenance
from .sequence import validate_sequences
from .structure import validate_structure


def _status(issues: list[ValidationIssue]) -> str:
    severities = {issue.severity for issue in issues}
    for value in ("FATAL", "ERROR", "WARN"):
        if value in severities:
            return value
    return "PASS"


def validate_document(registry: Registry, document_id: str, blocks: list[ExtractedBlock], nodes: list[LegalNode]) -> ValidationReport:
    issues: list[ValidationIssue] = []
    issues.extend(validate_identity(registry.document(document_id), blocks))
    issues.extend(validate_structure(nodes))
    issues.extend(validate_sequences(nodes))
    content_issues, retention = validate_content(blocks, nodes)
    provenance_issues, provenance = validate_provenance(nodes)
    issues.extend(content_issues)
    issues.extend(provenance_issues)
    counts = Counter(node.type for node in nodes)
    metrics: dict[str, float | int | str] = {
        "node_count": len(nodes),
        "article_count": counts["article"],
        "clause_count": counts["clause"],
        "point_count": counts["point"],
        "text_retention": round(retention, 6),
        "provenance_coverage": round(provenance, 6),
    }
    expected_path = registry.root / "data" / "06_gold" / "expected_counts.yaml"
    if expected_path.is_file():
        with expected_path.open(encoding="utf-8") as handle:
            expected_all = yaml.safe_load(handle) or {}
        expected = expected_all.get(document_id, {})
        mapping = {"chapters": "chapter", "articles": "article", "clauses": "clause", "points": "point"}
        for expected_key, node_type in mapping.items():
            if expected_key in expected and counts[node_type] != expected[expected_key]:
                issues.append(ValidationIssue(
                    code=f"{node_type.upper()}_COUNT_MISMATCH",
                    severity="ERROR",
                    message=f"Expected {expected[expected_key]}, got {counts[node_type]}",
                ))
    return ValidationReport(document_id=document_id, status=_status(issues), metrics=metrics, issues=issues)
