from __future__ import annotations

from src.parser.models import LegalNode

from .models import ValidationIssue


def validate_provenance(nodes: list[LegalNode]) -> tuple[list[ValidationIssue], float]:
    issues: list[ValidationIssue] = []
    covered = 0
    for node in nodes:
        valid = bool(node.source and node.source.source_file_id and node.source.blocks)
        if valid:
            valid = all(block.block_id and block.page > 0 and len(block.bbox) == 4 for block in node.source.blocks)
        if valid:
            covered += 1
        else:
            issues.append(ValidationIssue(code="PROVENANCE_MISSING", severity="ERROR", message="Node has incomplete provenance", node_id=node.id))
    return issues, covered / len(nodes) if nodes else 0.0

