from __future__ import annotations

from src.parser.models import LegalNode

from .models import ValidationIssue


def validate_structure(nodes: list[LegalNode]) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    by_id = {node.id: node for node in nodes}
    if len(by_id) != len(nodes):
        issues.append(ValidationIssue(code="DUPLICATE_NODE_ID", severity="ERROR", message="Node IDs are not unique"))
    for node in nodes:
        if node.type == "unknown_block":
            issues.append(ValidationIssue(
                code="UNKNOWN_STRUCTURE", severity="WARN",
                message="Unrecognized structure was preserved", node_id=node.id,
                page=node.source.page_start if node.source else None,
            ))
        if "__DUP" in node.id:
            issues.append(ValidationIssue(code="DUPLICATE_STABLE_ID", severity="ERROR", message=node.id, node_id=node.id))
        if node.type == "document":
            if node.parent_id is not None:
                issues.append(ValidationIssue(code="DOCUMENT_HAS_PARENT", severity="ERROR", message=node.id, node_id=node.id))
            continue
        parent = by_id.get(node.parent_id or "")
        if parent is None:
            issues.append(ValidationIssue(code=f"ORPHAN_{node.type.upper()}", severity="ERROR", message="Parent not found", node_id=node.id))
            continue
        expected = {"clause": "article", "point": "clause"}.get(node.type)
        if expected and parent.type != expected:
            issues.append(ValidationIssue(
                code=f"INVALID_{node.type.upper()}_PARENT",
                severity="ERROR",
                message=f"Expected {expected}, got {parent.type}",
                node_id=node.id,
            ))
        if node.id not in parent.children_ids:
            issues.append(ValidationIssue(code="PARENT_CHILD_MISMATCH", severity="ERROR", message=node.id, node_id=node.id))
    return issues
