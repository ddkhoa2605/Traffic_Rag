from __future__ import annotations

from collections import defaultdict

from src.parser.models import LegalNode

from .models import ValidationIssue


# Vietnamese legislative enumeration omits f, j, w and z.
POINT_ORDER = list("abcdđeghiklmnopqrstuvxy")


def _numeric_gap(nodes: list[LegalNode], code: str) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    values = []
    for node in nodes:
        key = node.hierarchy[node.type]
        if key and key.isdigit():
            values.append((int(key), node))
    for (previous, _), (current, node) in zip(values, values[1:]):
        if current != previous + 1:
            issues.append(ValidationIssue(
                code=code,
                severity="WARN",
                message=f"Sequence jumps from {previous} to {current}",
                page=node.source.page_start if node.source else None,
                node_id=node.id,
                vision_review_required=True,
            ))
    return issues


def validate_sequences(nodes: list[LegalNode]) -> list[ValidationIssue]:
    issues = _numeric_gap([node for node in nodes if node.type == "article"], "ARTICLE_SEQUENCE_GAP")
    clauses: dict[str, list[LegalNode]] = defaultdict(list)
    points: dict[str, list[LegalNode]] = defaultdict(list)
    for node in nodes:
        if node.type == "clause" and node.parent_id:
            clauses[node.parent_id].append(node)
        elif node.type == "point" and node.parent_id:
            points[node.parent_id].append(node)
    for group in clauses.values():
        issues.extend(_numeric_gap(group, "CLAUSE_SEQUENCE_GAP"))
    order = {letter: index for index, letter in enumerate(POINT_ORDER)}
    for group in points.values():
        comparable = [(order[node.hierarchy["point"]], node) for node in group if node.hierarchy["point"] in order]
        for (previous, _), (current, node) in zip(comparable, comparable[1:]):
            if current != previous + 1:
                issues.append(ValidationIssue(
                    code="POINT_SEQUENCE_GAP",
                    severity="WARN",
                    message=f"Point sequence jumps from {POINT_ORDER[previous]} to {POINT_ORDER[current]}",
                    page=node.source.page_start if node.source else None,
                    node_id=node.id,
                    vision_review_required=True,
                ))
    return issues
