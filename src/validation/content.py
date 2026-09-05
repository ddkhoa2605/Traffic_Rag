from __future__ import annotations

import re

from src.parser.detectors import detect
from src.parser.models import ExtractedBlock, LegalNode

from .models import ValidationIssue


_SPACE = re.compile(r"\s+")
_TRUNCATED = re.compile(r"(?:\bvà|\bhoặc|[,;])$", re.IGNORECASE)


def _meaningful_length(value: str) -> int:
    return len(_SPACE.sub("", value))


def validate_content(blocks: list[ExtractedBlock], nodes: list[LegalNode]) -> tuple[list[ValidationIssue], float]:
    issues: list[ValidationIssue] = []
    source_text = "".join(block.normalized_text for block in blocks if block.include_in_legal_text)
    canonical_text = "".join(node.text for node in nodes)
    source_length = _meaningful_length(source_text)
    canonical_length = _meaningful_length(canonical_text)
    retention = min(canonical_length / source_length, 1.0) if source_length else 0.0
    if retention < 0.995:
        issues.append(ValidationIssue(
            code="TEXT_RETENTION_LOW",
            severity="ERROR",
            message=f"Text retention {retention:.4%} is below 99.5%",
            vision_review_required=True,
        ))
    block_by_id = {block.block_id: block for block in blocks}
    included = [block for block in blocks if block.include_in_legal_text]
    position = {block.block_id: index for index, block in enumerate(included)}
    page_bottom = {}
    for block in included:
        page_bottom[block.page] = max(page_bottom.get(block.page, 0), block.bbox[3])
    for node in nodes:
        if "�" in node.text:
            issues.append(ValidationIssue(code="UNICODE_REPLACEMENT_CHARACTER", severity="ERROR", message="Canonical text contains U+FFFD", node_id=node.id))
        suspicious = False
        if node.source and node.source.blocks and _TRUNCATED.search(node.text.strip()):
            last_ref = node.source.blocks[-1]
            last_block = block_by_id.get(last_ref.block_id)
            index = position.get(last_ref.block_id)
            if last_block is not None and index is not None and index + 1 < len(included):
                following = included[index + 1]
                at_page_bottom = last_block.bbox[3] >= page_bottom[last_block.page] * 0.98
                continuation_next_page = following.page == last_block.page + 1 and detect(following.normalized_text) is None
                suspicious = at_page_bottom and continuation_next_page
        if suspicious:
            issues.append(ValidationIssue(
                code="SUSPICIOUS_TRUNCATION",
                severity="WARN",
                message="Node ends at a page break with a continuation token",
                page=node.source.page_end if node.source else None,
                node_id=node.id,
                vision_review_required=True,
            ))
    return issues, retention
