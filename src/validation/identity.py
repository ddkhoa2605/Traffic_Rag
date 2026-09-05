from __future__ import annotations

import re

from src.parser.models import ExtractedBlock
from src.registry.models import DocumentRecord

from .models import ValidationIssue


LAW_NUMBER = re.compile(r"(?:Luật\s+số\s*:\s*)?(\d+/\d{4}/QH\d+)", re.IGNORECASE)


def validate_identity(document: DocumentRecord, blocks: list[ExtractedBlock]) -> list[ValidationIssue]:
    sample = " ".join(block.normalized_text for block in blocks[:80])
    found = LAW_NUMBER.search(sample)
    if not found:
        return [ValidationIssue(code="DOCUMENT_NUMBER_NOT_FOUND", severity="FATAL", message="Không tìm thấy số luật trong PDF")]
    if found.group(1).upper() != document.document_number.upper():
        return [ValidationIssue(
            code="DOCUMENT_IDENTITY_MISMATCH",
            severity="FATAL",
            message=f"Registry={document.document_number}, PDF={found.group(1)}",
        )]
    return []

