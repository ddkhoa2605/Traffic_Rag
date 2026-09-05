from __future__ import annotations

from pathlib import Path

from src.validation.models import ValidationIssue

from .base import VisionResult, VisionVerifier


def verify_flagged_pages(
    verifier: VisionVerifier,
    issues: list[ValidationIssue],
    image_dir: Path,
) -> list[VisionResult]:
    """Verify only flagged pages; caller must send disagreements to manual review."""
    results: list[VisionResult] = []
    seen: set[tuple[int, str]] = set()
    for issue in issues:
        if not issue.vision_review_required or issue.page is None:
            continue
        key = (issue.page, issue.code)
        if key in seen:
            continue
        seen.add(key)
        image_path = image_dir / f"page_{issue.page:03d}.png"
        results.append(verifier.inspect_page(str(image_path), issue.code, context=issue.model_dump(mode="json")))
    return results
