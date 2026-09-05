from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import yaml

from src.parser.io import write_json
from src.registry.loader import Registry

from .base import VisionVerifier


def run_benchmark(registry: Registry, verifier: VisionVerifier | None = None) -> dict:
    """Run the configured 20-page verifier benchmark without changing corpus text."""
    config_path = registry.root / "data/06_gold/vision_benchmark_pages.yaml"
    with config_path.open(encoding="utf-8") as handle:
        pages = yaml.safe_load(handle)
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "page_count": sum(len(value) for value in pages.values()),
        "pages": pages,
        "native": {"status": "AVAILABLE", "canonical_source": True},
        "vision": {"status": "NOT_CONFIGURED" if verifier is None else "PENDING"},
        "hybrid": {"status": "NATIVE_ONLY" if verifier is None else "PENDING", "automatic_overwrite": False},
        "results": [],
    }
    if verifier is not None:
        for document_id, selected_pages in pages.items():
            image_dir = registry.root / "data/02_pages" / document_id
            for page in selected_pages:
                result = verifier.inspect_page(str(image_dir / f"page_{page:03d}.png"), "VERIFY_STRUCTURE")
                report["results"].append({"document_id": document_id, **result.model_dump(mode="json")})
        report["vision"]["status"] = "COMPLETE"
        report["hybrid"]["status"] = "REVIEW_REQUIRED"
    write_json(registry.root / "reports/vision_benchmark.json", report)
    return report
