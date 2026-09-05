from __future__ import annotations

from pathlib import Path

import pymupdf

from src.registry.loader import Registry

from .models import InspectionReport


def inspect_pdf(registry: Registry, document_id: str) -> InspectionReport:
    source = registry.source_for(document_id)
    path = registry.source_path(document_id)
    try:
        pdf = pymupdf.open(path)
        page_count = pdf.page_count
        char_counts: list[int] = []
        image_pages = 0
        for page in pdf:
            char_counts.append(len(page.get_text().strip()))
            if page.get_images(full=True):
                image_pages += 1
    except Exception:
        return InspectionReport(
            document_id=document_id,
            source_file_id=source.source_file_id,
            page_count=0,
            text_pages=0,
            text_pages_ratio=0,
            average_chars_per_page=0,
            image_pages=0,
            pdf_type="BROKEN",
            recommended_route="quarantine",
        )
    finally:
        if "pdf" in locals():
            pdf.close()

    text_pages = sum(count >= 100 for count in char_counts)
    ratio = text_pages / page_count if page_count else 0.0
    average = sum(char_counts) / page_count if page_count else 0.0
    if ratio >= 0.95 and average >= 500:
        pdf_type, route = "DIGITAL", "native"
    elif ratio >= 0.20:
        pdf_type, route = "MIXED", "hybrid"
    else:
        pdf_type, route = "SCANNED", "vision"
    return InspectionReport(
        document_id=document_id,
        source_file_id=source.source_file_id,
        page_count=page_count,
        text_pages=text_pages,
        text_pages_ratio=round(ratio, 6),
        average_chars_per_page=round(average, 2),
        image_pages=image_pages,
        pdf_type=pdf_type,
        recommended_route=route,
    )

