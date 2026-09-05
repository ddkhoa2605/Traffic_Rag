from __future__ import annotations

from pathlib import Path

import pymupdf

from src.registry.loader import Registry

from .models import BlockStyle, ExtractedBlock, Line, Span
from .normalizer import classify_marginal_blocks, normalize_text


def _bbox(value) -> tuple[float, float, float, float]:
    return tuple(round(float(number), 3) for number in value)


def extract_native(
    registry: Registry,
    document_id: str,
    *,
    render_pages: bool = True,
    dpi: int = 150,
) -> tuple[list[dict], list[ExtractedBlock]]:
    source = registry.source_for(document_id)
    output_images = registry.root / "data" / "02_pages" / document_id
    if render_pages:
        output_images.mkdir(parents=True, exist_ok=True)
    pages: list[dict] = []
    blocks: list[ExtractedBlock] = []
    page_heights: dict[int, float] = {}

    with pymupdf.open(registry.source_path(document_id)) as pdf:
        for page_index, page in enumerate(pdf):
            page_number = page_index + 1
            page_heights[page_number] = float(page.rect.height)
            image_path = output_images / f"page_{page_number:03d}.png"
            if render_pages and not image_path.exists():
                page.get_pixmap(dpi=dpi, alpha=False).save(image_path)
            data = page.get_text("dict", sort=True)
            page_blocks = 0
            page_chars = 0
            for block_index, raw_block in enumerate(data.get("blocks", [])):
                if raw_block.get("type") != 0:
                    continue
                parsed_lines: list[Line] = []
                text_lines: list[str] = []
                weighted_bold = 0
                total_chars = 0
                sizes: list[float] = []
                for raw_line in raw_block.get("lines", []):
                    spans: list[Span] = []
                    line_text = ""
                    for raw_span in raw_line.get("spans", []):
                        value = raw_span.get("text", "")
                        flags = int(raw_span.get("flags", 0))
                        font = str(raw_span.get("font", ""))
                        length = len(value)
                        is_bold = bool(flags & 16) or "bold" in font.casefold()
                        weighted_bold += length if is_bold else 0
                        total_chars += length
                        size = float(raw_span.get("size", 0))
                        sizes.append(size)
                        spans.append(Span(text=value, bbox=_bbox(raw_span["bbox"]), font=font, font_size=size, font_flags=flags))
                        line_text += value
                    parsed_lines.append(Line(bbox=_bbox(raw_line["bbox"]), spans=spans))
                    text_lines.append(line_text)
                raw_text = "\n".join(text_lines)
                normalized = normalize_text(raw_text)
                page_chars += len(normalized)
                page_blocks += 1
                short_doc_id = document_id.replace("LAW_", "LAW").replace("_2024", "")
                blocks.append(ExtractedBlock(
                    block_id=f"{short_doc_id}_P{page_number:03d}_B{block_index:03d}",
                    document_id=document_id,
                    source_file_id=source.source_file_id,
                    page=page_number,
                    block=block_index,
                    bbox=_bbox(raw_block["bbox"]),
                    raw_text=raw_text,
                    normalized_text=normalized,
                    lines=parsed_lines,
                    style=BlockStyle(
                        font_size_max=round(max(sizes, default=0), 3),
                        bold_ratio=round(weighted_bold / total_chars, 4) if total_chars else 0,
                    ),
                ))
            pages.append({
                "document_id": document_id,
                "page": page_number,
                "width": round(float(page.rect.width), 3),
                "height": round(float(page.rect.height), 3),
                "text_chars": page_chars,
                "block_count": page_blocks,
                "image_path": image_path.relative_to(registry.root).as_posix() if render_pages else None,
            })
    classify_marginal_blocks(blocks, page_heights)
    return pages, blocks
