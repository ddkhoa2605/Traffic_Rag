from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from src.legal_tree.models import BBox, LegalNode, NodeSource, SourceBlockRef


class Span(BaseModel):
    text: str
    bbox: BBox
    font: str
    font_size: float
    font_flags: int


class Line(BaseModel):
    bbox: BBox
    spans: list[Span]


class BlockStyle(BaseModel):
    font_size_max: float = 0.0
    bold_ratio: float = 0.0


class ExtractedBlock(BaseModel):
    block_id: str
    document_id: str
    source_file_id: str
    page: int
    block: int
    bbox: BBox
    type: Literal["text", "image"] = "text"
    raw_text: str
    normalized_text: str
    lines: list[Line] = Field(default_factory=list)
    style: BlockStyle = Field(default_factory=BlockStyle)
    excluded_line_indexes: list[int] = Field(default_factory=list)
    classification: Literal["body", "header", "footer", "page_number", "signature", "empty"] = "body"
    include_in_legal_text: bool = True


class InspectionReport(BaseModel):
    document_id: str
    source_file_id: str
    page_count: int
    text_pages: int
    text_pages_ratio: float
    average_chars_per_page: float
    image_pages: int
    pdf_type: Literal["DIGITAL", "MIXED", "SCANNED", "BROKEN"]
    recommended_route: Literal["native", "hybrid", "vision", "quarantine"]
