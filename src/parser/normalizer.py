from __future__ import annotations

import re
import unicodedata
from collections import Counter

from .models import ExtractedBlock


_SPACE = re.compile(r"[\t\u00a0\u2000-\u200b\u202f\u205f\u3000]+")
_REPEATED_SPACE = re.compile(r" {2,}")
_PAGE_NUMBER = re.compile(r"^\s*[-–—]?\s*\d{1,4}\s*[-–—]?\s*$")


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text.replace("\u00ad", ""))
    lines = []
    for line in text.splitlines():
        line = _REPEATED_SPACE.sub(" ", _SPACE.sub(" ", line)).strip()
        if line:
            lines.append(line)
    return " ".join(lines).strip()


def _frequency_key(text: str) -> str:
    text = normalize_text(text).casefold()
    text = re.sub(r"\d+", "#", text)
    return re.sub(r"\s+", " ", text)


def classify_marginal_blocks(blocks: list[ExtractedBlock], page_heights: dict[int, float]) -> None:
    """Remove repeated marginal *lines* while preserving their containing raw block.

    Công Báo PDFs sometimes merge the running header and the first continuation
    line into one PyMuPDF block. Classifying only at block level would delete law
    text, so canonical normalized_text is rebuilt from retained lines while
    raw_text and all span geometry remain untouched.
    """
    page_count = len(page_heights)
    candidates: Counter[str] = Counter()
    for block in blocks:
        height = page_heights[block.page]
        for line in block.lines:
            line_text = "".join(span.text for span in line.spans)
            top, bottom = line.bbox[1], line.bbox[3]
            if top <= height * 0.08 or bottom >= height * 0.92:
                candidates[_frequency_key(line_text)] += 1

    threshold = max(2, int(page_count * 0.60 + 0.999))
    repeated = {key for key, count in candidates.items() if count >= threshold and key}
    for block in blocks:
        retained: list[str] = []
        removed_types: list[str] = []
        height = page_heights[block.page]
        block.excluded_line_indexes = []
        for line_index, line in enumerate(block.lines):
            line_text = normalize_text("".join(span.text for span in line.spans))
            if not line_text:
                continue
            top, bottom = line.bbox[1], line.bbox[3]
            marginal = top <= height * 0.08 or bottom >= height * 0.92
            key = _frequency_key(line_text)
            if marginal and _PAGE_NUMBER.fullmatch(line_text):
                removed_types.append("page_number")
                block.excluded_line_indexes.append(line_index)
            elif marginal and key in repeated:
                removed_types.append("header" if top <= height * 0.08 else "footer")
                block.excluded_line_indexes.append(line_index)
            elif top <= height * 0.08 and line_text.casefold().startswith("người ký:"):
                removed_types.append("signature")
                block.excluded_line_indexes.append(line_index)
            else:
                retained.append(line_text)
        block.normalized_text = normalize_text("\n".join(retained))
        if not block.normalized_text:
            if removed_types:
                # Prefer a semantic type over page_number when a merged marginal block exists.
                block.classification = next((value for value in removed_types if value != "page_number"), removed_types[0])
            else:
                block.classification = "empty"
            block.include_in_legal_text = False
