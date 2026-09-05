from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal


NodeKind = Literal["chapter", "section", "article", "clause", "point"]


@dataclass(frozen=True)
class Detection:
    kind: NodeKind
    key: str
    title: str | None


CHAPTER_RE = re.compile(r"^Chương\s+([IVXLCDM]+|\d+)\b\s*[.:]?\s*(.*)$", re.IGNORECASE)
SECTION_RE = re.compile(r"^Mục\s+(\d+)\b\s*[.:]?\s*(.*)$", re.IGNORECASE)
ARTICLE_RE = re.compile(r"^Điều\s+(\d+[A-Za-z]?)\s*[.]?\s*(.*)$", re.IGNORECASE)
CLAUSE_RE = re.compile(r"^(\d{1,3})\.\s+(.+)$", re.DOTALL)
POINT_RE = re.compile(r"^([a-zA-ZđĐ])\)\s+(.+)$", re.DOTALL)


def detect(text: str) -> Detection | None:
    text = text.strip()
    for kind, pattern in (
        ("chapter", CHAPTER_RE),
        ("section", SECTION_RE),
        ("article", ARTICLE_RE),
        ("clause", CLAUSE_RE),
        ("point", POINT_RE),
    ):
        match = pattern.match(text)
        if match:
            key = match.group(1)
            if kind == "chapter":
                key = key.upper()
            elif kind == "point":
                key = key.lower().replace("Đ", "đ")
            title = match.group(2).strip() or None
            return Detection(kind=kind, key=key, title=title)
    return None


def is_article(text: str) -> bool:
    result = detect(text)
    return result is not None and result.kind == "article"


def is_clause(text: str) -> bool:
    result = detect(text)
    return result is not None and result.kind == "clause"


def is_point(text: str) -> bool:
    result = detect(text)
    return result is not None and result.kind == "point"

