from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from .models import ExtractedBlock, LegalNode, NodeSource, SourceBlockRef
from .normalizer import normalize_text


def content_hash(text: str) -> str:
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


def stable_node_id(
    document_id: str,
    node_type: str,
    *,
    chapter: str | None = None,
    section: str | None = None,
    article: str | None = None,
    clause: str | None = None,
    point: str | None = None,
    source_page: int | None = None,
    source_block: str | None = None,
    segment: int = 0,
) -> str:
    if node_type == "document":
        return document_id
    if node_type == "chapter":
        return f"{document_id}__CH{chapter}"
    if node_type == "section":
        return f"{document_id}__CH{chapter}__S{section}"
    if node_type == "article":
        return f"{document_id}__A{article}"
    if node_type == "clause":
        return f"{document_id}__A{article}__C{clause}"
    if node_type == "point":
        return f"{document_id}__A{article}__C{clause}__P{point}"
    if source_page is None or source_block is None:
        raise ValueError(f"Source-based node type {node_type} requires page and block")
    type_key = node_type.upper()
    block_key = "".join(char if char.isalnum() else "_" for char in source_block).strip("_")
    return f"{document_id}__{type_key}__P{source_page:03d}__B{block_key}__S{segment:02d}"


@dataclass
class MutableNode:
    id: str
    document_id: str
    type: str
    hierarchy: dict[str, str | None]
    title: str | None
    parent_id: str | None
    source_file_id: str
    children_ids: list[str] = field(default_factory=list)
    blocks: list[ExtractedBlock] = field(default_factory=list)

    def append(self, block: ExtractedBlock) -> None:
        self.blocks.append(block)

    def freeze(self) -> LegalNode:
        text = "\n".join(block.normalized_text for block in self.blocks).strip()
        source = None
        if self.blocks:
            source = NodeSource(
                source_file_id=self.source_file_id,
                page_start=min(block.page for block in self.blocks),
                page_end=max(block.page for block in self.blocks),
                blocks=[SourceBlockRef(block_id=block.block_id, page=block.page, bbox=block.bbox) for block in self.blocks],
            )
        return LegalNode(
            id=self.id,
            document_id=self.document_id,
            type=self.type,
            hierarchy=self.hierarchy,
            title=self.title,
            text=text,
            parent_id=self.parent_id,
            children_ids=self.children_ids,
            source=source,
            content_hash=content_hash(text),
        )
