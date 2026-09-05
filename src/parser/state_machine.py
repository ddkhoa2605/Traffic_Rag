from __future__ import annotations

from dataclasses import dataclass

from .detectors import Detection, detect
from .models import ExtractedBlock, LegalNode
from .normalizer import normalize_text
from .node_builder import MutableNode, stable_node_id


@dataclass
class ParseResult:
    nodes: list[LegalNode]
    issues: list[dict]


class LegalStructureParser:
    def __init__(self, document_id: str, source_file_id: str):
        self.document_id = document_id
        self.source_file_id = source_file_id
        self.nodes: list[MutableNode] = []
        self.by_id: dict[str, MutableNode] = {}
        self.issues: list[dict] = []
        self.current_chapter: MutableNode | None = None
        self.current_section: MutableNode | None = None
        self.current_article: MutableNode | None = None
        self.current_clause: MutableNode | None = None
        self.current_point: MutableNode | None = None
        self.document = self._open("document", {}, None, None)

    def _open(self, node_type: str, hierarchy: dict[str, str | None], title: str | None, parent: MutableNode | None) -> MutableNode:
        node_id = stable_node_id(self.document_id, node_type, **hierarchy)
        if node_id in self.by_id:
            self.issues.append({"code": "DUPLICATE_NODE_ID", "severity": "ERROR", "node_id": node_id})
            node_id = f"{node_id}__DUP{sum(key.startswith(node_id) for key in self.by_id) + 1}"
        node = MutableNode(
            id=node_id,
            document_id=self.document_id,
            type=node_type,
            hierarchy={
                "chapter": hierarchy.get("chapter"),
                "section": hierarchy.get("section"),
                "article": hierarchy.get("article"),
                "clause": hierarchy.get("clause"),
                "point": hierarchy.get("point"),
            },
            title=title,
            parent_id=parent.id if parent else None,
            source_file_id=self.source_file_id,
        )
        self.nodes.append(node)
        self.by_id[node.id] = node
        if parent:
            parent.children_ids.append(node.id)
        return node

    def _hierarchy(self, **updates) -> dict[str, str | None]:
        values = {
            "chapter": self.current_chapter.hierarchy["chapter"] if self.current_chapter else None,
            "section": self.current_section.hierarchy["section"] if self.current_section else None,
            "article": self.current_article.hierarchy["article"] if self.current_article else None,
            "clause": self.current_clause.hierarchy["clause"] if self.current_clause else None,
            "point": self.current_point.hierarchy["point"] if self.current_point else None,
        }
        values.update(updates)
        return values

    def _open_detected(self, detection: Detection, block: ExtractedBlock) -> MutableNode:
        if detection.kind == "chapter":
            hierarchy = {"chapter": detection.key}
            node = self._open("chapter", hierarchy, detection.title, self.document)
            self.current_chapter, self.current_section = node, None
            self.current_article = self.current_clause = self.current_point = None
        elif detection.kind == "section":
            parent = self.current_chapter or self.document
            hierarchy = self._hierarchy(section=detection.key, article=None, clause=None, point=None)
            node = self._open("section", hierarchy, detection.title, parent)
            self.current_section = node
            self.current_article = self.current_clause = self.current_point = None
        elif detection.kind == "article":
            parent = self.current_section or self.current_chapter or self.document
            hierarchy = self._hierarchy(article=detection.key, clause=None, point=None)
            node = self._open("article", hierarchy, detection.title, parent)
            self.current_article = node
            self.current_clause = self.current_point = None
        elif detection.kind == "clause":
            parent = self.current_article
            hierarchy = self._hierarchy(clause=detection.key, point=None)
            if parent is None:
                self.issues.append({"code": "ORPHAN_CLAUSE", "severity": "ERROR", "page": block.page, "block_id": block.block_id})
                parent = self.document
            node = self._open("clause", hierarchy, None, parent)
            self.current_clause = node
            self.current_point = None
        else:
            parent = self.current_clause
            hierarchy = self._hierarchy(point=detection.key)
            if parent is None:
                self.issues.append({"code": "ORPHAN_POINT", "severity": "ERROR", "page": block.page, "block_id": block.block_id})
                parent = self.current_article or self.document
            node = self._open("point", hierarchy, None, parent)
            self.current_point = node
        node.append(block)
        return node

    def _continuation_target(self) -> MutableNode:
        return self.current_point or self.current_clause or self.current_article or self.current_section or self.current_chapter or self.document

    @staticmethod
    def _logical_segments(block: ExtractedBlock) -> list[ExtractedBlock]:
        """Split a physical PDF block when a new legal marker starts on a line."""
        if not block.lines:
            return [block]
        groups: list[list] = []
        current: list = []
        excluded = set(block.excluded_line_indexes)
        for line_index, line in enumerate(block.lines):
            if line_index in excluded:
                continue
            line_text = normalize_text("".join(span.text for span in line.spans))
            if not line_text:
                continue
            if detect(line_text) is not None and current:
                groups.append(current)
                current = []
            current.append((line, line_text))
        if current:
            groups.append(current)
        if len(groups) <= 1:
            return [block]

        segments: list[ExtractedBlock] = []
        for group in groups:
            lines = [item[0] for item in group]
            x0 = min(line.bbox[0] for line in lines)
            y0 = min(line.bbox[1] for line in lines)
            x1 = max(line.bbox[2] for line in lines)
            y1 = max(line.bbox[3] for line in lines)
            segments.append(block.model_copy(update={
                "bbox": (x0, y0, x1, y1),
                "raw_text": "\n".join(item[1] for item in group),
                "normalized_text": normalize_text("\n".join(item[1] for item in group)),
                "lines": lines,
                "excluded_line_indexes": [],
            }))
        return segments

    @staticmethod
    def _detect_segment(block: ExtractedBlock) -> Detection | None:
        result = detect(block.normalized_text)
        if result is None or result.kind not in {"chapter", "section", "article"}:
            return result
        if not block.lines:
            return result
        first_line = next(
            (line for line in block.lines if normalize_text("".join(span.text for span in line.spans))),
            block.lines[0],
        )
        total = bold = 0
        for span in first_line.spans:
            length = len(span.text.strip())
            total += length
            if span.font_flags & 16 or "bold" in span.font.casefold():
                bold += length
        # Isolated cross-references such as "Điều 24 của Luật này" are not headings.
        return result if total and bold / total >= 0.5 else None

    def parse(self, blocks: list[ExtractedBlock]) -> ParseResult:
        for block in blocks:
            if not block.include_in_legal_text or not block.normalized_text:
                continue
            for segment in self._logical_segments(block):
                detection = self._detect_segment(segment)
                if detection:
                    self._open_detected(detection, segment)
                else:
                    self._continuation_target().append(segment)
        return ParseResult(nodes=[node.freeze() for node in self.nodes], issues=self.issues)


def parse_blocks(document_id: str, source_file_id: str, blocks: list[ExtractedBlock]) -> ParseResult:
    return LegalStructureParser(document_id, source_file_id).parse(blocks)
