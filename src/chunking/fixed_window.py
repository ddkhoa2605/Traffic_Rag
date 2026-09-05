from __future__ import annotations

from dataclasses import dataclass

from src.legal_tree.models import LegalDocument, LegalNode
from src.legal_tree.resolver import LegalTreeResolver

from .base import ChunkStrategy
from .common import DEPTH, make_passage


@dataclass(frozen=True)
class TextSpan:
    node: LegalNode
    start: int
    end: int


def flatten_direct_text(nodes: tuple[LegalNode, ...]) -> tuple[str, list[TextSpan]]:
    parts: list[str] = []
    spans: list[TextSpan] = []
    cursor = 0
    for node in nodes:
        text = node.text.strip()
        if not text:
            continue
        if parts:
            parts.append("\n")
            cursor += 1
        start = cursor
        parts.append(text)
        cursor += len(text)
        spans.append(TextSpan(node=node, start=start, end=cursor))
    return "".join(parts), spans


class FixedWindowStrategy(ChunkStrategy):
    name = "B0_fixed_window"

    def build(self, document: LegalDocument):
        resolver = LegalTreeResolver(document.nodes)
        text, spans = flatten_direct_text(resolver.nodes)
        encoded = self.token_counter.encode_with_offsets(text)
        window = int(self.config.get("window_tokens", 400))
        overlap = int(self.config.get("overlap_tokens", 80))
        if window <= 0 or overlap < 0 or overlap >= window:
            raise ValueError("Fixed-window config requires 0 <= overlap < window")
        passages = []
        step = window - overlap
        for ordinal, token_start in enumerate(range(0, len(encoded.token_ids), step), start=1):
            token_end = min(token_start + window, len(encoded.token_ids))
            offsets = encoded.offsets[token_start:token_end]
            valid = [pair for pair in offsets if pair[1] > pair[0]]
            if not valid:
                continue
            char_start, char_end = valid[0][0], valid[-1][1]
            window_text = text[char_start:char_end]
            overlapping: list[tuple[int, LegalNode]] = []
            for span in spans:
                count = sum(1 for start, end in valid if end > span.start and start < span.end)
                if count:
                    overlapping.append((count, span.node))
            if not overlapping:
                continue
            order = {node.id: index for index, node in enumerate(resolver.nodes)}
            primary = sorted(overlapping, key=lambda item: (-item[0], -DEPTH.get(item[1].type, 0), order[item[1].id]))[0][1]
            source_nodes = [node for _, node in overlapping]
            citations = [node for node in source_nodes if node.type in {"article", "clause", "point", "subpoint", "unknown_block"}]
            citations = citations or [primary]
            passages.append(make_passage(
                strategy=self.name,
                passage_id=f"B0__{document.document_id}__W{ordinal:04d}",
                primary=primary, source_nodes=source_nodes, index_nodes=[primary],
                context_nodes=[node for node in source_nodes if node.id != primary.id],
                citation_nodes=citations, index_text=window_text,
                evidence_text=window_text, counter=self.token_counter,
            ))
            if token_end == len(encoded.token_ids):
                break
        return passages
