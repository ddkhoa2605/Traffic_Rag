from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable

from src.legal_tree.models import LegalNode
from src.legal_tree.resolver import LegalTreeResolver
from src.registry.loader import Registry

from .models import LegalReferenceIntent, ReferenceResolution, ReferenceSpan


POINT_ORDER = tuple("abcdđefghiklmnopqrstuvxy")
ARTICLE_RE = re.compile(r"\bđiều\s+(\d+[a-z]?)\b", re.IGNORECASE)
CLAUSE_RE = re.compile(r"\bkhoản\s+(\d+[a-z]?)\b", re.IGNORECASE)
POINT_START_RE = re.compile(r"\bđiểm\s+([a-zđ])\b", re.IGNORECASE)


def normalize_reference_text(value: str) -> str:
    value = unicodedata.normalize("NFC", value).casefold()
    value = value.replace("–", "-").replace("—", "-")
    return re.sub(r"\s+", " ", value).strip()


def _span(kind: str, match: re.Match[str]) -> ReferenceSpan:
    return ReferenceSpan(kind=kind, text=match.group(0), start=match.start(), end=match.end())


def _expand_range(start: str, end: str) -> list[str]:
    if start not in POINT_ORDER or end not in POINT_ORDER:
        return [start, end]
    left, right = POINT_ORDER.index(start), POINT_ORDER.index(end)
    if left > right:
        return [start, end]
    return list(POINT_ORDER[left:right + 1])


def _parse_points(normalized: str) -> tuple[list[str], list[ReferenceSpan]]:
    starts = list(POINT_START_RE.finditer(normalized))
    if not starts:
        return [], []
    first = starts[0]
    end_candidates = [
        match.start()
        for pattern in (CLAUSE_RE, ARTICLE_RE)
        for match in pattern.finditer(normalized, first.end())
    ]
    segment_end = min(end_candidates) if end_candidates else len(normalized)
    segment = normalized[first.start():segment_end]
    labels = re.findall(r"(?<!\w)([a-zđ])(?!\w)", segment)
    range_match = re.search(
        r"\bđiểm\s+([a-zđ])\s+(?:đến|tới)\s+(?:điểm\s+)?([a-zđ])\b",
        segment,
    )
    if range_match:
        labels = _expand_range(range_match.group(1), range_match.group(2))
    labels = list(dict.fromkeys(label.casefold() for label in labels))
    return labels, [_span("point", match) for match in starts]


def _document_aliases(registry: Registry) -> list[tuple[str, str, str]]:
    aliases: list[tuple[str, str, str]] = []
    for document in registry.documents.values():
        values = [document.document_number, document.title, *document.reference_aliases]
        for value in values:
            normalized = normalize_reference_text(value)
            if normalized:
                aliases.append((normalized, document.document_id, value))
    return sorted(set(aliases), key=lambda item: (-len(item[0]), item[1], item[0]))


def parse_legal_reference(query: str, registry: Registry) -> LegalReferenceIntent:
    normalized = normalize_reference_text(query)
    article_match = ARTICLE_RE.search(normalized)
    clause_match = CLAUSE_RE.search(normalized)
    points, point_spans = _parse_points(normalized)
    matched_aliases: list[tuple[str, str]] = []
    alias_spans: list[ReferenceSpan] = []
    occupied: list[tuple[int, int]] = []
    for alias, document_id, original in _document_aliases(registry):
        start = normalized.find(alias)
        if start < 0:
            continue
        end = start + len(alias)
        if any(start < old_end and end > old_start for old_start, old_end in occupied):
            continue
        occupied.append((start, end))
        matched_aliases.append((document_id, original))
        alias_spans.append(ReferenceSpan(kind="document", text=normalized[start:end], start=start, end=end))
    explicit_ids = list(dict.fromkeys(item[0] for item in matched_aliases))
    explicit_alias = matched_aliases[0][1] if len(explicit_ids) == 1 else None
    spans = [
        *([_span("article", article_match)] if article_match else []),
        *([_span("clause", clause_match)] if clause_match else []),
        *point_spans,
        *alias_spans,
    ]
    spans.sort(key=lambda item: (item.start, item.end, item.kind))
    return LegalReferenceIntent(
        article=article_match.group(1) if article_match else None,
        clause=clause_match.group(1) if clause_match else None,
        points=points,
        explicit_document_alias=explicit_alias,
        explicit_document_ids=explicit_ids,
        matched_spans=spans,
    )


def _matching_nodes(resolver: LegalTreeResolver, intent: LegalReferenceIntent) -> list[LegalNode]:
    if intent.points:
        if not intent.article or not intent.clause:
            return []
        by_point = {
            node.hierarchy.get("point"): node
            for node in resolver.nodes
            if node.type == "point"
            and node.hierarchy.get("article") == intent.article
            and node.hierarchy.get("clause") == intent.clause
        }
        if not all(point in by_point for point in intent.points):
            return []
        requested = set(intent.points)
        return [
            node for node in resolver.nodes
            if node.type == "point"
            and node.hierarchy.get("article") == intent.article
            and node.hierarchy.get("clause") == intent.clause
            and node.hierarchy.get("point") in requested
        ]
    if intent.clause:
        if not intent.article:
            return []
        return [
            node for node in resolver.nodes
            if node.type == "clause"
            and node.hierarchy.get("article") == intent.article
            and node.hierarchy.get("clause") == intent.clause
        ]
    if intent.article:
        return [
            node for node in resolver.nodes
            if node.type == "article" and node.hierarchy.get("article") == intent.article
        ]
    return []


def resolve_legal_reference(
    intent: LegalReferenceIntent,
    resolvers: dict[str, LegalTreeResolver],
    document_ids: Iterable[str] | None = None,
) -> ReferenceResolution:
    requested = list(dict.fromkeys(document_ids or []))
    unknown = [document_id for document_id in requested if document_id not in resolvers]
    if unknown:
        raise ValueError(f"Unknown document_ids: {unknown}")
    if not any((intent.article, intent.clause, intent.points)):
        return ReferenceResolution(status="NO_REFERENCE")
    if len(intent.explicit_document_ids) > 1:
        return ReferenceResolution(
            status="CONFLICT",
            candidate_document_ids=intent.explicit_document_ids,
            message="Query contains aliases for multiple documents",
        )
    explicit = intent.explicit_document_ids
    if explicit and requested and explicit[0] not in requested:
        return ReferenceResolution(
            status="CONFLICT",
            candidate_document_ids=list(dict.fromkeys([*explicit, *requested])),
            message="Explicit document conflicts with caller document scope",
        )
    if (intent.points and (not intent.article or not intent.clause)) or (intent.clause and not intent.article):
        return ReferenceResolution(
            status="PARTIAL",
            candidate_document_ids=explicit or requested,
            message="Clause/Point references require their containing Article",
        )
    candidates = explicit or requested or list(resolvers)
    matches: dict[str, list[LegalNode]] = {}
    for document_id in candidates:
        nodes = _matching_nodes(resolvers[document_id], intent)
        if nodes:
            matches[document_id] = nodes
    if not matches:
        return ReferenceResolution(
            status="NOT_FOUND", candidate_document_ids=candidates,
            message="No canonical node matches the parsed hierarchy",
        )
    if len(matches) > 1:
        return ReferenceResolution(
            status="AMBIGUOUS", candidate_document_ids=list(matches),
            message="Reference exists in multiple documents; provide document_ids or a law alias",
        )
    document_id, nodes = next(iter(matches.items()))
    return ReferenceResolution(
        status="RESOLVED",
        candidate_document_ids=[document_id],
        resolved_node_ids=[node.id for node in nodes],
    )
