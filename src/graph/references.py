from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from src.legal_tree.models import LegalNode
from src.legal_tree.resolver import LegalTreeResolver
from src.registry.loader import Registry
from .identity import graph_edge_id
from .models import GraphEdge, GraphProvenance


ARTICLE_RE = re.compile(r"\bđiều\s+(\d+[a-z]?)\b", re.IGNORECASE)
CLAUSE_RE = re.compile(r"\bkhoản\s+(\d+[a-z]?)\b", re.IGNORECASE)
POINT_RE = re.compile(r"\bđiểm\s+([a-zđ])\b", re.IGNORECASE)
POINT_LIST_RE = re.compile(
    r"\b(?:các\s+)?điểm\s+(.+?)(?=\s+khoản\b|\s+điều\b|[.;]|$)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ReferenceFinding:
    source_node_id: str
    text: str
    status: str
    target_node_ids: tuple[str, ...] = ()
    message: str | None = None


@dataclass(frozen=True)
class CorpusReferenceIntent:
    article: str | None
    clause: str | None
    points: tuple[str, ...]
    explicit_document_ids: tuple[str, ...]


@dataclass(frozen=True)
class CorpusReferenceResolution:
    status: str
    candidate_document_ids: tuple[str, ...] = ()
    resolved_node_ids: tuple[str, ...] = ()
    message: str | None = None


def _normalize(value: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", value).casefold()).strip()


def _parse_corpus_reference(text: str, registry: Registry) -> CorpusReferenceIntent:
    normalized = _normalize(text)
    article = ARTICLE_RE.search(normalized)
    clause = CLAUSE_RE.search(normalized)
    points: tuple[str, ...] = ()
    point_list = POINT_LIST_RE.search(normalized)
    if point_list:
        points = tuple(
            dict.fromkeys(
                match.casefold()
                for match in re.findall(
                    r"(?<!\w)([a-zđ])(?!\w)", point_list.group(1), re.IGNORECASE
                )
            )
        )
    elif match := POINT_RE.search(normalized):
        points = (match.group(1).casefold(),)
    explicit: list[str] = []
    for document in registry.documents.values():
        aliases = [
            document.document_number,
            document.title,
            *document.reference_aliases,
        ]
        if any(_normalize(alias) in normalized for alias in aliases if alias.strip()):
            explicit.append(document.document_id)
    return CorpusReferenceIntent(
        article=article.group(1) if article else None,
        clause=clause.group(1) if clause else None,
        points=points,
        explicit_document_ids=tuple(dict.fromkeys(explicit)),
    )


def _resolve_corpus_reference(
    intent: CorpusReferenceIntent,
    resolvers: dict[str, LegalTreeResolver],
    source_document_id: str,
) -> CorpusReferenceResolution:
    if not intent.article:
        return CorpusReferenceResolution(
            status="PARTIAL" if intent.clause or intent.points else "NOT_FOUND",
            message="Corpus reference has no Article",
        )
    if intent.points and not intent.clause:
        return CorpusReferenceResolution(
            status="PARTIAL",
            candidate_document_ids=intent.explicit_document_ids or (source_document_id,),
            message="Point reference has no containing Clause",
        )
    if len(intent.explicit_document_ids) > 1:
        return CorpusReferenceResolution(
            status="AMBIGUOUS",
            candidate_document_ids=intent.explicit_document_ids,
            message="Reference segment names multiple legal documents",
        )
    candidates = intent.explicit_document_ids or (source_document_id,)
    matches: dict[str, list[LegalNode]] = {}
    for document_id in candidates:
        resolver = resolvers.get(document_id)
        if resolver is None:
            continue
        if intent.points:
            requested = set(intent.points)
            nodes = [
                node for node in resolver.nodes
                if node.type == "point"
                and node.hierarchy.get("article") == intent.article
                and node.hierarchy.get("clause") == intent.clause
                and node.hierarchy.get("point") in requested
            ]
            if {node.hierarchy.get("point") for node in nodes} != requested:
                nodes = []
        elif intent.clause:
            nodes = [
                node for node in resolver.nodes
                if node.type == "clause"
                and node.hierarchy.get("article") == intent.article
                and node.hierarchy.get("clause") == intent.clause
            ]
        else:
            nodes = [
                node for node in resolver.nodes
                if node.type == "article"
                and node.hierarchy.get("article") == intent.article
            ]
        if nodes:
            matches[document_id] = nodes
    if not matches:
        return CorpusReferenceResolution(
            status="NOT_FOUND",
            candidate_document_ids=tuple(candidates),
            message="No canonical provision matches the corpus reference",
        )
    if len(matches) > 1:
        return CorpusReferenceResolution(
            status="AMBIGUOUS",
            candidate_document_ids=tuple(matches),
            message="Corpus reference resolves in multiple documents",
        )
    document_id, nodes = next(iter(matches.items()))
    return CorpusReferenceResolution(
        status="RESOLVED",
        candidate_document_ids=(document_id,),
        resolved_node_ids=tuple(node.id for node in nodes),
    )


def _reference_segments(text: str) -> list[tuple[int, int, str]]:
    matches = list(ARTICLE_RE.finditer(text))
    segments: list[tuple[int, int, str]] = []
    for index, match in enumerate(matches):
        previous_end = matches[index - 1].end() if index else 0
        next_start = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        left_boundary = max(previous_end, match.start() - 120)
        right_boundary = min(next_start, match.end() + 100)
        punctuation = [text.rfind(mark, left_boundary, match.start()) for mark in ".;\n"]
        start = max([left_boundary, *(value + 1 for value in punctuation if value >= 0)])
        endings = [text.find(mark, match.end(), right_boundary) for mark in ".;\n"]
        endings = [value for value in endings if value >= 0]
        end = min(endings) + 1 if endings else right_boundary
        segment = text[start:end].strip()
        if segment:
            actual_start = text.find(segment, start, end)
            segments.append((actual_start, actual_start + len(segment), segment))
    return segments


def extract_reference_edges(
    nodes: list[LegalNode],
    registry: Registry,
    resolvers: dict[str, LegalTreeResolver],
) -> tuple[list[GraphEdge], list[ReferenceFinding]]:
    edges: list[GraphEdge] = []
    findings: list[ReferenceFinding] = []
    seen: set[str] = set()
    for node in nodes:
        if not node.text.strip():
            continue
        for start, end, segment in _reference_segments(node.text):
            intent = _parse_corpus_reference(segment, registry)
            resolution = _resolve_corpus_reference(
                intent, resolvers, node.document_id
            )
            targets = tuple(resolution.resolved_node_ids)
            findings.append(
                ReferenceFinding(
                    source_node_id=node.id,
                    text=segment,
                    status=resolution.status,
                    target_node_ids=targets,
                    message=resolution.message,
                )
            )
            if resolution.status != "RESOLVED":
                continue
            provenance = GraphProvenance(
                canonical_node_id=node.id,
                evidence_text=segment,
                start_offset=start,
                end_offset=end,
            )
            for target in targets:
                if target == node.id:
                    continue
                edge_id = graph_edge_id(node.id, "REFERENCES", target, [provenance])
                if edge_id in seen:
                    continue
                seen.add(edge_id)
                edges.append(
                    GraphEdge(
                        edge_id=edge_id,
                        source_node_id=node.id,
                        target_node_id=target,
                        relation_type="REFERENCES",
                        description=segment,
                        provenance=[provenance],
                    )
                )
    return edges, findings
