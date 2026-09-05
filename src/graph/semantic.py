from __future__ import annotations

from dataclasses import dataclass

from .identity import graph_edge_id, semantic_node_id
from .models import (
    ArticleExtraction,
    GraphEdge,
    GraphNode,
    GraphProvenance,
    GraphSourceDocument,
)


DEFAULT_ENTITY_TYPES = {
    "Actor",
    "Vehicle",
    "Infrastructure",
    "Action",
    "Violation",
    "Sanction",
    "TimeConstraint",
    "LegalConcept",
    "Condition",
    "Exception",
}


@dataclass(frozen=True)
class SemanticBuildResult:
    nodes: tuple[GraphNode, ...]
    edges: tuple[GraphEdge, ...]
    rejected: tuple[dict, ...]


def _provenance(
    source: GraphSourceDocument,
    values,
    *,
    owner: str,
) -> tuple[list[GraphProvenance], list[dict]]:
    result: list[GraphProvenance] = []
    rejected: list[dict] = []
    allowed = set(source.source_node_ids)
    for value in values:
        if value.canonical_node_id not in allowed:
            rejected.append(
                {"owner": owner, "reason": "CROSS_ARTICLE_PROVENANCE", "value": value.model_dump()}
            )
            continue
        text = source.node_texts[value.canonical_node_id]
        start = text.find(value.evidence_text)
        if start < 0:
            rejected.append(
                {"owner": owner, "reason": "EVIDENCE_NOT_IN_CANONICAL_TEXT", "value": value.model_dump()}
            )
            continue
        result.append(
            GraphProvenance(
                canonical_node_id=value.canonical_node_id,
                evidence_text=value.evidence_text,
                start_offset=start,
                end_offset=start + len(value.evidence_text),
            )
        )
    return result, rejected


def build_semantic_graph(
    source: GraphSourceDocument,
    extraction: ArticleExtraction,
    *,
    extraction_version: str,
    entity_types: set[str] | None = None,
) -> SemanticBuildResult:
    entity_types = entity_types or DEFAULT_ENTITY_TYPES
    nodes: list[GraphNode] = []
    edges: list[GraphEdge] = []
    rejected: list[dict] = []
    local_map: dict[str, str] = {}

    for entity in extraction.entities:
        if entity.entity_type not in entity_types:
            rejected.append(
                {"owner": entity.local_key, "reason": "UNKNOWN_ENTITY_TYPE", "value": entity.model_dump()}
            )
            continue
        provenance, failures = _provenance(source, entity.provenance, owner=entity.local_key)
        rejected.extend(failures)
        if not provenance:
            rejected.append(
                {"owner": entity.local_key, "reason": "NO_VALID_PROVENANCE", "value": entity.model_dump()}
            )
            continue
        node_id = semantic_node_id(
            source.canonical_document_id, entity.entity_type, entity.label
        )
        if entity.local_key in local_map and local_map[entity.local_key] != node_id:
            rejected.append(
                {"owner": entity.local_key, "reason": "DUPLICATE_LOCAL_KEY", "value": entity.model_dump()}
            )
            continue
        local_map[entity.local_key] = node_id
        nodes.append(
            GraphNode(
                graph_node_id=node_id,
                node_kind="semantic",
                node_type=entity.entity_type,
                label=entity.label,
                document_id=source.canonical_document_id,
                properties={
                    "description": entity.description,
                    "aliases": list(dict.fromkeys(entity.aliases)),
                    "source_node_ids": list(dict.fromkeys(p.canonical_node_id for p in provenance)),
                    "provenance": [item.model_dump(mode="json") for item in provenance],
                },
            )
        )

    allowed_canonical = set(source.source_node_ids)
    for relation in extraction.relationships:
        if relation.relation_type == "OTHER":
            rejected.append(
                {
                    "owner": f"{relation.source_ref}->{relation.target_ref}",
                    "reason": "OTHER_RELATION",
                    "value": relation.model_dump(mode="json"),
                }
            )
            continue
        source_id = local_map.get(relation.source_ref, relation.source_ref)
        target_id = local_map.get(relation.target_ref, relation.target_ref)
        if source_id not in local_map.values() and source_id not in allowed_canonical:
            rejected.append(
                {"owner": relation.source_ref, "reason": "UNKNOWN_RELATION_SOURCE", "value": relation.model_dump()}
            )
            continue
        if target_id not in local_map.values() and target_id not in allowed_canonical:
            rejected.append(
                {"owner": relation.target_ref, "reason": "UNKNOWN_RELATION_TARGET", "value": relation.model_dump()}
            )
            continue
        if source_id == target_id:
            rejected.append(
                {"owner": relation.source_ref, "reason": "SELF_LOOP", "value": relation.model_dump()}
            )
            continue
        provenance, failures = _provenance(
            source, relation.provenance, owner=f"{relation.source_ref}->{relation.target_ref}"
        )
        rejected.extend(failures)
        if not provenance:
            rejected.append(
                {"owner": relation.source_ref, "reason": "NO_VALID_PROVENANCE", "value": relation.model_dump()}
            )
            continue
        edges.append(
            GraphEdge(
                edge_id=graph_edge_id(
                    source_id, relation.relation_type, target_id, provenance
                ),
                source_node_id=source_id,
                target_node_id=target_id,
                relation_type=relation.relation_type,
                description=relation.description,
                confidence=relation.confidence,
                extraction_version=extraction_version,
                provenance=provenance,
            )
        )

    node_by_id: dict[str, GraphNode] = {}
    for node in nodes:
        existing = node_by_id.get(node.graph_node_id)
        if existing is None:
            node_by_id[node.graph_node_id] = node
            continue
        aliases = list(
            dict.fromkeys(
                [
                    *existing.properties.get("aliases", []),
                    *node.properties.get("aliases", []),
                    *([node.label] if node.label != existing.label else []),
                ]
            )
        )
        source_ids = list(
            dict.fromkeys(
                [
                    *existing.properties.get("source_node_ids", []),
                    *node.properties.get("source_node_ids", []),
                ]
            )
        )
        existing.properties["aliases"] = aliases
        existing.properties["source_node_ids"] = source_ids
        existing_provenance = existing.properties.get("provenance", [])
        incoming_provenance = node.properties.get("provenance", [])
        provenance_by_key = {
            (
                item["canonical_node_id"],
                item["evidence_text"],
                item["start_offset"],
                item["end_offset"],
            ): item
            for item in [*existing_provenance, *incoming_provenance]
        }
        existing.properties["provenance"] = list(provenance_by_key.values())
    edge_by_id = {edge.edge_id: edge for edge in edges}
    return SemanticBuildResult(
        nodes=tuple(node_by_id.values()),
        edges=tuple(edge_by_id.values()),
        rejected=tuple(rejected),
    )
