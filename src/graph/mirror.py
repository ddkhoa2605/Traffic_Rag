from __future__ import annotations

import json
import hashlib
from collections import defaultdict

from .models import GraphEdge, GraphNode, GraphSourceDocument


def build_lightrag_payload(
    sources: list[GraphSourceDocument],
    nodes: list[GraphNode],
    edges: list[GraphEdge],
) -> tuple[dict, dict]:
    article_by_node: dict[str, str] = {}
    first_article_by_document: dict[str, str] = {}
    source_by_article = {item.article_node_id: item for item in sources}
    for source in sources:
        first_article_by_document.setdefault(
            source.canonical_document_id, source.article_node_id
        )
        for node_id in source.source_node_ids:
            article_by_node[node_id] = source.article_node_id

    def article_for_node(node: GraphNode) -> str:
        source_ids = node.properties.get("source_node_ids", [])
        for source_id in source_ids:
            if source_id in article_by_node:
                return article_by_node[source_id]
        if node.graph_node_id in article_by_node:
            return article_by_node[node.graph_node_id]
        if node.document_id in first_article_by_document:
            return first_article_by_document[node.document_id]
        return sources[0].article_node_id

    chunks = [
        {
            "content": source.text,
            "source_id": source.article_node_id,
            "file_path": f"canonical://{source.article_node_id}",
            "chunk_order_index": 0,
        }
        for source in sources
    ]
    node_by_id = {node.graph_node_id: node for node in nodes}
    entities = []
    entity_source: dict[str, str] = {}
    for node in nodes:
        article_id = article_for_node(node)
        entity_source[node.graph_node_id] = article_id
        description = node.properties.get("description") or node.label
        entities.append(
            {
                "entity_name": node.graph_node_id,
                "entity_type": node.node_type,
                "description": f"{node.label}\n{description}",
                "source_id": article_id,
                "file_path": f"canonical://{article_id}",
            }
        )

    grouped: dict[tuple[str, str], list[GraphEdge]] = defaultdict(list)
    for edge in edges:
        grouped[tuple(sorted((edge.source_node_id, edge.target_node_id)))].append(edge)
    relationships = []
    aggregate_map: dict[str, list[str]] = {}
    for (left, right), values in sorted(grouped.items()):
        evidence_sources = list(
            dict.fromkeys(
                item.canonical_node_id
                for edge in values
                for item in edge.provenance
            )
        )
        article_id = next(
            (article_by_node[value] for value in evidence_sources if value in article_by_node),
            entity_source[left],
        )
        descriptions = [
            f"[{edge.source_node_id} -{edge.relation_type}-> {edge.target_node_id}] {edge.description}".strip()
            for edge in sorted(values, key=lambda item: item.edge_id)
        ]
        aggregate_id = "MIRROR_EDGE::" + hashlib.sha256(
            f"{left}\0{right}".encode("utf-8")
        ).hexdigest()
        aggregate_map[aggregate_id] = [edge.edge_id for edge in values]
        relationships.append(
            {
                "src_id": left,
                "tgt_id": right,
                "description": "\n".join(descriptions),
                "keywords": ", ".join(sorted({edge.relation_type for edge in values})),
                "weight": float(max(1, len(set(evidence_sources)))),
                "source_id": article_id,
                "file_path": f"canonical://{article_id}",
            }
        )
    payload = {"chunks": chunks, "entities": entities, "relationships": relationships}
    manifest = {
        "chunk_count": len(chunks),
        "entity_count": len(entities),
        "directed_edge_count": len(edges),
        "mirrored_undirected_edge_count": len(relationships),
        "aggregated_edge_map": aggregate_map,
        "chunk_source_digest": hashlib.sha256(
            "\0".join(item.article_node_id for item in sources).encode("utf-8")
        ).hexdigest(),
        "entity_digest": hashlib.sha256(
            "\0".join(sorted(node.graph_node_id for node in nodes)).encode("utf-8")
        ).hexdigest(),
        "directed_edge_digest": hashlib.sha256(
            "\0".join(sorted(edge.edge_id for edge in edges)).encode("utf-8")
        ).hexdigest(),
        "payload_digest": hashlib.sha256(
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
        "warning": "LightRAG edges are an exploratory undirected mirror; use legal_graph_edge for direction and citations.",
    }
    return payload, manifest
