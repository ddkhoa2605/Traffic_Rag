from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from src.legal_tree.loader import load_legal_document
from src.legal_tree.models import LegalNode
from src.legal_tree.resolver import LegalTreeResolver
from src.registry.loader import Registry, load_registry

from .identity import graph_edge_id
from .models import GraphEdge, GraphNode
from .references import ReferenceFinding, extract_reference_edges


def _node_label(node: LegalNode, document_title: str) -> str:
    if node.type == "document":
        return document_title
    if node.title and node.title.strip():
        return node.title.strip()
    if node.text.strip():
        return node.text.strip().splitlines()[0]
    return node.id


def _edge(source: str, relation: str, target: str) -> GraphEdge:
    return GraphEdge(
        edge_id=graph_edge_id(source, relation, target),
        source_node_id=source,
        target_node_id=target,
        relation_type=relation,
    )


def build_structural_graph(
    root: str | Path,
    *,
    registry: Registry | None = None,
    include_references: bool = True,
) -> tuple[list[GraphNode], list[GraphEdge], list[ReferenceFinding]]:
    root_path = Path(root).resolve()
    registry = registry or load_registry(root_path)
    canonical_nodes: list[LegalNode] = []
    resolvers: dict[str, LegalTreeResolver] = {}
    graph_nodes: list[GraphNode] = []
    edges: list[GraphEdge] = []

    for document_id, record in registry.documents.items():
        document = load_legal_document(root_path, document_id)
        resolver = LegalTreeResolver(document.nodes)
        resolvers[document_id] = resolver
        canonical_nodes.extend(resolver.nodes)
        for node in resolver.nodes:
            graph_nodes.append(
                GraphNode(
                    graph_node_id=node.id,
                    node_kind="canonical",
                    node_type=node.type,
                    label=_node_label(node, record.title),
                    document_id=document_id,
                    canonical_node_id=node.id,
                    properties={
                        "hierarchy": node.hierarchy,
                        "parent_id": node.parent_id,
                        "content_hash": node.content_hash,
                    },
                )
            )
            if node.parent_id:
                edges.append(_edge(node.parent_id, "CONTAINS", node.id))

        articles = [node for node in resolver.nodes if node.type == "article"]
        for left, right in zip(articles, articles[1:]):
            edges.append(_edge(left.id, "NEXT", right.id))

        sibling_groups: dict[tuple[str | None, str], list[LegalNode]] = defaultdict(list)
        for node in resolver.nodes:
            if node.type != "article":
                sibling_groups[(node.parent_id, node.type)].append(node)
        for siblings in sibling_groups.values():
            for left, right in zip(siblings, siblings[1:]):
                edges.append(_edge(left.id, "NEXT", right.id))

        authority = record.issuing_authority
        if authority:
            if not any(item.graph_node_id == authority.authority_id for item in graph_nodes):
                graph_nodes.append(
                    GraphNode(
                        graph_node_id=authority.authority_id,
                        node_kind="authority",
                        node_type="Authority",
                        label=authority.title,
                    )
                )
            root_node = next(node for node in resolver.nodes if node.type == "document")
            edges.append(_edge(root_node.id, "ISSUED_BY", authority.authority_id))

    findings: list[ReferenceFinding] = []
    if include_references:
        reference_edges, findings = extract_reference_edges(
            canonical_nodes, registry, resolvers
        )
        edges.extend(reference_edges)
    return graph_nodes, edges, findings
