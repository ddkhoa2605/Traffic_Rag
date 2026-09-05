from __future__ import annotations

from collections import Counter, defaultdict

from .models import GraphEdge, GraphNode, GraphSourceDocument


def validate_article_sources(
    sources: list[GraphSourceDocument],
    *,
    expected_counts: dict[str, int] | None = None,
) -> list[str]:
    errors: list[str] = []
    ids = [item.article_node_id for item in sources]
    if len(ids) != len(set(ids)):
        errors.append("duplicate article_node_id")
    if len(sources) != 175:
        errors.append(f"expected 175 articles, found {len(sources)}")
    counts = Counter(item.canonical_document_id for item in sources)
    expected_counts = expected_counts or {"LAW_35_2024": 86, "LAW_36_2024": 89}
    if dict(counts) != expected_counts:
        errors.append(f"unexpected article distribution: {dict(counts)}")
    for item in sources:
        if item.token_count > 8192:
            errors.append(
                f"{item.article_node_id}: {item.token_count} tokens exceeds 8192"
            )
        if len(item.source_node_ids) != len(set(item.source_node_ids)):
            errors.append(f"{item.article_node_id}: duplicate source_node_id")
        if item.article_node_id not in item.source_node_ids:
            errors.append(f"{item.article_node_id}: Article missing from provenance")
        if set(item.source_node_ids) != set(item.node_texts):
            errors.append(f"{item.article_node_id}: node_texts/source_node_ids mismatch")
        normalized = item.text.casefold()
        noise_markers = (
            "thongtinchinhphu@chinhphu.vn",
            "cơ quan: văn phòng chính phủ",
            "thời gian ký:",
        )
        if any(marker in normalized for marker in noise_markers):
            errors.append(f"{item.article_node_id}: PDF header/signature noise")
        marker_positions: list[int] = []
        for node_id in item.source_node_ids:
            marker = f"[NODE_ID={node_id} "
            if item.extraction_text.count(marker) != 1:
                errors.append(
                    f"{item.article_node_id}: expected exactly one marker for {node_id}"
                )
            marker_positions.append(item.extraction_text.find(marker))
        if marker_positions != sorted(marker_positions) or any(value < 0 for value in marker_positions):
            errors.append(f"{item.article_node_id}: source markers are not in canonical order")
    return errors


def validate_graph(
    nodes: list[GraphNode],
    edges: list[GraphEdge],
    *,
    canonical_texts: dict[str, str] | None = None,
) -> list[str]:
    errors: list[str] = []
    by_id = {node.graph_node_id: node for node in nodes}
    if len(by_id) != len(nodes):
        errors.append("duplicate graph node ID")
    edge_ids = {edge.edge_id for edge in edges}
    if len(edge_ids) != len(edges):
        errors.append("duplicate graph edge ID")
    for node in nodes:
        if node.node_kind != "semantic":
            continue
        provenance = node.properties.get("provenance", [])
        if not provenance:
            errors.append(f"{node.graph_node_id}: semantic entity has no provenance")
            continue
        for item in provenance:
            canonical_id = item.get("canonical_node_id")
            canonical = by_id.get(canonical_id)
            if canonical is None or canonical.node_kind != "canonical":
                errors.append(
                    f"{node.graph_node_id}: unknown entity provenance node {canonical_id}"
                )
                continue
            if node.document_id != canonical.document_id:
                errors.append(
                    f"{node.graph_node_id}: cross-document semantic entity provenance"
                )
            if canonical_texts is None:
                continue
            text = canonical_texts.get(canonical_id)
            start = item.get("start_offset")
            end = item.get("end_offset")
            evidence = item.get("evidence_text")
            if (
                text is None
                or not isinstance(start, int)
                or not isinstance(end, int)
                or text[start:end] != evidence
            ):
                errors.append(
                    f"{node.graph_node_id}: entity provenance evidence/offset mismatch"
                )
    for edge in edges:
        if edge.source_node_id not in by_id:
            errors.append(f"{edge.edge_id}: dangling source {edge.source_node_id}")
        if edge.target_node_id not in by_id:
            errors.append(f"{edge.edge_id}: dangling target {edge.target_node_id}")
        if edge.relation_type == "CONTAINS":
            source = by_id.get(edge.source_node_id)
            target = by_id.get(edge.target_node_id)
            if source and target and source.document_id != target.document_id:
                errors.append(f"{edge.edge_id}: cross-document CONTAINS edge")
        if edge.relation_type not in {"CONTAINS", "NEXT", "ISSUED_BY", "REFERENCES"} and not edge.provenance:
            errors.append(f"{edge.edge_id}: semantic edge has no provenance")
        for provenance in edge.provenance:
            if canonical_texts is None:
                continue
            text = canonical_texts.get(provenance.canonical_node_id)
            if text is None:
                errors.append(f"{edge.edge_id}: unknown provenance node {provenance.canonical_node_id}")
                continue
            if provenance.end_offset > len(text):
                errors.append(f"{edge.edge_id}: provenance offset outside canonical text")
            elif text[provenance.start_offset:provenance.end_offset] != provenance.evidence_text:
                errors.append(f"{edge.edge_id}: provenance evidence/offset mismatch")

    contains: dict[str, list[str]] = defaultdict(list)
    for edge in edges:
        if edge.relation_type == "CONTAINS":
            contains[edge.source_node_id].append(edge.target_node_id)
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node_id: str) -> None:
        if node_id in visiting:
            errors.append(f"CONTAINS cycle at {node_id}")
            return
        if node_id in visited:
            return
        visiting.add(node_id)
        for child in contains.get(node_id, []):
            visit(child)
        visiting.remove(node_id)
        visited.add(node_id)

    for node_id in by_id:
        visit(node_id)
    return errors
