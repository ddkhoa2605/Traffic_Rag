from __future__ import annotations

import pytest

from src.legal_tree.models import LegalNode
from src.legal_tree.resolver import LegalTreeResolver


def node(node_id: str, node_type: str, parent_id: str | None = None, children=()):
    return LegalNode(
        id=node_id,
        document_id="LAW_TEST",
        type=node_type,
        hierarchy={"article": "1"} if node_type != "document" else {},
        text=node_id,
        parent_id=parent_id,
        children_ids=list(children),
        content_hash=f"hash-{node_id}",
    )


def test_old_schema_defaults_and_future_node_type():
    item = node("u", "unknown_block")
    assert item.node_status == "active"
    assert item.metadata == {}
    assert set(item.hierarchy) == {
        "part", "chapter", "section", "subsection", "article", "clause", "point", "subpoint"
    }


def test_resolver_preserves_order_and_traverses():
    nodes = [
        node("d", "document", children=("a",)),
        node("a", "article", "d", ("c",)),
        node("c", "clause", "a", ("p",)),
        node("p", "point", "c"),
    ]
    resolver = LegalTreeResolver(nodes)
    assert [item.id for item in resolver.get_descendants("a")] == ["c", "p"]
    assert [item.id for item in resolver.get_ancestors("p")] == ["c", "a", "d"]
    assert resolver.get_article("p").id == "a"
    assert resolver.get_clause("p").id == "c"
    assert resolver.get_references("p") == []


@pytest.mark.parametrize("nodes, message", [
    ([node("x", "article"), node("x", "clause")], "Duplicate"),
    ([node("x", "article", "missing")], "Dangling"),
    ([node("x", "article", "y"), node("y", "clause", "x")], "Cycle"),
])
def test_resolver_rejects_invalid_tree(nodes, message):
    with pytest.raises(ValueError, match=message):
        LegalTreeResolver(nodes)
