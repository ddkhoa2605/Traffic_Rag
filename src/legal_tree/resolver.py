from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

from .models import LegalDocument, LegalNode, LegalRelation


class LegalTreeResolver:
    def __init__(self, nodes: Iterable[LegalNode], relations: Iterable[LegalRelation] = ()):
        ordered = list(nodes)
        self._nodes: dict[str, LegalNode] = {}
        self._order: dict[str, int] = {}
        for index, node in enumerate(ordered):
            if node.id in self._nodes:
                raise ValueError(f"Duplicate legal node ID: {node.id}")
            self._nodes[node.id] = node
            self._order[node.id] = index
        self._children: dict[str, list[str]] = defaultdict(list)
        for node in ordered:
            if node.parent_id is not None:
                if node.parent_id not in self._nodes:
                    raise ValueError(f"Dangling parent {node.parent_id} for {node.id}")
                self._children[node.parent_id].append(node.id)
        for parent_id, child_ids in self._children.items():
            declared = self._nodes[parent_id].children_ids
            if declared and declared != child_ids:
                raise ValueError(f"Children order mismatch for {parent_id}")
        self._relations = tuple(relations)
        self._validate_cycles()

    @classmethod
    def from_document(cls, document: LegalDocument) -> "LegalTreeResolver":
        return cls(document.nodes)

    @property
    def nodes(self) -> tuple[LegalNode, ...]:
        return tuple(sorted(self._nodes.values(), key=lambda node: self._order[node.id]))

    def get_node(self, node_id: str) -> LegalNode:
        try:
            return self._nodes[node_id]
        except KeyError as exc:
            raise KeyError(f"Unknown legal node: {node_id}") from exc

    def get_parent(self, node_id: str) -> LegalNode | None:
        node = self.get_node(node_id)
        return self._nodes.get(node.parent_id) if node.parent_id else None

    def get_children(self, node_id: str) -> list[LegalNode]:
        self.get_node(node_id)
        return [self._nodes[child_id] for child_id in self._children.get(node_id, [])]

    def get_ancestors(self, node_id: str) -> list[LegalNode]:
        result: list[LegalNode] = []
        current = self.get_parent(node_id)
        while current is not None:
            result.append(current)
            current = self.get_parent(current.id)
        return result

    def get_descendants(self, node_id: str) -> list[LegalNode]:
        result: list[LegalNode] = []
        for child in self.get_children(node_id):
            result.append(child)
            result.extend(self.get_descendants(child.id))
        return result

    def get_article(self, node_id: str) -> LegalNode | None:
        node = self.get_node(node_id)
        if node.type == "article":
            return node
        return next((item for item in self.get_ancestors(node_id) if item.type == "article"), None)

    def get_clause(self, node_id: str) -> LegalNode | None:
        node = self.get_node(node_id)
        if node.type == "clause":
            return node
        return next((item for item in self.get_ancestors(node_id) if item.type == "clause"), None)

    def _relations_for(self, node_id: str, relation_type: str) -> list[LegalNode]:
        self.get_node(node_id)
        targets = [rel.target_node_id for rel in self._relations if rel.source_node_id == node_id and rel.relation_type == relation_type]
        return [self.get_node(target) for target in targets]

    def get_references(self, node_id: str) -> list[LegalNode]:
        return self._relations_for(node_id, "REFERENCES")

    def get_amendments(self, node_id: str) -> list[LegalNode]:
        return self._relations_for(node_id, "AMENDS")

    def get_repealed_by(self, node_id: str) -> list[LegalNode]:
        self.get_node(node_id)
        sources = [rel.source_node_id for rel in self._relations if rel.target_node_id == node_id and rel.relation_type == "REPEALS"]
        return [self.get_node(source) for source in sources]

    def is_ancestor(self, ancestor_id: str, node_id: str) -> bool:
        return any(node.id == ancestor_id for node in self.get_ancestors(node_id))

    def _validate_cycles(self) -> None:
        for node_id in self._nodes:
            seen: set[str] = set()
            current = node_id
            while current:
                if current in seen:
                    raise ValueError(f"Cycle detected at {current}")
                seen.add(current)
                parent = self._nodes[current].parent_id
                current = parent or ""
