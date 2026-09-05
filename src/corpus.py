from __future__ import annotations

from pathlib import Path

from src.parser.io import read_jsonl
from src.parser.models import LegalNode


class NodeView:
    def __init__(self, node: LegalNode, by_id: dict[str, LegalNode]):
        self._node = node
        self._by_id = by_id

    def __getattr__(self, name):
        return getattr(self._node, name)

    def _child(self, node_type: str, key: str) -> "NodeView":
        field = {"article": "article", "clause": "clause", "point": "point"}[node_type]
        for node in self._by_id.values():
            if node.type == node_type and node.hierarchy.get(field) == str(key):
                if node_type == "article" or node.parent_id == self.id:
                    return NodeView(node, self._by_id)
        raise KeyError(f"{node_type} {key} not found under {self.id}")

    def article(self, key: str) -> "NodeView":
        return self._child("article", key)

    def clause(self, key: str) -> "NodeView":
        return self._child("clause", key)

    def point(self, key: str) -> "NodeView":
        return self._child("point", key.lower().replace("Đ", "đ"))


class Corpus:
    def __init__(self, root: Path):
        self.root = root

    def document(self, document_id: str) -> NodeView:
        path = self.root / "data" / "05_validated" / document_id / "nodes.jsonl"
        if not path.is_file():
            raise KeyError(f"Validated document not found: {document_id}")
        nodes = [LegalNode.model_validate(item) for item in read_jsonl(path)]
        by_id = {node.id: node for node in nodes}
        return NodeView(by_id[document_id], by_id)


def load_validated_corpus(root: str | Path | None = None) -> Corpus:
    return Corpus(Path(root or Path.cwd()).resolve())

