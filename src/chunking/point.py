from __future__ import annotations

from src.legal_tree.models import LegalDocument, LegalNode
from src.legal_tree.resolver import LegalTreeResolver

from .base import ChunkStrategy
from .common import assemble, make_passage, subtree_content


def leaf_targets(resolver: LegalTreeResolver) -> list[LegalNode]:
    targets = [node for node in resolver.nodes if node.type == "point"]
    for clause in (node for node in resolver.nodes if node.type == "clause"):
        if not any(child.type == "point" for child in resolver.get_children(clause.id)):
            targets.append(clause)
    for article in (node for node in resolver.nodes if node.type == "article"):
        if not any(child.type == "clause" for child in resolver.get_children(article.id)):
            targets.append(article)
    order = {node.id: index for index, node in enumerate(resolver.nodes)}
    return sorted(targets, key=lambda node: order[node.id])


class PointStrategy(ChunkStrategy):
    name = "B3_point"

    def build(self, document: LegalDocument):
        resolver = LegalTreeResolver(document.nodes)
        passages = []
        for node in leaf_targets(resolver):
            nodes = subtree_content(resolver, node)
            text = assemble(nodes)
            passages.append(make_passage(
                strategy=self.name, passage_id=f"B3__{node.id}", primary=node,
                source_nodes=nodes, index_nodes=[node], context_nodes=[],
                citation_nodes=[node], index_text=text, evidence_text=text,
                counter=self.token_counter,
            ))
        return passages
