from __future__ import annotations

from src.legal_tree.models import LegalDocument
from src.legal_tree.resolver import LegalTreeResolver

from .base import ChunkStrategy
from .common import assemble, make_passage, subtree_content


class ClauseStrategy(ChunkStrategy):
    name = "B2_clause"

    def build(self, document: LegalDocument):
        resolver = LegalTreeResolver(document.nodes)
        passages = []
        for clause in (node for node in resolver.nodes if node.type == "clause"):
            nodes = subtree_content(resolver, clause)
            text = assemble(nodes)
            passages.append(make_passage(
                strategy=self.name, passage_id=f"B2__{clause.id}", primary=clause,
                source_nodes=nodes, index_nodes=[clause], context_nodes=[],
                citation_nodes=[clause], index_text=text, evidence_text=text,
                counter=self.token_counter,
            ))
        for article in (node for node in resolver.nodes if node.type == "article"):
            if not any(child.type == "clause" for child in resolver.get_children(article.id)):
                nodes = subtree_content(resolver, article)
                text = assemble(nodes)
                passages.append(make_passage(
                    strategy=self.name, passage_id=f"B2__{article.id}", primary=article,
                    source_nodes=nodes, index_nodes=[article], context_nodes=[],
                    citation_nodes=[article], index_text=text, evidence_text=text,
                    counter=self.token_counter,
                ))
        return passages
