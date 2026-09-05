from __future__ import annotations

from src.legal_tree.models import LegalDocument
from src.legal_tree.resolver import LegalTreeResolver

from .base import ChunkStrategy
from .common import assemble, make_passage, subtree_content


class ArticleStrategy(ChunkStrategy):
    name = "B1_article"

    def build(self, document: LegalDocument):
        resolver = LegalTreeResolver(document.nodes)
        passages = []
        for article in (node for node in resolver.nodes if node.type == "article"):
            nodes = subtree_content(resolver, article)
            text = assemble(nodes)
            passages.append(make_passage(
                strategy=self.name, passage_id=f"B1__{article.id}", primary=article,
                source_nodes=nodes, index_nodes=[article], context_nodes=[],
                citation_nodes=[article], index_text=text, evidence_text=text,
                counter=self.token_counter,
            ))
        return passages
