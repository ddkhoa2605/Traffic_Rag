from __future__ import annotations

from src.legal_tree.models import LegalDocument
from src.legal_tree.resolver import LegalTreeResolver

from .base import ChunkStrategy
from .common import article_heading, assemble, make_passage, subtree_content
from .point import leaf_targets


class PointParentStrategy(ChunkStrategy):
    name = "B4_point_clause_context"

    def build(self, document: LegalDocument):
        resolver = LegalTreeResolver(document.nodes)
        passages = []
        for node in leaf_targets(resolver):
            article = resolver.get_article(node.id)
            clause = resolver.get_clause(node.id)
            if node.type == "point" and article and clause:
                source_nodes = [article, clause, node]
                context_nodes = [article, clause]
                text = "\n".join(part for part in (article_heading(article), clause.text.strip(), node.text.strip()) if part)
            elif node.type == "clause" and article:
                source_nodes = [article, *subtree_content(resolver, node)]
                context_nodes = [article]
                text = "\n".join(part for part in (article_heading(article), assemble(subtree_content(resolver, node))) if part)
            else:
                source_nodes = subtree_content(resolver, node)
                context_nodes = []
                text = assemble(source_nodes)
            passages.append(make_passage(
                strategy=self.name, passage_id=f"B4__{node.id}", primary=node,
                source_nodes=source_nodes, index_nodes=[node], context_nodes=context_nodes,
                citation_nodes=[node], index_text=text, evidence_text=text,
                counter=self.token_counter,
            ))
        return passages
