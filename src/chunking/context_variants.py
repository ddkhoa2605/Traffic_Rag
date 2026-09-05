from __future__ import annotations

from src.legal_tree.models import LegalDocument
from src.legal_tree.resolver import LegalTreeResolver

from .base import ChunkStrategy
from .common import article_heading, assemble, make_passage, subtree_content
from .point import leaf_targets


class ContextVariantStrategy(ChunkStrategy):
    variant = "B4a"
    name = "B4a_point"

    def build(self, document: LegalDocument):
        resolver = LegalTreeResolver(document.nodes)
        document_node = next((node for node in resolver.nodes if node.type == "document"), None)
        document_titles = self.config.get("document_titles", {})
        document_title = str(document_titles.get(document.document_id, "")).strip()
        if self.variant == "B4e" and not document_title:
            raise ValueError(f"Missing registry document title for {document.document_id}")
        passages = []
        for node in leaf_targets(resolver):
            article = resolver.get_article(node.id)
            clause = resolver.get_clause(node.id)
            if node.type == "point":
                components, source_nodes, context_nodes = [], [], []
                if self.variant == "B4e" and document_node:
                    components.append(document_title); source_nodes.append(document_node); context_nodes.append(document_node)
                if self.variant in {"B4b", "B4d", "B4e"} and article:
                    components.append(article_heading(article)); source_nodes.append(article); context_nodes.append(article)
                if self.variant in {"B4c", "B4d", "B4e"} and clause:
                    components.append(clause.text.strip()); source_nodes.append(clause); context_nodes.append(clause)
                components.append(node.text.strip()); source_nodes.append(node)
                text = "\n".join(part for part in components if part)
            elif node.type == "clause":
                own_nodes = subtree_content(resolver, node)
                components, source_nodes, context_nodes = [], [], []
                if self.variant == "B4e" and document_node:
                    components.append(document_title); source_nodes.append(document_node); context_nodes.append(document_node)
                if self.variant in {"B4b", "B4d", "B4e"} and article:
                    components.append(article_heading(article)); source_nodes.append(article); context_nodes.append(article)
                components.append(assemble(own_nodes)); source_nodes.extend(own_nodes)
                text = "\n".join(part for part in components if part)
            else:
                own_nodes = subtree_content(resolver, node)
                if self.variant == "B4e" and document_node:
                    source_nodes = [document_node, *own_nodes]
                    context_nodes = [document_node]
                    text = "\n".join(part for part in (document_title, assemble(own_nodes)) if part)
                else:
                    source_nodes = own_nodes
                    context_nodes = []
                    text = assemble(source_nodes)
            passages.append(make_passage(
                strategy=self.name, passage_id=f"{self.variant}__{node.id}", primary=node,
                source_nodes=source_nodes, index_nodes=[node], context_nodes=context_nodes,
                citation_nodes=[node], index_text=text, evidence_text=text,
                counter=self.token_counter,
            ))
        return passages


class B4aStrategy(ContextVariantStrategy):
    variant, name = "B4a", "B4a_point"


class B4bStrategy(ContextVariantStrategy):
    variant, name = "B4b", "B4b_article_point"


class B4cStrategy(ContextVariantStrategy):
    variant, name = "B4c", "B4c_clause_point"


class B4dStrategy(ContextVariantStrategy):
    variant, name = "B4d", "B4d_article_clause_point"


class B4eStrategy(ContextVariantStrategy):
    variant, name = "B4e", "B4e_document_article_clause_point"
