from __future__ import annotations

from src.legal_tree.models import LegalDocument
from src.legal_tree.resolver import LegalTreeResolver

from .base import ChunkStrategy
from .common import article_heading, assemble, make_passage, subtree_content
from .models import EvidenceBundle, ExpansionPolicy
from .point import leaf_targets
from .token_counter import TokenCounter


class ChildParentStrategy(ChunkStrategy):
    name = "B5_child_parent"

    def build(self, document: LegalDocument):
        resolver = LegalTreeResolver(document.nodes)
        passages = []
        for node in leaf_targets(resolver):
            article = resolver.get_article(node.id)
            if node.type == "point" and article:
                index_text = "\n".join(part for part in (article_heading(article), node.text.strip()) if part)
                evidence_text = node.text
                source_nodes = [article, node]
                context_nodes = [article]
            elif node.type == "clause" and article:
                text = assemble(subtree_content(resolver, node))
                index_text = "\n".join(part for part in (article_heading(article), text) if part)
                evidence_text = index_text
                source_nodes = [article, *subtree_content(resolver, node)]
                context_nodes = [article]
            else:
                source_nodes = subtree_content(resolver, node)
                context_nodes = []
                index_text = evidence_text = assemble(source_nodes)
            passages.append(make_passage(
                strategy=self.name, passage_id=f"B5__{node.id}", primary=node,
                source_nodes=source_nodes, index_nodes=[node], context_nodes=context_nodes,
                citation_nodes=[node], index_text=index_text, evidence_text=evidence_text,
                counter=self.token_counter,
            ))
        return passages


def expand_evidence(
    primary_node_id: str,
    policy: ExpansionPolicy,
    resolver: LegalTreeResolver,
    token_counter: TokenCounter,
) -> EvidenceBundle:
    node = resolver.get_node(primary_node_id)
    article = resolver.get_article(node.id)
    clause = resolver.get_clause(node.id)
    included = []
    context = []
    parts = []
    if policy.include_article_heading and article and article.id != node.id:
        included.append(article)
        context.append(article)
        parts.append(article_heading(article))
    if policy.include_clause_intro and clause and clause.id != node.id:
        included.append(clause)
        context.append(clause)
        parts.append(clause.text.strip())
    included.append(node)
    parts.append(node.text.strip())
    text = "\n".join(part for part in parts if part)
    return EvidenceBundle(
        primary_node_id=node.id,
        included_node_ids=[item.id for item in included],
        context_node_ids=[item.id for item in context],
        citation_node_ids=[node.id],
        evidence_text=text,
        token_count=token_counter.count(text),
    )
