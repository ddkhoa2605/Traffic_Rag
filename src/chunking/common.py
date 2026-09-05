from __future__ import annotations

import hashlib

from src.legal_tree.models import LegalNode
from src.legal_tree.resolver import LegalTreeResolver
from src.parser.normalizer import normalize_text

from .models import RetrievalPassage, StrategyName
from .token_counter import TokenCounter


DEPTH = {
    "document": 0, "preamble": 1, "part": 1, "chapter": 2, "section": 3,
    "subsection": 4, "article": 5, "clause": 6, "point": 7,
    "subpoint": 8, "paragraph": 9, "table": 9, "table_row": 10,
    "appendix": 2, "form": 3, "footnote": 9, "amendment_instruction": 5,
    "signature": 1, "unknown_block": 9,
}


def unique_nodes(values: list[LegalNode]) -> list[LegalNode]:
    seen: set[str] = set()
    result: list[LegalNode] = []
    for node in values:
        if node.id not in seen:
            seen.add(node.id)
            result.append(node)
    return result


def article_heading(node: LegalNode) -> str:
    return node.text.splitlines()[0].strip() if node.text.strip() else ""


def subtree_content(resolver: LegalTreeResolver, node: LegalNode) -> list[LegalNode]:
    return [item for item in [node, *resolver.get_descendants(node.id)] if item.text.strip()]


def assemble(nodes: list[LegalNode]) -> str:
    return "\n".join(node.text.strip() for node in nodes if node.text.strip()).strip()


def passage_content_hash(strategy: str, index_text: str, evidence_text: str, source_nodes: list[LegalNode]) -> str:
    value = "\0".join([
        strategy,
        normalize_text(index_text),
        normalize_text(evidence_text),
        *(f"{node.id}:{node.content_hash}" for node in source_nodes),
    ])
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def make_passage(
    *,
    strategy: StrategyName,
    passage_id: str,
    primary: LegalNode,
    source_nodes: list[LegalNode],
    index_nodes: list[LegalNode],
    context_nodes: list[LegalNode],
    citation_nodes: list[LegalNode],
    index_text: str,
    evidence_text: str,
    counter: TokenCounter,
) -> RetrievalPassage:
    index_text = index_text.strip()
    evidence_text = evidence_text.strip()
    index_tokens = counter.count(index_text)
    evidence_tokens = counter.count(evidence_text)
    if not index_text or not evidence_text or not index_tokens or not evidence_tokens:
        raise ValueError(f"Empty passage text for {passage_id}")
    if index_tokens > counter.max_tokens:
        raise ValueError(f"{passage_id} has {index_tokens} index tokens; limit is {counter.max_tokens}")
    source_nodes = unique_nodes(source_nodes)
    return RetrievalPassage(
        passage_id=passage_id,
        strategy=strategy,
        document_id=primary.document_id,
        primary_node_id=primary.id,
        source_node_ids=[node.id for node in source_nodes],
        index_node_ids=[node.id for node in unique_nodes(index_nodes)],
        context_node_ids=[node.id for node in unique_nodes(context_nodes)],
        citation_node_ids=[node.id for node in unique_nodes(citation_nodes)],
        hierarchy=primary.hierarchy,
        index_text=index_text,
        evidence_text=evidence_text,
        display_text=evidence_text,
        token_count_index=index_tokens,
        token_count_evidence=evidence_tokens,
        content_hash=passage_content_hash(strategy, index_text, evidence_text, source_nodes),
    )
