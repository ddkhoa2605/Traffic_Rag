from __future__ import annotations

import hashlib
from dataclasses import dataclass

import orjson

from src.chunking.common import article_heading
from src.chunking.models import EvidenceBundle, RetrievalPassage
from src.chunking.token_counter import TokenCounter
from src.legal_tree.models import LegalNode
from src.legal_tree.resolver import LegalTreeResolver
from src.parser.normalizer import normalize_text
from src.retrieval_eval.models import EvidenceComponent
from src.retrieval_eval.round2 import PassageComponent


@dataclass(frozen=True)
class DirectEvidence:
    document_id: str
    primary_node: LegalNode
    member_node_ids: list[str]
    passage_id: str
    bundle: EvidenceBundle
    components: list[EvidenceComponent]


def _unique(nodes: list[LegalNode]) -> list[LegalNode]:
    seen: set[str] = set()
    result: list[LegalNode] = []
    for node in nodes:
        if node.id not in seen:
            seen.add(node.id)
            result.append(node)
    return result


def _token_component(
    node: LegalNode,
    role: str,
    text: str,
    counter: TokenCounter,
) -> PassageComponent:
    normalized = text.strip()
    if not normalized:
        raise ValueError(f"Empty {role} component for {node.id}")
    encoded = counter.encode_with_offsets(normalized)
    if not encoded.token_ids:
        raise ValueError(f"Zero-token {role} component for {node.id}")
    return PassageComponent(
        node_id=node.id,
        role=role,
        text=normalized,
        token_sequence_hash=hashlib.sha256(orjson.dumps(encoded.token_ids)).hexdigest(),
        token_count=len(encoded.token_ids),
    )


def project_b7_passage_components(
    passage: RetrievalPassage,
    resolver: LegalTreeResolver,
    document_title: str,
    counter: TokenCounter,
) -> list[PassageComponent]:
    """Reproject frozen B4e provenance with registry-backed document metadata.

    The frozen passage remains the retrieval/index artifact. B7 only rebuilds
    evidence components, so passage IDs, embeddings, rankings, provenance, and
    canonical hashes are not changed.
    """
    if passage.strategy != "B4e_document_article_clause_point":
        raise ValueError(f"B7 evidence projection requires B4e passage: {passage.passage_id}")
    context_ids = set(passage.context_node_ids)
    components: list[PassageComponent] = []
    for node_id in passage.source_node_ids:
        node = resolver.get_node(node_id)
        if node_id in context_ids and node.type == "document":
            role, component_text = "document_title", document_title
        elif node_id in context_ids and node.type == "article":
            role, component_text = "article_heading", article_heading(node)
        elif node_id in context_ids and node.type == "clause":
            role, component_text = "clause_intro", node.text
        else:
            role, component_text = node.type, node.text
        components.append(_token_component(node, role, component_text, counter))
    return components


def project_b7_passage_bundle(
    passage: RetrievalPassage,
    resolver: LegalTreeResolver,
    document_title: str,
    counter: TokenCounter,
) -> tuple[EvidenceBundle, list[EvidenceComponent]]:
    projected = project_b7_passage_components(passage, resolver, document_title, counter)
    evidence_text = "\n".join(component.text for component in projected).strip()
    token_count = counter.count(evidence_text)
    if not evidence_text or token_count <= 0:
        raise ValueError(f"Empty B7 evidence projection for {passage.passage_id}")
    if token_count > counter.max_tokens:
        raise ValueError(
            f"B7 evidence for {passage.passage_id} has {token_count} tokens; "
            f"limit is {counter.max_tokens}"
        )
    bundle = EvidenceBundle(
        primary_node_id=passage.primary_node_id,
        included_node_ids=list(passage.source_node_ids),
        context_node_ids=list(passage.context_node_ids),
        citation_node_ids=list(passage.citation_node_ids),
        evidence_text=evidence_text,
        token_count=token_count,
    )
    components = [EvidenceComponent(
        node_id=item.node_id,
        role=item.role,
        token_sequence_hash=item.token_sequence_hash,
        token_count=item.token_count,
    ) for item in projected]
    return bundle, components


def build_b7_article_scout_bundle(
    article_passage: RetrievalPassage,
    evidence_passage_ids: list[str],
    fine_by_id: dict[str, RetrievalPassage],
    resolvers: dict[str, LegalTreeResolver],
    document_titles: dict[str, str],
    counter: TokenCounter,
) -> tuple[EvidenceBundle, list[EvidenceComponent]]:
    if not evidence_passage_ids:
        raise ValueError(f"B7 Article scout has no B4e fallback: {article_passage.primary_node_id}")
    projected: list[PassageComponent] = []
    citations = [article_passage.primary_node_id]
    for passage_id in evidence_passage_ids:
        passage = fine_by_id.get(passage_id)
        if passage is None or passage.document_id != article_passage.document_id:
            raise ValueError(f"Invalid B7 evidence passage for {article_passage.passage_id}: {passage_id}")
        resolver = resolvers[passage.document_id]
        article = resolver.get_article(passage.primary_node_id)
        if article is None or article.id != article_passage.primary_node_id:
            raise ValueError(f"B7 evidence crosses Article boundary: {passage_id}")
        projected.extend(project_b7_passage_components(
            passage, resolver, document_titles[passage.document_id], counter,
        ))
        citations.extend(passage.citation_node_ids)
    unique_components: list[PassageComponent] = []
    seen_hashes: set[str] = set()
    for component in projected:
        if component.token_sequence_hash not in seen_hashes:
            seen_hashes.add(component.token_sequence_hash)
            unique_components.append(component)
    evidence_text = "\n".join(component.text for component in unique_components).strip()
    token_count = counter.count(evidence_text)
    if not evidence_text or token_count <= 0:
        raise ValueError(f"Empty B7 Article evidence bundle: {article_passage.passage_id}")
    if token_count > counter.max_tokens:
        raise ValueError(
            f"B7 Article evidence for {article_passage.passage_id} has {token_count} tokens; "
            f"limit is {counter.max_tokens}"
        )
    included = list(dict.fromkeys(component.node_id for component in unique_components))
    contexts = list(dict.fromkeys(
        component.node_id for component in unique_components
        if component.role in {"document_title", "article_heading", "clause_intro"}
    ))
    bundle = EvidenceBundle(
        primary_node_id=article_passage.primary_node_id,
        included_node_ids=included,
        context_node_ids=contexts,
        citation_node_ids=list(dict.fromkeys(citations)),
        evidence_text=evidence_text,
        token_count=token_count,
    )
    components = [EvidenceComponent(
        node_id=item.node_id,
        role=item.role,
        token_sequence_hash=item.token_sequence_hash,
        token_count=item.token_count,
    ) for item in unique_components]
    return bundle, components


def _build_bundle(
    primary: LegalNode,
    included: list[LegalNode],
    context: list[LegalNode],
    citations: list[LegalNode],
    component_items: list[tuple[LegalNode, str, str]],
    counter: TokenCounter,
) -> tuple[EvidenceBundle, list[EvidenceComponent]]:
    text = "\n".join(text.strip() for _, _, text in component_items if text.strip()).strip()
    token_count = counter.count(text)
    if not text or token_count <= 0:
        raise ValueError(f"Direct evidence is empty for {primary.id}")
    if token_count > counter.max_tokens:
        raise ValueError(
            f"Direct evidence for {primary.id} has {token_count} tokens; limit is {counter.max_tokens}"
        )
    bundle = EvidenceBundle(
        primary_node_id=primary.id,
        included_node_ids=[node.id for node in _unique(included)],
        context_node_ids=[node.id for node in _unique(context)],
        citation_node_ids=[node.id for node in _unique(citations)],
        evidence_text=text,
        token_count=token_count,
    )
    components = []
    for node, role, component_text in component_items:
        encoded = counter.encode_with_offsets(component_text.strip())
        components.append(EvidenceComponent(
            node_id=node.id,
            role=role,
            token_sequence_hash=hashlib.sha256(orjson.dumps(encoded.token_ids)).hexdigest(),
            token_count=len(encoded.token_ids),
        ))
    return bundle, components


def build_direct_evidence(
    node_ids: list[str],
    resolver: LegalTreeResolver,
    fine_by_primary_id: dict[str, RetrievalPassage],
    counter: TokenCounter,
    *,
    document_title: str,
) -> DirectEvidence:
    if not node_ids:
        raise ValueError("Direct evidence requires at least one canonical node")
    targets = [resolver.get_node(node_id) for node_id in node_ids]
    if len({node.document_id for node in targets}) != 1:
        raise ValueError("Direct evidence cannot cross documents")
    primary = targets[0]
    if len(targets) == 1 and primary.type == "point":
        passage = fine_by_primary_id.get(primary.id)
        if passage is None:
            raise ValueError(f"No frozen B4e projection for {primary.id}")
        bundle, components = project_b7_passage_bundle(
            passage, resolver, document_title, counter,
        )
        return DirectEvidence(primary.document_id, primary, [primary.id], passage.passage_id, bundle, components)

    if len(targets) > 1:
        if any(node.type != "point" for node in targets):
            raise ValueError("Multi-target direct evidence currently supports Point lists only")
        clauses = {resolver.get_clause(node.id).id for node in targets if resolver.get_clause(node.id)}
        articles = {resolver.get_article(node.id).id for node in targets if resolver.get_article(node.id)}
        if len(clauses) != 1 or len(articles) != 1:
            raise ValueError("Multi-point evidence must share one Clause and Article")
        clause = resolver.get_node(next(iter(clauses)))
        article = resolver.get_node(next(iter(articles)))
        document = next(node for node in resolver.nodes if node.type == "document")
        component_items = [
            (document, "document_title", document_title),
            (article, "article_heading", article_heading(article)),
            (clause, "clause_intro", clause.text),
            *((node, "point", node.text) for node in targets),
        ]
        included = [document, article, clause, *targets]
        bundle, components = _build_bundle(
            primary, included, [document, article, clause], targets, component_items, counter,
        )
    elif primary.type == "clause":
        points = [node for node in resolver.get_children(primary.id) if node.type == "point"]
        included = [primary, *points]
        bundle, components = _build_bundle(
            primary, included, [], included,
            [(node, node.type, node.text) for node in included], counter,
        )
    elif primary.type == "article":
        included = [primary, *resolver.get_descendants(primary.id)]
        included = [node for node in included if node.text.strip()]
        bundle, components = _build_bundle(
            primary, included, [], included,
            [(node, node.type, node.text) for node in included], counter,
        )
    else:
        raise ValueError(f"Unsupported direct target type: {primary.type}")

    digest = hashlib.sha256(
        "\0".join([primary.id, *(node.id for node in targets), normalize_text(bundle.evidence_text)]).encode("utf-8")
    ).hexdigest()[:16]
    return DirectEvidence(
        document_id=primary.document_id,
        primary_node=primary,
        member_node_ids=[node.id for node in targets],
        passage_id=f"B7__DIRECT__{primary.id}__{digest}",
        bundle=bundle,
        components=components,
    )
