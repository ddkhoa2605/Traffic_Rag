from __future__ import annotations

import hashlib
import re

from src.chunking.context_variants import B4bStrategy, B4dStrategy, B4eStrategy
from src.chunking.token_counter import EncodedText
from src.legal_tree.models import LegalDocument, LegalNode
from src.legal_tree.resolver import LegalTreeResolver
from src.retrieval_eval.round2 import (
    compute_query_redundancy,
    project_b4_components,
    select_round2_candidate,
)


class ContentTokenCounter:
    model_name = "test-content"
    revision = "v1"
    max_tokens = 8192

    def encode_with_offsets(self, text: str) -> EncodedText:
        matches = list(re.finditer(r"\S+", text))
        return EncodedText(
            token_ids=[int.from_bytes(hashlib.sha256(match.group().encode()).digest()[:4], "big") for match in matches],
            offsets=[match.span() for match in matches],
        )

    def count(self, text: str) -> int:
        return len(self.encode_with_offsets(text).token_ids)


def _node(node_id, node_type, text, parent=None, children=()):
    return LegalNode(
        id=node_id, document_id="LAW_TEST", type=node_type, text=text,
        parent_id=parent, children_ids=list(children), hierarchy={}, content_hash=f"h-{node_id}",
    )


def _document():
    nodes = (
        _node("d", "document", "Traffic Law\nDocument body is not title", children=("a1", "a2")),
        _node("a1", "article", "Article One\nArticle body must not count", "d", ("c1",)),
        _node("c1", "clause", "1. Shared intro", "a1", ("p1", "p2")),
        _node("p1", "point", "a) First rule", "c1"),
        _node("p2", "point", "b) Second rule", "c1"),
        _node("a2", "article", "Article Two\nFallback body", "d"),
    )
    return LegalDocument(document_id="LAW_TEST", nodes=nodes)


def _rows(passages):
    return [
        {"rank": index, "retrieved_passage_id": passage.passage_id}
        for index, passage in enumerate(passages, start=1)
    ]


def test_component_ratio_k1_and_repeated_heading_clause_document():
    document = _document()
    resolver = LegalTreeResolver(document.nodes)
    counter = ContentTokenCounter()
    for strategy_type, expected_roles in (
        (B4bStrategy, ["article_heading", "point"]),
        (B4dStrategy, ["article_heading", "clause_intro", "point"]),
        (B4eStrategy, ["document_title", "article_heading", "clause_intro", "point"]),
    ):
        passages = strategy_type(
            counter, {"document_titles": {"LAW_TEST": "Traffic Law"}}
        ).build(document)
        point_passages = passages[:2]
        component_map = {
            passage.passage_id: project_b4_components(passage, resolver, counter)
            for passage in passages
        }
        assert [item.role for item in component_map[point_passages[0].passage_id]] == expected_roles
        assert compute_query_redundancy(_rows(point_passages), {p.passage_id: p for p in passages}, component_map, 1)["duplicate_token_ratio"] == 0
        result = compute_query_redundancy(_rows(point_passages), {p.passage_id: p for p in passages}, component_map, 2)
        assert 0 < result["duplicate_token_ratio"] < 1
        assert result["duplicate_token_count"] > 0


def test_article_projection_uses_heading_and_fallback_reconstructs_exactly():
    document = _document()
    resolver = LegalTreeResolver(document.nodes)
    counter = ContentTokenCounter()
    passages = B4eStrategy(counter, {"document_titles": {"LAW_TEST": "Traffic Law"}}).build(document)
    point_components = project_b4_components(passages[0], resolver, counter)
    article = next(item for item in point_components if item.role == "article_heading")
    assert article.text == "Article One"
    assert article.token_count == 2
    fallback = next(passage for passage in passages if passage.primary_node_id == "a2")
    components = project_b4_components(fallback, resolver, counter)
    assert "\n".join(item.text for item in components) == fallback.evidence_text
    assert [item.role for item in components] == ["document_title", "article"]


def test_distinct_articles_are_not_marked_duplicate():
    document = _document()
    resolver = LegalTreeResolver(document.nodes)
    counter = ContentTokenCounter()
    passages = B4bStrategy(counter, {}).build(document)
    selected = [passages[0], passages[-1]]
    component_map = {
        passage.passage_id: project_b4_components(passage, resolver, counter)
        for passage in passages
    }
    result = compute_query_redundancy(_rows(selected), {p.passage_id: p for p in passages}, component_map, 2)
    assert result["duplicate_token_count"] == 0
    assert result["repeated_component_ids"] == []


def _summary(strategy, retriever, recall, mrr, coverage, cost, duplicate):
    return {
        "run_id": f"{strategy}__{retriever.upper()}__v001__dev",
        "retriever": retriever,
        "metrics": {
            "recall@5": recall, "mrr": mrr, "evidence_coverage@5": coverage,
            "avg_evidence_tokens@5": cost, "duplicate_token_ratio@5": duplicate,
        },
    }


def test_selection_keeps_duplicate_ratio_as_tie_break_after_cost():
    summaries = []
    values = {
        "B4a": (.50, .40, .40, 100, .00),
        "B4b": (.55, .42, .44, 110, .10),
        "B4c": (.58, .44, .46, 115, .08),
        "B4d": (.60, .45, .50, 120, .05),
        "B4e": (.61, .30, .30, 200, .90),
    }
    for strategy, metrics in values.items():
        for retriever in ("bm25", "dense"):
            summaries.append(_summary(strategy, retriever, *metrics))
    result = select_round2_candidate(summaries)
    assert result["winner"] == "B4e"  # Recall cannot be overridden by redundancy.

    tied = []
    for strategy in values:
        duplicate = .01 if strategy == "B4d" else .02
        for retriever in ("bm25", "dense"):
            tied.append(_summary(strategy, retriever, .6, .5, .5, 120, duplicate))
    assert select_round2_candidate(tied)["winner"] == "B4d"
