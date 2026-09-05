from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.chunking.models import RetrievalPassage
from src.chunking.token_counter import RegexTokenCounter
from src.legal_tree.loader import load_legal_document
from src.legal_tree.resolver import LegalTreeResolver
from src.registry.loader import load_registry
from src.retrieval_eval.b7_dataset import validate_b7_dataset, validate_routing_golden_suite
from src.retrieval_eval.b7_evaluator import _query_metrics, _redundancy
from src.retrieval_runtime.evidence import build_b7_article_scout_bundle, build_direct_evidence
from src.retrieval_runtime.reference import parse_legal_reference, resolve_legal_reference


ROOT = Path(__file__).resolve().parents[2]


def _resolvers():
    return {
        document_id: LegalTreeResolver(load_legal_document(ROOT, document_id).nodes)
        for document_id in ("LAW_35_2024", "LAW_36_2024")
    }


def _load_passages(folder: str) -> list[RetrievalPassage]:
    passage_path = ROOT / "data/07_retrieval_ablation" / folder / "passages.jsonl"
    return [
        RetrievalPassage.model_validate(json.loads(line))
        for line in passage_path.read_text(encoding="utf-8").splitlines()
    ]


def test_parse_single_reference_and_explicit_law_alias():
    registry = load_registry(ROOT)
    intent = parse_legal_reference(
        "Theo Luật số 35/2024/QH15, Điều 11 khoản 3 quy định gì?", registry,
    )
    assert intent.article == "11" and intent.clause == "3"
    assert intent.explicit_document_ids == ["LAW_35_2024"]
    resolution = resolve_legal_reference(intent, _resolvers())
    assert resolution.status == "RESOLVED"
    assert resolution.resolved_node_ids == ["LAW_35_2024__A11__C3"]


def test_parse_multi_point_list_and_canonical_order():
    registry = load_registry(ROOT)
    intent = parse_legal_reference("Điểm b, a và đ khoản 2 Điều 18 quy định gì?", registry)
    assert intent.points == ["b", "a", "đ"]
    resolution = resolve_legal_reference(intent, _resolvers(), ["LAW_36_2024"])
    if resolution.status == "RESOLVED":
        labels = [node_id.rsplit("__P", 1)[1] for node_id in resolution.resolved_node_ids]
        assert labels == ["a", "b", "đ"]


def test_parse_point_range_uses_vietnamese_legal_order():
    intent = parse_legal_reference(
        "Các điểm c đến e khoản 1 Điều 7 quy định gì?", load_registry(ROOT),
    )
    assert intent.points == ["c", "d", "đ", "e"]


def test_ambiguous_reference_never_selects_a_law():
    registry = load_registry(ROOT)
    intent = parse_legal_reference("Khoản 1 Điều 2 quy định gì?", registry)
    resolution = resolve_legal_reference(intent, _resolvers())
    assert resolution.status == "AMBIGUOUS"
    assert set(resolution.candidate_document_ids) == {"LAW_35_2024", "LAW_36_2024"}
    assert resolution.resolved_node_ids == []


def test_explicit_law_and_caller_scope_conflict():
    registry = load_registry(ROOT)
    intent = parse_legal_reference("Theo Luật Đường bộ, khoản 1 Điều 2 quy định gì?", registry)
    resolution = resolve_legal_reference(intent, _resolvers(), ["LAW_36_2024"])
    assert resolution.status == "CONFLICT"


def test_bare_law_number_is_not_an_alias():
    registry = load_registry(ROOT)
    intent = parse_legal_reference("Điều 35 quy định gì?", registry)
    assert intent.explicit_document_ids == []


def test_multi_point_direct_evidence_deduplicates_parent_context():
    resolver = _resolvers()["LAW_35_2024"]
    document_title = load_registry(ROOT).document("LAW_35_2024").title
    clause = next(
        node for node in resolver.nodes
        if node.type == "clause" and len([c for c in resolver.get_children(node.id) if c.type == "point"]) >= 2
    )
    points = [child for child in resolver.get_children(clause.id) if child.type == "point"][:2]
    direct = build_direct_evidence(
        [point.id for point in points], resolver, {}, RegexTokenCounter(),
        document_title=document_title,
    )
    assert direct.member_node_ids == [point.id for point in points]
    assert direct.bundle.citation_node_ids == [point.id for point in points]
    assert len(direct.bundle.context_node_ids) == 3
    roles = [component.role for component in direct.components]
    assert roles[:3] == ["document_title", "article_heading", "clause_intro"]
    assert roles.count("document_title") == 1
    assert direct.bundle.evidence_text.splitlines()[0] == document_title


@pytest.mark.parametrize("document_id", ["LAW_35_2024", "LAW_36_2024"])
def test_registry_title_reprojects_single_point_without_mutating_frozen_b4e(document_id):
    registry = load_registry(ROOT)
    resolver = _resolvers()[document_id]
    passages = _load_passages("B4e_document_article_clause_point")
    passage = next(
        item for item in passages
        if item.document_id == document_id and resolver.get_node(item.primary_node_id).type == "point"
    )
    frozen_text = passage.evidence_text
    direct = build_direct_evidence(
        [passage.primary_node_id],
        resolver,
        {passage.primary_node_id: passage},
        RegexTokenCounter(),
        document_title=registry.document(document_id).title,
    )
    assert direct.bundle.evidence_text.splitlines()[0] == registry.document(document_id).title
    assert direct.bundle.included_node_ids == passage.source_node_ids
    assert direct.bundle.context_node_ids == passage.context_node_ids
    assert direct.bundle.citation_node_ids == passage.citation_node_ids
    assert passage.evidence_text == frozen_text
    assert direct.bundle.evidence_text != frozen_text
    assert "document_title" in [component.role for component in direct.components]


def test_b7_article_scout_uses_registry_title_and_deduplicates_it():
    registry = load_registry(ROOT)
    resolvers = _resolvers()
    article_passages = _load_passages("B1_article")
    fine_passages = _load_passages("B4e_document_article_clause_point")
    article = next(
        candidate for candidate in article_passages
        if sum(
            passage.document_id == candidate.document_id
            and resolvers[passage.document_id].get_article(passage.primary_node_id).id == candidate.primary_node_id
            for passage in fine_passages
        ) >= 2
    )
    evidence = [
        passage for passage in fine_passages
        if passage.document_id == article.document_id
        and resolvers[passage.document_id].get_article(passage.primary_node_id).id == article.primary_node_id
    ][:2]
    bundle, components = build_b7_article_scout_bundle(
        article,
        [passage.passage_id for passage in evidence],
        {passage.passage_id: passage for passage in evidence},
        resolvers,
        {document_id: record.title for document_id, record in registry.documents.items()},
        RegexTokenCounter(),
    )
    title = registry.document(article.document_id).title
    assert bundle.evidence_text.splitlines()[0] == title
    assert [component.role for component in components].count("document_title") == 1
    assert bundle.context_node_ids[0] == article.document_id
    assert bundle.token_count == RegexTokenCounter().count(bundle.evidence_text)


def test_b7_draft_and_routing_suite_validate():
    assert validate_b7_dataset(ROOT) == []
    assert validate_routing_golden_suite(ROOT) == []


def test_bundle_metrics_and_component_redundancy():
    results = [{
        "rank": 1,
        "primary_node_id": "a",
        "member_node_ids": ["a", "b"],
        "included_node_ids": ["a", "b"],
        "evidence_tokens": 10,
        "evidence_components": [
            {"token_sequence_hash": "same", "token_count": 3},
            {"token_sequence_hash": "leaf-a", "token_count": 2},
        ],
    }, {
        "rank": 2,
        "primary_node_id": "c",
        "member_node_ids": ["c"],
        "included_node_ids": ["c"],
        "evidence_tokens": 7,
        "evidence_components": [
            {"token_sequence_hash": "same", "token_count": 3},
            {"token_sequence_hash": "leaf-c", "token_count": 2},
        ],
    }]
    ratio, duplicate, total = _redundancy(results, 2)
    assert (duplicate, total, ratio) == (3, 10, 0.3)
    metrics = _query_metrics(results, {"gold_node_ids": ["a", "b"], "required_node_ids": ["a", "b"]}, 1)
    assert metrics["recall"] == metrics["bundle_exact_hit"] == metrics["evidence_coverage"] == 1.0
