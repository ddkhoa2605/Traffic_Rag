from __future__ import annotations

from src.chunking.article import ArticleStrategy
from src.chunking.child_parent import ChildParentStrategy, expand_evidence
from src.chunking.clause import ClauseStrategy
from src.chunking.context_variants import B4aStrategy, B4bStrategy, B4cStrategy, B4dStrategy, B4eStrategy
from src.chunking.fixed_window import FixedWindowStrategy
from src.chunking.models import ExpansionPolicy
from src.chunking.point import PointStrategy
from src.chunking.point_parent import PointParentStrategy
from src.chunking.token_counter import RegexTokenCounter
from src.chunking.validator import validate_passages
from src.legal_tree.models import LegalDocument, LegalNode
from src.legal_tree.resolver import LegalTreeResolver


def make_node(node_id, node_type, text, parent=None, children=(), hierarchy=None):
    return LegalNode(
        id=node_id, document_id="LAW_TEST", type=node_type, text=text,
        parent_id=parent, children_ids=list(children), hierarchy=hierarchy or {},
        content_hash=f"hash-{node_id}",
    )


def fixture_document():
    nodes = (
        make_node("d", "document", "Luật thử", children=("a1", "a2")),
        make_node("a1", "article", "Điều 1. Quy tắc\nDẫn nhập điều", "d", ("c1",), {"article": "1"}),
        make_node("c1", "clause", "1. Điều kiện chung", "a1", ("p1",), {"article": "1", "clause": "1"}),
        make_node("p1", "point", "đ) Nội dung riêng", "c1", hierarchy={"article": "1", "clause": "1", "point": "đ"}),
        make_node("a2", "article", "Điều 2. Không có khoản", "d", hierarchy={"article": "2"}),
    )
    return LegalDocument("LAW_TEST", nodes)


def test_b1_to_b5_fallbacks_and_context_projection():
    document = fixture_document()
    counter = RegexTokenCounter()
    expected_counts = {ArticleStrategy: 2, ClauseStrategy: 2, PointStrategy: 2, PointParentStrategy: 2, ChildParentStrategy: 2}
    resolver = LegalTreeResolver(document.nodes)
    for strategy_type, count in expected_counts.items():
        passages = strategy_type(counter, {}).build(document)
        assert len(passages) == count
        validate_passages(passages, resolver, counter)
    b4 = PointParentStrategy(counter, {}).build(document)[0]
    assert b4.index_text == "Điều 1. Quy tắc\n1. Điều kiện chung\nđ) Nội dung riêng"
    b5 = ChildParentStrategy(counter, {}).build(document)[0]
    assert "Điều kiện chung" not in b5.index_text
    bundle = expand_evidence("p1", ExpansionPolicy(), resolver, counter)
    assert bundle.included_node_ids == ["a1", "c1", "p1"]
    assert bundle.citation_node_ids == ["p1"]


def test_b0_offset_mapping_overlap_and_deepest_primary_tie_break():
    document = fixture_document()
    passages = FixedWindowStrategy(
        RegexTokenCounter(), {"window_tokens": 4, "overlap_tokens": 1}
    ).build(document)
    assert len(passages) > 1
    assert passages[0].passage_id == "B0__LAW_TEST__W0001"
    assert all(item.token_count_index <= 4 for item in passages)
    assert any("p1" in item.source_node_ids for item in passages)


def test_round2_context_variants_isolate_parent_components():
    document = fixture_document()
    counter = RegexTokenCounter()
    texts = {
        strategy.variant: strategy(
            counter, {"document_titles": {"LAW_TEST": "Luật thử"}}
        ).build(document)[0].index_text
        for strategy in (B4aStrategy, B4bStrategy, B4cStrategy, B4dStrategy, B4eStrategy)
    }
    assert texts["B4a"] == "đ) Nội dung riêng"
    assert texts["B4b"] == "Điều 1. Quy tắc\nđ) Nội dung riêng"
    assert texts["B4c"] == "1. Điều kiện chung\nđ) Nội dung riêng"
    assert texts["B4d"] == "Điều 1. Quy tắc\n1. Điều kiện chung\nđ) Nội dung riêng"
    assert texts["B4e"].startswith("Luật thử\nĐiều 1. Quy tắc")


def test_b4e_uses_registry_projection_title_not_document_pdf_header():
    document = fixture_document()
    document.nodes[0].text = "Email: metadata@example.test\nQUỐC HỘI"
    passage = B4eStrategy(
        RegexTokenCounter(),
        {"document_titles": {"LAW_TEST": "Luật thử đúng"}},
    ).build(document)[0]
    assert passage.index_text.splitlines()[0] == "Luật thử đúng"
    assert "metadata@example.test" not in passage.index_text
