from __future__ import annotations

import hashlib
import re

from src.chunking.context_variants import B4eStrategy
from src.chunking.article import ArticleStrategy
from src.chunking.token_counter import EncodedText
from src.legal_tree.models import LegalDocument, LegalNode
from src.legal_tree.resolver import LegalTreeResolver
from src.retrieval_eval.evaluator import _article_scout_bundle
from src.retrieval_eval.b6_analysis import select_b6_candidate
from src.retrieval_eval.hybrid import DualGranularityRetriever, HybridConfig
from src.retrieval_eval.models import SearchHit


class Counter:
    model_name = "test"; revision = "v1"; max_tokens = 8192
    def encode_with_offsets(self, text):
        matches = list(re.finditer(r"\S+", text))
        return EncodedText(
            [int.from_bytes(hashlib.sha256(match.group().encode()).digest()[:4], "big") for match in matches],
            [match.span() for match in matches],
        )
    def count(self, text): return len(self.encode_with_offsets(text).token_ids)


class FakeRetriever:
    def __init__(self, passage_ids): self.passage_ids = passage_ids
    def search(self, query, top_k=10):
        return [SearchHit(passage_id=value, score=100-index, rank=index) for index, value in enumerate(self.passage_ids[:top_k], 1)]


def _node(node_id, node_type, text, parent=None, children=()):
    return LegalNode(id=node_id, document_id="D", type=node_type, text=text, parent_id=parent,
                     children_ids=list(children), hierarchy={}, content_hash=f"h-{node_id}")


def _document():
    nodes = [_node("d", "document", "Document title", children=("a1", "a2", "a3"))]
    for article_index in range(1, 4):
        article = f"a{article_index}"; clause = f"c{article_index}"
        points = tuple(f"p{article_index}{point_index}" for point_index in range(1, 6))
        nodes += [_node(article, "article", f"Article {article_index}", "d", (clause,)),
                  _node(clause, "clause", f"{article_index}. Intro", article, points)]
        nodes += [_node(point, "point", f"{point}) rule", clause) for point in points]
    return LegalDocument(document_id="D", nodes=tuple(nodes))


def test_rrf_guidance_caps_and_determinism():
    document = _document(); counter = Counter(); resolver = LegalTreeResolver(document.nodes)
    articles = ArticleStrategy(counter, {}).build(document); fine = B4eStrategy(counter, {"document_titles": {"D": "Traffic Law"}}).build(document)
    article_ids = [item.passage_id for item in articles]
    # Interleave Articles in fine ranking so caps have observable work to do.
    fine_ids = [item.passage_id for item in fine]
    retriever = DualGranularityRetriever(
        FakeRetriever(article_ids), FakeRetriever(fine_ids), articles, fine, {"D": resolver}, HybridConfig()
    )
    first = retriever.search("q"); second = retriever.search("q")
    assert [item.model_dump() for item in first] == [item.model_dump() for item in second]
    assert [item.rank for item in first] == list(range(1, 11))
    assert sum(item.source_strategy == "B1" for item in first[:5]) <= 1
    assert sum(item.source_strategy == "B1" for item in first) <= 2
    fine_per_article = {}
    fine_map = {item.passage_id: item for item in fine}
    for hit in first:
        if hit.source_strategy == "B4e":
            article_id = resolver.get_article(fine_map[hit.passage_id].primary_node_id).id
            fine_per_article[article_id] = fine_per_article.get(article_id, 0) + 1
    assert max(fine_per_article.values()) <= 3
    pool = retriever.search_candidate_pool("q")
    assert len(pool) > 10
    assert [item.rank for item in pool] == list(range(1, len(pool) + 1))
    assert [item.model_dump() for item in pool] == [
        item.model_dump() for item in retriever.search_candidate_pool("q")
    ]


def test_article_bundle_is_granular_deduplicated_and_canonical():
    document = _document(); counter = Counter(); resolver = LegalTreeResolver(document.nodes)
    articles = ArticleStrategy(counter, {}).build(document); fine = B4eStrategy(counter, {"document_titles": {"D": "Traffic Law"}}).build(document)
    article = articles[0]
    descendants = [item for item in fine if resolver.get_article(item.primary_node_id).id == article.primary_node_id][:3]
    bundle, components = _article_scout_bundle(
        article, [item.passage_id for item in descendants], {item.passage_id: item for item in fine}, {"D": resolver}, counter
    )
    assert bundle.primary_node_id == article.primary_node_id
    assert article.primary_node_id in bundle.citation_node_ids
    assert all(item.primary_node_id in bundle.citation_node_ids for item in descendants)
    hashes = [item.token_sequence_hash for item in components]
    assert len(hashes) == len(set(hashes))
    assert bundle.token_count < article.token_count_evidence


def test_rrf_formula_and_guided_rank_are_locked():
    config = HybridConfig(rrf_k=60, article_weight=1.0, fine_weight=1.0)
    assert config.guided_leaves_per_article * (2 - 1) + 3 == 6
    document = _document(); counter = Counter(); resolver = LegalTreeResolver(document.nodes)
    articles = ArticleStrategy(counter, {}).build(document); fine = B4eStrategy(counter, {"document_titles": {"D": "Traffic Law"}}).build(document)
    retriever = DualGranularityRetriever(FakeRetriever([p.passage_id for p in articles]), FakeRetriever([p.passage_id for p in fine]), articles, fine, {"D": resolver}, config)
    assert retriever._rrf(1, 1.0) == 1 / 61


def test_b6_selection_is_strict_recall_first():
    summaries = []
    values = {
        "B6": (.61, .20, .20, 900, .80),
        "B4e": (.60, .90, .90, 100, .01),
        "B4d": (.59, .95, .95, 50, 0),
    }
    for strategy, metric in values.items():
        for retriever in ("bm25", "dense"):
            summaries.append({
                "run_id": f"{strategy}__{retriever.upper()}__v001__dev", "retriever": retriever,
                "metrics": {"recall@5": metric[0], "mrr": metric[1], "evidence_coverage@5": metric[2],
                            "avg_evidence_tokens@5": metric[3], "duplicate_token_ratio@5": metric[4]},
            })
    assert select_b6_candidate(summaries)["winner"] == "B6"
