from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np
import pytest

from src.chunking.article import ArticleStrategy
from src.chunking.context_variants import B4eStrategy
from src.chunking.token_counter import EncodedText
from src.legal_tree.models import LegalDocument, LegalNode
from src.legal_tree.resolver import LegalTreeResolver
from src.postgres_store.artifacts import DIMENSIONS, load_frozen_release, sha256_text
from src.postgres_store.dense import validate_query_vector
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


class GuidedFakeRetriever:
    def __init__(self, global_ids, guided):
        self.global_ids = global_ids
        self.guided = guided

    def search(self, query, top_k=10):
        return [
            SearchHit(passage_id=value, score=1 - index / 1000, rank=index)
            for index, value in enumerate(self.global_ids[:top_k], 1)
        ]

    def search_guided(self, query, article_node_ids, *, global_top_n, leaves_per_article):
        global_hits = self.search(query, global_top_n)
        guided = {
            article_id: [
                SearchHit(passage_id=value, score=.5, rank=index)
                for index, value in enumerate(self.guided[article_id][:leaves_per_article], 1)
            ]
            for article_id in article_node_ids
        }
        return global_hits, guided


def test_frozen_release_preserves_locked_hashes_and_projection_counts():
    root = Path(__file__).resolve().parents[2]
    release = load_frozen_release(root)
    assert len(release.dataset_id) == 64
    assert [len(item.passages) for item in release.projections] == [175, 1460]
    assert release.release_manifest["embedding_dimensions"] == 1024
    lock_bytes = (root / "reports/chunk_ablation/strategy_lock.json").read_bytes()
    assert release.strategy_lock_sha256 == hashlib.sha256(lock_bytes).hexdigest()
    passage = release.projections[1].passages[0]
    assert passage.content_hash != sha256_text(passage.index_text)


def test_query_vector_validation():
    vector = np.zeros(DIMENSIONS, dtype=np.float32)
    vector[0] = 1
    assert validate_query_vector(vector).dtype == np.float32
    with pytest.raises(ValueError, match="shape"):
        validate_query_vector(np.ones(3, dtype=np.float32))
    with pytest.raises(ValueError, match="normalized"):
        validate_query_vector(np.ones(DIMENSIONS, dtype=np.float32))
    vector[1] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        validate_query_vector(vector)


def test_guided_candidate_outside_global_pool_is_fused():
    document = _document(); counter = Counter(); resolver = LegalTreeResolver(document.nodes)
    articles = ArticleStrategy(counter, {}).build(document)
    fine = B4eStrategy(counter, {"document_titles": {"D": "Traffic Law"}}).build(document)
    fine_ids = [item.passage_id for item in fine]
    article_ids = [item.passage_id for item in articles]
    guided = {}
    for article in articles:
        values = [
            item.passage_id for item in fine
            if resolver.get_article(item.primary_node_id).id == article.primary_node_id
        ]
        guided[article.primary_node_id] = list(reversed(values))[:3]
    retriever = DualGranularityRetriever(
        FakeRetriever(article_ids), GuidedFakeRetriever(fine_ids[:5], guided),
        articles, fine, {"D": resolver}, HybridConfig(fine_global_top_n=5),
    )
    trace = retriever.search_with_trace("q")
    global_ids = {hit.passage_id for hit in trace.fine_global_hits}
    guided_ids = {hit.passage_id for values in trace.guided_hits.values() for hit in values}
    assert guided_ids - global_ids
    assert any(hit.passage_id in guided_ids - global_ids for hit in trace.hits)


def test_migration_schema_is_versioned_and_exact_only():
    root = Path(__file__).resolve().parents[2]
    sql = (root / "sql/migrations/0001_retrieval_store.sql").read_text(encoding="utf-8")
    assert "PRIMARY KEY (dataset_id, node_id)" in sql
    assert "PRIMARY KEY (dataset_id, passage_id)" in sql
    assert "embedding vector(1024)" in sql
    assert "hnsw" not in sql.casefold() and "ivfflat" not in sql.casefold()
    assert "retrieval_dataset_one_active" in sql
