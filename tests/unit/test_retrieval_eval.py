from __future__ import annotations

from pathlib import Path

from src.chunking.models import RetrievalPassage
from src.retrieval_eval.bm25 import BM25Retriever, bm25_tokenize
from src.retrieval_eval.dataset import load_dataset, validate_dataset


def test_bm25_unicode_tokenizer_and_stable_tie_break():
    assert bm25_tokenize("  ĐƯỜNG-bộ, đường  ") == ["đường", "bộ", "đường"]
    def passage(passage_id):
        return RetrievalPassage(
            passage_id=passage_id, strategy="B3_point", document_id="d",
            primary_node_id="n", source_node_ids=["n"], index_node_ids=["n"],
            context_node_ids=[], citation_node_ids=["n"], hierarchy={},
            index_text="xe cơ giới", evidence_text="xe cơ giới", display_text="xe cơ giới",
            token_count_index=3, token_count_evidence=3, content_hash="h",
        )
    index = BM25Retriever([passage("p2"), passage("p1")])
    assert [hit.passage_id for hit in index.search("xe", 2)] == ["p1", "p2"]


def test_benchmark_draft_is_valid():
    project_root = Path(__file__).resolve().parents[2]
    assert validate_dataset(project_root) == []
    errors = validate_dataset(project_root, require_approved=True)
    _, gold = load_dataset(project_root)
    assert len(errors) == sum(item.review_status != "approved" for item in gold)
