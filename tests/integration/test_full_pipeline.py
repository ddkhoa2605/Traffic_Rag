from collections import Counter

import yaml

from src.registry.validator import validate_registry
from src.validation.report import validate_document
from src.corpus import load_validated_corpus


def test_registry_is_valid(registry):
    assert validate_registry(registry) == []


def test_both_pdfs_pass_and_match_expected_counts(registry, parsed_documents):
    with (registry.root / "data/06_gold/expected_counts.yaml").open(encoding="utf-8") as handle:
        expected = yaml.safe_load(handle)
    for document_id, (blocks, nodes) in parsed_documents.items():
        report = validate_document(registry, document_id, blocks, nodes)
        assert report.status == "PASS", report.issues
        counts = Counter(node.type for node in nodes)
        assert counts["chapter"] == expected[document_id]["chapters"]
        assert counts["article"] == expected[document_id]["articles"]
        assert counts["clause"] == expected[document_id]["clauses"]
        assert counts["point"] == expected[document_id]["points"]
        assert report.metrics["text_retention"] >= 0.995
        assert report.metrics["provenance_coverage"] == 1.0


def test_validated_corpus_navigation(registry):
    point = load_validated_corpus(registry.root).document("LAW_36_2024").article("11").clause("2").point("đ")
    assert point.id == "LAW_36_2024__A11__C2__Pđ"
    assert point.text.startswith("đ)")
    assert point.source.blocks
