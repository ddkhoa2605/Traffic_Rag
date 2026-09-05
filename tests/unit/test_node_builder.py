from src.parser.node_builder import content_hash, stable_node_id


def test_stable_ids():
    assert stable_node_id("LAW_36_2024", "article", article="10") == "LAW_36_2024__A10"
    assert stable_node_id("LAW_36_2024", "clause", article="10", clause="2") == "LAW_36_2024__A10__C2"
    assert stable_node_id("LAW_36_2024", "point", article="10", clause="2", point="đ") == "LAW_36_2024__A10__C2__Pđ"


def test_content_hash_is_deterministic_after_normalization():
    assert content_hash("Nội dung  luật") == content_hash("Nội dung luật")
    assert content_hash("Nội dung luật") != content_hash("Nội dung khác")

