from src.parser.models import ExtractedBlock
from src.parser.state_machine import parse_blocks


def block(block_id, page, text, bold=False):
    flags = 16 if bold else 0
    return ExtractedBlock(
        block_id=block_id,
        document_id="LAW_TEST",
        source_file_id="SRC_TEST",
        page=page,
        block=1,
        bbox=(1, 1, 10, 10),
        raw_text=text,
        normalized_text=text,
        lines=[{"bbox": (1, 1, 10, 10), "spans": [{"text": text, "bbox": (1, 1, 10, 10), "font": "Bold" if bold else "Regular", "font_size": 12, "font_flags": flags}]}],
    )


def test_hierarchy_and_multi_page_continuation():
    result = parse_blocks("LAW_TEST", "SRC_TEST", [
        block("b1", 1, "Chương I", bold=True),
        block("b2", 1, "Điều 10. Tiêu đề", bold=True),
        block("b3", 1, "2. Nội dung khoản"),
        block("b4", 1, "đ) Nội dung điểm"),
        block("b5", 2, "tiếp tục sang trang sau"),
    ])
    by_id = {node.id: node for node in result.nodes}
    point = by_id["LAW_TEST__A10__C2__Pđ"]
    assert point.parent_id == "LAW_TEST__A10__C2"
    assert point.source.page_start == 1
    assert point.source.page_end == 2
    assert "tiếp tục sang trang sau" in point.text


def test_non_bold_article_reference_is_continuation():
    result = parse_blocks("LAW_TEST", "SRC_TEST", [
        block("b1", 1, "Điều 1. Heading", bold=True),
        block("b2", 1, "1. Theo quy định tại"),
        block("b3", 1, "Điều 24 của Luật này."),
    ])
    assert [node.hierarchy["article"] for node in result.nodes if node.type == "article"] == ["1"]

