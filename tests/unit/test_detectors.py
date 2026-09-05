import pytest

from src.parser.detectors import detect


@pytest.mark.parametrize(("text", "kind", "key"), [
    ("Chương IV", "chapter", "IV"),
    ("Mục 2", "section", "2"),
    ("Điều 10. Quy định chung", "article", "10"),
    ("12. Nội dung của khoản", "clause", "12"),
    ("d) điểm d", "point", "d"),
    ("đ) điểm đ", "point", "đ"),
    ("Đ) điểm viết hoa", "point", "đ"),
])
def test_detectors(text, kind, key):
    result = detect(text)
    assert result is not None
    assert (result.kind, result.key) == (kind, key)


@pytest.mark.parametrize("text", ["1.000.000 đồng", "35/2024/QH15", "Điều này không có số"])
def test_misleading_numbered_text_is_not_a_legal_marker(text):
    assert detect(text) is None

