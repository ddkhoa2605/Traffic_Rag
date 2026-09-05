import unicodedata

from src.parser.normalizer import normalize_text


def test_unicode_nfc_and_whitespace():
    decomposed = "điều\u00a0  1"
    assert normalize_text(decomposed) == unicodedata.normalize("NFC", "điều 1")


def test_numeric_and_legal_identifiers_are_preserved():
    text = "1.000.000 đồng; 35/2024/QH15; NĐ-CP; đ)"
    assert normalize_text(text) == text

