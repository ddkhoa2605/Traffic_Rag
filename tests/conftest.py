from __future__ import annotations

from pathlib import Path

import pytest

from src.parser.pipeline import extract_document, parse_document
from src.registry.loader import load_registry


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def registry():
    return load_registry(ROOT)


@pytest.fixture(scope="session")
def parsed_documents(registry):
    result = {}
    for document_id in registry.documents:
        _, blocks, _ = extract_document(registry, document_id, render_pages=False)
        parsed = parse_document(registry, document_id, blocks)
        result[document_id] = (blocks, parsed.nodes)
    return result

