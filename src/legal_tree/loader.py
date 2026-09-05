from __future__ import annotations

from pathlib import Path

from src.parser.io import read_jsonl

from .models import LegalDocument, LegalNode


def load_legal_document(root: str | Path, document_id: str) -> LegalDocument:
    root_path = Path(root).resolve()
    path = root_path / "data" / "05_validated" / document_id / "nodes.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"Validated canonical nodes not found: {path}")
    nodes = tuple(LegalNode.model_validate(item) for item in read_jsonl(path))
    return LegalDocument(document_id=document_id, nodes=nodes)


def load_legal_documents(root: str | Path, document_ids: list[str]) -> list[LegalDocument]:
    return [load_legal_document(root, document_id) for document_id in document_ids]
