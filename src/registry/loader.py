from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from .models import DocumentRecord, SourceFileRecord


@dataclass(frozen=True)
class Registry:
    root: Path
    version: str
    documents: dict[str, DocumentRecord]
    sources: dict[str, SourceFileRecord]

    def document(self, document_id: str) -> DocumentRecord:
        try:
            return self.documents[document_id]
        except KeyError as exc:
            raise KeyError(f"Unknown document_id: {document_id}") from exc

    def source_for(self, document_id: str) -> SourceFileRecord:
        matches = [item for item in self.sources.values() if item.document_id == document_id]
        if len(matches) != 1:
            raise ValueError(f"Expected one source for {document_id}, found {len(matches)}")
        return matches[0]

    def source_path(self, document_id: str) -> Path:
        path = self.source_for(document_id).local_path
        return path if path.is_absolute() else self.root / path


def load_registry(root: str | Path | None = None) -> Registry:
    root_path = Path(root or Path.cwd()).resolve()
    registry_dir = root_path / "data" / "00_registry"
    with (registry_dir / "documents.yaml").open(encoding="utf-8") as handle:
        document_data = yaml.safe_load(handle)
    with (registry_dir / "source_files.yaml").open(encoding="utf-8") as handle:
        source_data = yaml.safe_load(handle)

    document_list = [DocumentRecord.model_validate(item) for item in document_data["documents"]]
    source_list = [SourceFileRecord.model_validate(item) for item in source_data["source_files"]]
    documents = {item.document_id: item for item in document_list}
    sources = {item.source_file_id: item for item in source_list}
    if len(documents) != len(document_list):
        raise ValueError("Duplicate document_id in registry")
    if len(sources) != len(source_list):
        raise ValueError("Duplicate source_file_id in registry")
    return Registry(root_path, str(document_data.get("version", "0.1.0")), documents, sources)

