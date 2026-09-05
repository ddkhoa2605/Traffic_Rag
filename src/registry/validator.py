from __future__ import annotations

import hashlib
from dataclasses import dataclass

from .loader import Registry


@dataclass(frozen=True)
class RegistryIssue:
    code: str
    message: str


def sha256_file(path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def validate_registry(registry: Registry, verify_hashes: bool = True) -> list[RegistryIssue]:
    issues: list[RegistryIssue] = []
    for source in registry.sources.values():
        if source.document_id not in registry.documents:
            issues.append(RegistryIssue("DANGLING_DOCUMENT_ID", source.source_file_id))
            continue
        path = source.local_path if source.local_path.is_absolute() else registry.root / source.local_path
        if not path.is_file():
            issues.append(RegistryIssue("SOURCE_NOT_FOUND", str(path)))
        elif verify_hashes and sha256_file(path) != source.sha256:
            issues.append(RegistryIssue("SHA256_MISMATCH", str(path)))
    for document_id in registry.documents:
        count = sum(source.document_id == document_id for source in registry.sources.values())
        if count != 1:
            issues.append(RegistryIssue("SOURCE_CARDINALITY", f"{document_id}: {count}"))
    return issues
