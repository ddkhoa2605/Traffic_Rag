from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.chunking.builder import OUTPUT_NAMES
from src.chunking.models import RetrievalPassage
from src.legal_tree.loader import load_legal_documents
from src.legal_tree.models import LegalDocument
from src.legal_tree.resolver import LegalTreeResolver
from src.parser.io import read_jsonl


EXPECTED_COUNTS = {"B1": 175, "B4e": 1460}
DIMENSIONS = 1024
NORM_TOLERANCE = 1e-5


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def stable_digest(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class FrozenProjection:
    strategy_id: str
    passages: tuple[RetrievalPassage, ...]
    manifest: dict
    manifest_sha256: str
    cache_metadata: dict
    cache_metadata_sha256: str
    matrix_path: Path
    matrix_sha256: str
    matrix: np.ndarray
    article_node_ids: tuple[str, ...]


@dataclass(frozen=True)
class FrozenRelease:
    dataset_id: str
    release_manifest: dict
    strategy_lock: dict
    strategy_lock_sha256: str
    documents: tuple[LegalDocument, ...]
    projections: tuple[FrozenProjection, ...]

    @property
    def passages(self) -> tuple[RetrievalPassage, ...]:
        return tuple(passage for projection in self.projections for passage in projection.passages)


def _load_projection(
    root: Path,
    strategy_id: str,
    lock: dict,
    resolvers: dict[str, LegalTreeResolver],
    artifact_root: Path,
) -> FrozenProjection:
    folder = artifact_root / OUTPUT_NAMES[strategy_id]
    manifest_path = folder / "manifest.json"
    passages_path = folder / "passages.jsonl"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_sha = sha256_file(manifest_path)
    expected_manifest_sha = lock["passage_manifest_sha256"][strategy_id]
    if manifest_sha != expected_manifest_sha:
        raise ValueError(f"{strategy_id} manifest changed after strategy lock")
    passages = tuple(RetrievalPassage.model_validate(row) for row in read_jsonl(passages_path))
    if len(passages) != EXPECTED_COUNTS[strategy_id] or len(passages) != manifest["passage_count"]:
        raise ValueError(f"Unexpected {strategy_id} passage count: {len(passages)}")

    expected_ids = [passage.passage_id for passage in passages]
    matches: list[tuple[Path, dict]] = []
    for metadata_path in sorted((folder / "embeddings").glob("dense_*.json")):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (
            metadata.get("model") == lock["dense_runtime"]["model"]
            and metadata.get("revision") == lock["dense_runtime"]["revision"]
            and metadata.get("passage_manifest_hash", manifest["passage_digest"])
            == manifest["passage_digest"]
            and metadata.get("passage_ids") == expected_ids
            and metadata_path.with_suffix(".npy").is_file()
        ):
            matches.append((metadata_path, metadata))
    if len(matches) != 1:
        raise ValueError(f"Expected one frozen {strategy_id} embedding cache, found {len(matches)}")
    metadata_path, metadata = matches[0]
    matrix_path = metadata_path.with_suffix(".npy")
    matrix = np.load(matrix_path, allow_pickle=False)
    if matrix.dtype != np.float32 or matrix.shape != (len(passages), DIMENSIONS):
        raise ValueError(f"Invalid {strategy_id} embedding matrix: {matrix.dtype} {matrix.shape}")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{strategy_id} embedding matrix contains non-finite values")
    norm_error = float(np.max(np.abs(np.linalg.norm(matrix, axis=1) - 1.0)))
    if norm_error > NORM_TOLERANCE:
        raise ValueError(f"{strategy_id} embeddings are not normalized: max error {norm_error}")

    article_ids: list[str] = []
    for passage in passages:
        article = resolvers[passage.document_id].get_article(passage.primary_node_id)
        if article is None:
            raise ValueError(f"Passage has no Article ancestor: {passage.passage_id}")
        article_ids.append(article.id)
    return FrozenProjection(
        strategy_id=strategy_id,
        passages=passages,
        manifest=manifest,
        manifest_sha256=manifest_sha,
        cache_metadata=metadata,
        cache_metadata_sha256=sha256_file(metadata_path),
        matrix_path=matrix_path,
        matrix_sha256=sha256_file(matrix_path),
        matrix=matrix,
        article_node_ids=tuple(article_ids),
    )


def load_frozen_release(
    root: str | Path,
    *,
    lock_path: str | Path = "reports/chunk_ablation/strategy_lock.json",
    artifact_root: str | Path = "data/07_retrieval_ablation",
    release_name: str | None = None,
) -> FrozenRelease:
    root_path = Path(root).resolve()
    lock_path = Path(lock_path)
    if not lock_path.is_absolute():
        lock_path = root_path / lock_path
    artifact_root = Path(artifact_root)
    if not artifact_root.is_absolute():
        artifact_root = root_path / artifact_root
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("winner") != "B6" or lock.get("strategies") != ["B6"]:
        raise ValueError("PostgreSQL migration is locked to the frozen B6 winner")
    strategy_lock_sha = sha256_file(lock_path)
    document_ids = ["LAW_35_2024", "LAW_36_2024"]
    documents = tuple(load_legal_documents(root_path, document_ids))
    resolvers = {document.document_id: LegalTreeResolver(document.nodes) for document in documents}
    projections = tuple(
        _load_projection(root_path, strategy_id, lock, resolvers, artifact_root)
        for strategy_id in ("B1", "B4e")
    )
    article, fine = projections
    if article.manifest["document_digests"] != fine.manifest["document_digests"]:
        raise ValueError("B1 and B4e canonical document digests differ")
    release_manifest = {
        "schema_version": "0.2.0" if release_name else "0.1.0",
        "release_name": release_name or "retrieval-law35-36-b6-v0.1.0",
        "corpus_version": article.manifest["corpus_version"],
        "document_digests": article.manifest["document_digests"],
        "article_passage_digest": article.manifest["passage_digest"],
        "fine_passage_digest": fine.manifest["passage_digest"],
        "article_manifest_sha256": article.manifest_sha256,
        "fine_manifest_sha256": fine.manifest_sha256,
        "strategy_lock_sha256": strategy_lock_sha,
        "embedding_model": lock["dense_runtime"]["model"],
        "embedding_revision": lock["dense_runtime"]["revision"],
        "embedding_dimensions": DIMENSIONS,
        "normalized_embeddings": True,
        "embedding_cache_sha256": {
            projection.strategy_id: projection.matrix_sha256 for projection in projections
        },
        "hybrid_b6_config": lock["hybrid_b6_config"],
    }
    if release_name:
        release_manifest["artifact_root"] = (
            artifact_root.relative_to(root_path).as_posix()
            if artifact_root.is_relative_to(root_path) else str(artifact_root)
        )
    dataset_id = stable_digest(release_manifest)
    return FrozenRelease(
        dataset_id=dataset_id,
        release_manifest=release_manifest,
        strategy_lock=lock,
        strategy_lock_sha256=strategy_lock_sha,
        documents=documents,
        projections=projections,
    )
