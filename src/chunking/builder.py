from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import orjson
import yaml

from src.legal_tree.loader import load_legal_document
from src.legal_tree.resolver import LegalTreeResolver
from src.parser.io import write_json, write_jsonl
from src.registry.loader import load_registry

from .article import ArticleStrategy
from .child_parent import ChildParentStrategy
from .clause import ClauseStrategy
from .context_variants import B4aStrategy, B4bStrategy, B4cStrategy, B4dStrategy, B4eStrategy
from .fixed_window import FixedWindowStrategy
from .models import PassageManifest, RetrievalPassage
from .point import PointStrategy
from .point_parent import PointParentStrategy
from .token_counter import BGETokenCounter, TokenCounter
from .validator import require_valid_passages


CHUNK_BUILDER_VERSION = "0.2.0"
DEFAULT_ARTIFACT_ROOT = Path("data/07_retrieval_ablation")
ROUND1_STRATEGIES = {
    "B0": FixedWindowStrategy,
    "B1": ArticleStrategy,
    "B2": ClauseStrategy,
    "B3": PointStrategy,
    "B4": PointParentStrategy,
    "B5": ChildParentStrategy,
}
ROUND2_STRATEGIES = {
    "B4a": B4aStrategy, "B4b": B4bStrategy, "B4c": B4cStrategy,
    "B4d": B4dStrategy, "B4e": B4eStrategy,
}
STRATEGIES = {**ROUND1_STRATEGIES, **ROUND2_STRATEGIES}
OUTPUT_NAMES = {
    "B0": "B0_fixed_window", "B1": "B1_article", "B2": "B2_clause",
    "B3": "B3_point", "B4": "B4_point_clause_context", "B5": "B5_child_parent",
    "B4a": "B4a_point", "B4b": "B4b_article_point", "B4c": "B4c_clause_point",
    "B4d": "B4d_article_clause_point", "B4e": "B4e_document_article_clause_point",
}


def load_chunk_config(root: str | Path) -> dict:
    path = Path(root).resolve() / "configs" / "chunking.yaml"
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _code_digest(root: Path) -> str:
    digest = hashlib.sha256()
    files = sorted((root / "src").rglob("*.py")) + sorted((root / "configs").glob("*.yaml"))
    for path in files:
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _git_state(root: Path) -> tuple[str | None, bool | None]:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True))
        return commit, dirty
    except (OSError, subprocess.SubprocessError):
        return None, None


def _passage_digest(passages: list[RetrievalPassage]) -> str:
    digest = hashlib.sha256()
    for passage in passages:
        digest.update(orjson.dumps(passage.model_dump(mode="json"), option=orjson.OPT_SORT_KEYS))
        digest.update(b"\n")
    return digest.hexdigest()


def _require_content_coverage(document, passages: list[RetrievalPassage], strategy_id: str) -> None:
    resolver = LegalTreeResolver(document.nodes)
    content = [node for node in document.nodes if node.text.strip()]
    if strategy_id == "B0":
        required = {node.id for node in content}
        represented = {node_id for passage in passages for node_id in passage.source_node_ids}
    elif strategy_id == "B1":
        required = {node.id for node in content if node.type == "article"}
        represented = {passage.primary_node_id for passage in passages}
    elif strategy_id == "B2":
        required = {node.id for node in content if node.type == "clause"}
        required |= {
            node.id for node in content if node.type == "article"
            and not any(child.type == "clause" for child in resolver.get_children(node.id))
        }
        represented = {passage.primary_node_id for passage in passages}
    else:
        required = {node.id for node in content if node.type == "point"}
        required |= {
            node.id for node in content if node.type == "clause"
            and not any(child.type == "point" for child in resolver.get_children(node.id))
        }
        required |= {
            node.id for node in content if node.type == "article"
            and not any(child.type == "clause" for child in resolver.get_children(node.id))
        }
        represented = {passage.primary_node_id for passage in passages}
    missing = sorted(required - represented)
    if missing:
        raise ValueError(f"{strategy_id} drops {len(missing)} content-bearing nodes: {missing[:10]}")


def build_passages(
    root: str | Path,
    strategy_ids: list[str] | None = None,
    *,
    token_counter: TokenCounter | None = None,
    artifact_root: str | Path = DEFAULT_ARTIFACT_ROOT,
) -> dict[str, PassageManifest]:
    root_path = Path(root).resolve()
    config = load_chunk_config(root_path)
    selected = strategy_ids or list(ROUND1_STRATEGIES)
    invalid = [name for name in selected if name not in STRATEGIES]
    if invalid:
        raise ValueError(f"Unknown strategies: {invalid}")
    counter = token_counter or BGETokenCounter(config["tokenizer"].get("revision", "main"))
    documents = config["documents"]
    registry = load_registry(root_path)
    document_titles = {
        document_id: registry.document(document_id).title.strip()
        for document_id in documents
    }
    if any(not title for title in document_titles.values()):
        raise ValueError("Registry document title must not be empty")
    artifact_path = Path(artifact_root)
    if not artifact_path.is_absolute():
        artifact_path = root_path / artifact_path
    manifests: dict[str, PassageManifest] = {}
    git_commit, git_dirty = _git_state(root_path)
    code_digest = _code_digest(root_path)
    for strategy_id in selected:
        strategy_config = {
            **config["strategies"].get(strategy_id, {}),
            "document_titles": document_titles,
        }
        strategy = STRATEGIES[strategy_id](counter, strategy_config)
        all_passages: list[RetrievalPassage] = []
        document_digests: dict[str, str] = {}
        for document_id in documents:
            document = load_legal_document(root_path, document_id)
            resolver = LegalTreeResolver(document.nodes)
            passages = strategy.build(document)
            require_valid_passages(passages, resolver, counter)
            _require_content_coverage(document, passages, strategy_id)
            all_passages.extend(passages)
            manifest_path = root_path / "data" / "05_validated" / document_id / "build_manifest.json"
            manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
            document_digests[document_id] = manifest_data["canonical_digest"]
        output = artifact_path / OUTPUT_NAMES[strategy_id]
        write_jsonl(output / "passages.jsonl", (item.model_dump(mode="json") for item in all_passages))
        write_jsonl(output / "regression_snapshot.jsonl", ({
            "passage_id": item.passage_id,
            "primary_node_id": item.primary_node_id,
            "source_node_ids": item.source_node_ids,
            "context_node_ids": item.context_node_ids,
            "citation_node_ids": item.citation_node_ids,
            "index_text_hash": hashlib.sha256(item.index_text.encode()).hexdigest(),
            "evidence_text_hash": hashlib.sha256(item.evidence_text.encode()).hexdigest(),
            "token_count_index": item.token_count_index,
            "token_count_evidence": item.token_count_evidence,
        } for item in all_passages))
        manifest = PassageManifest(
            strategy=strategy.name,
            corpus_version=config["corpus_version"],
            chunk_builder_version=CHUNK_BUILDER_VERSION,
            tokenizer_model=counter.model_name,
            tokenizer_revision=counter.revision,
            documents=documents,
            document_digests=document_digests,
            passage_count=len(all_passages),
            avg_index_tokens=round(sum(item.token_count_index for item in all_passages) / len(all_passages), 3),
            avg_evidence_tokens=round(sum(item.token_count_evidence for item in all_passages) / len(all_passages), 3),
            passage_digest=_passage_digest(all_passages),
            code_digest=code_digest,
            git_commit=git_commit,
            git_dirty=git_dirty,
            generated_at=datetime.now(timezone.utc).isoformat(),
        )
        write_json(output / "manifest.json", manifest.model_dump(mode="json"))
        manifests[strategy_id] = manifest
    return manifests
