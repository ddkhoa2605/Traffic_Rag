from __future__ import annotations

import hashlib
import json
from pathlib import Path

from src.chunking.token_counter import TokenCounter
from src.legal_tree.loader import load_legal_document
from src.legal_tree.models import LegalNode
from src.legal_tree.resolver import LegalTreeResolver
from src.parser.io import write_json, write_jsonl
from src.registry.loader import Registry, load_registry

from .models import GraphSourceDocument


def canonical_digest(root: str | Path, document_ids: list[str] | None = None) -> str:
    root_path = Path(root).resolve()
    registry = load_registry(root_path)
    ids = document_ids or list(registry.documents)
    digest = hashlib.sha256()
    for document_id in ids:
        path = root_path / "data/05_validated" / document_id / "nodes.jsonl"
        digest.update(document_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def _content_nodes(resolver: LegalTreeResolver, article: LegalNode) -> list[LegalNode]:
    return [
        node
        for node in (article, *resolver.get_descendants(article.id))
        if node.text.strip()
    ]


def _article_hash(
    document_id: str,
    article_id: str,
    title: str,
    nodes: list[LegalNode],
    extraction_text: str,
) -> str:
    payload = {
        "document_id": document_id,
        "article_id": article_id,
        "title": title,
        "nodes": [[node.id, node.content_hash] for node in nodes],
        "extraction_text": extraction_text,
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def project_articles(
    root: str | Path,
    token_counter: TokenCounter,
    *,
    registry: Registry | None = None,
) -> list[GraphSourceDocument]:
    root_path = Path(root).resolve()
    registry = registry or load_registry(root_path)
    result: list[GraphSourceDocument] = []
    for document_id, record in registry.documents.items():
        document = load_legal_document(root_path, document_id)
        resolver = LegalTreeResolver(document.nodes)
        for article in (node for node in resolver.nodes if node.type == "article"):
            nodes = _content_nodes(resolver, article)
            plain_body = "\n".join(node.text.strip() for node in nodes)
            text = f"{record.title}\n{plain_body}".strip()
            marked = "\n\n".join(
                f"[NODE_ID={node.id} TYPE={node.type}]\n{node.text.strip()}"
                for node in nodes
            )
            extraction_text = f"[DOCUMENT_ID={document_id}]\n[TITLE={record.title}]\n\n{marked}"
            token_count = token_counter.count(extraction_text)
            if token_count > token_counter.max_tokens:
                raise ValueError(
                    f"{article.id} has {token_count} extraction tokens; limit is {token_counter.max_tokens}"
                )
            result.append(
                GraphSourceDocument(
                    article_node_id=article.id,
                    canonical_document_id=document_id,
                    title=record.title,
                    text=text,
                    extraction_text=extraction_text,
                    source_node_ids=[node.id for node in nodes],
                    node_texts={node.id: node.text for node in nodes},
                    token_count=token_count,
                    content_hash=_article_hash(
                        document_id, article.id, record.title, nodes, extraction_text
                    ),
                )
            )
    return result


def export_articles(
    root: str | Path,
    output_dir: str | Path,
    token_counter: TokenCounter,
    *,
    b7_dataset_id: str,
) -> dict:
    root_path = Path(root).resolve()
    output_path = Path(output_dir)
    if not output_path.is_absolute():
        output_path = root_path / output_path
    documents = project_articles(root_path, token_counter)
    counts: dict[str, int] = {}
    for item in documents:
        counts[item.canonical_document_id] = counts.get(item.canonical_document_id, 0) + 1
    write_jsonl(output_path / "source_articles.jsonl", [item.model_dump(mode="json") for item in documents])
    manifest = {
        "schema_version": "1.0.0",
        "canonical_digest": canonical_digest(root_path),
        "b7_dataset_id": b7_dataset_id,
        "article_count": len(documents),
        "article_counts_by_document": counts,
        "tokenizer_model": token_counter.model_name,
        "tokenizer_revision": token_counter.revision,
        "max_article_tokens": max(item.token_count for item in documents),
        "source_digest": hashlib.sha256(
            "".join(item.content_hash for item in documents).encode("ascii")
        ).hexdigest(),
    }
    write_json(output_path / "source_manifest.json", manifest)
    return manifest
