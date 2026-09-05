from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from src.chunking.models import RetrievalPassage
from src.legal_tree.models import LegalDocument, LegalNode

from src.postgres_store.config import connect


@dataclass(frozen=True)
class ActiveRuntimeCorpus:
    dataset_id: str
    model_name: str
    model_revision: str
    release_manifest: dict
    documents: tuple[LegalDocument, ...]
    article_passages: tuple[RetrievalPassage, ...]
    fine_passages: tuple[RetrievalPassage, ...]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_active_runtime_corpus(
    root: str | Path,
    dsn: str,
    *,
    require_strategy_lock_match: bool = True,
    dataset_id: str | None = None,
    allow_building: bool = False,
    strategy_lock_path: str | Path = "reports/chunk_ablation/strategy_lock.json",
) -> ActiveRuntimeCorpus:
    root_path = Path(root).resolve()
    with connect(dsn) as connection:
        connection.execute("SET TRANSACTION READ ONLY")
        statuses = ["ACTIVE", "BUILDING"] if allow_building else ["ACTIVE"]
        dataset_filter = " AND dataset_id = %s" if dataset_id is not None else ""
        parameters = (statuses, dataset_id) if dataset_id is not None else (statuses,)
        datasets = connection.execute(f"""
            SELECT dataset_id, embedding_model, embedding_revision,
                   strategy_lock_sha256, release_manifest
            FROM traffic_rag.retrieval_dataset
            WHERE status = ANY(%s){dataset_filter}
        """, parameters).fetchall()
        if len(datasets) != 1:
            raise ValueError(f"Runtime requires exactly one selected dataset, found {len(datasets)}")
        dataset_id, model_name, model_revision, lock_sha, release_manifest = datasets[0]
        if require_strategy_lock_match:
            local_lock = Path(strategy_lock_path)
            if not local_lock.is_absolute():
                local_lock = root_path / local_lock
            if not local_lock.is_file() or _sha256(local_lock) != lock_sha:
                raise ValueError("ACTIVE PostgreSQL release does not match the local strategy lock")

        child_rows = connection.execute("""
            SELECT parent_node_id, child_node_id, ordinal
            FROM traffic_rag.legal_node_child
            WHERE dataset_id = %s
            ORDER BY parent_node_id, ordinal
        """, (dataset_id,)).fetchall()
        children: dict[str, list[str]] = defaultdict(list)
        for parent_id, child_id, _ in child_rows:
            children[parent_id].append(child_id)

        node_rows = connection.execute("""
            SELECT node_id, document_id, node_type, hierarchy, title, text_content,
                   parent_node_id, source, content_hash, node_status, metadata
            FROM traffic_rag.legal_node
            WHERE dataset_id = %s
            ORDER BY document_id, source_ordinal
        """, (dataset_id,)).fetchall()
        by_document: dict[str, list[LegalNode]] = defaultdict(list)
        for row in node_rows:
            node = LegalNode(
                id=row[0], document_id=row[1], type=row[2], hierarchy=row[3],
                title=row[4], text=row[5], parent_id=row[6],
                children_ids=children.get(row[0], []), source=row[7],
                content_hash=row[8], node_status=row[9], metadata=row[10],
            )
            by_document[node.document_id].append(node)
        documents = tuple(
            LegalDocument(document_id=document_id, nodes=tuple(nodes))
            for document_id, nodes in sorted(by_document.items())
        )

        link_rows = connection.execute("""
            SELECT passage_id, relation_type, node_id, ordinal
            FROM traffic_rag.passage_node_link
            WHERE dataset_id = %s
            ORDER BY passage_id, relation_type, ordinal
        """, (dataset_id,)).fetchall()
        links: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
        for passage_id, relation_type, node_id, _ in link_rows:
            links[passage_id][relation_type].append(node_id)

        passage_rows = connection.execute("""
            SELECT passage_id, strategy, document_id, primary_node_id, hierarchy,
                   index_text, evidence_text, display_text, token_count_index,
                   token_count_evidence, passage_content_hash
            FROM traffic_rag.retrieval_passage
            WHERE dataset_id = %s
            ORDER BY strategy, passage_id
        """, (dataset_id,)).fetchall()
        passages: list[RetrievalPassage] = []
        for row in passage_rows:
            passage_links = links[row[0]]
            passages.append(RetrievalPassage(
                passage_id=row[0], strategy=row[1], document_id=row[2],
                primary_node_id=row[3], hierarchy=row[4], index_text=row[5],
                evidence_text=row[6], display_text=row[7], token_count_index=row[8],
                token_count_evidence=row[9], content_hash=row[10],
                source_node_ids=passage_links["source"],
                index_node_ids=passage_links["index"],
                context_node_ids=passage_links["context"],
                citation_node_ids=passage_links["citation"],
            ))
    article = tuple(item for item in passages if item.strategy == "B1_article")
    fine = tuple(item for item in passages if item.strategy == "B4e_document_article_clause_point")
    if len(article) != 175 or len(fine) != 1460:
        raise ValueError(f"Unexpected B6 passage counts: B1={len(article)}, B4e={len(fine)}")
    return ActiveRuntimeCorpus(
        dataset_id=dataset_id,
        model_name=model_name,
        model_revision=model_revision,
        release_manifest=release_manifest,
        documents=documents,
        article_passages=article,
        fine_passages=fine,
    )
