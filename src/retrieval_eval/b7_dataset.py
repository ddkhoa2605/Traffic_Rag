from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import yaml

from src.legal_tree.loader import load_legal_document
from src.legal_tree.models import LegalNode
from src.legal_tree.resolver import LegalTreeResolver
from src.parser.io import read_jsonl, write_json, write_jsonl
from src.registry.loader import load_registry
from src.retrieval_runtime.reference import (
    normalize_reference_text,
    parse_legal_reference,
    resolve_legal_reference,
)

from .dataset import GoldRecord, QueryRecord


B7_DATA_DIR = Path("data/09_eval_b7")
B7_DRAFT_VERSION = "eval-law35-36-v0.2-draft"
B7_FROZEN_VERSION = "eval-law35-36-v0.2.0"
CATEGORY_QUOTAS = {
    "exact_reference": {"dev": 16, "test": 16},
    "multi_evidence": {"dev": 16, "test": 16},
    "semantic_paraphrase": {"dev": 16, "test": 16},
    "point_specific": {"dev": 12, "test": 12},
    "clause_intro_point": {"dev": 12, "test": 12},
    "hard_distractor": {"dev": 8, "test": 8},
}


def load_b7_research_config(root: str | Path) -> dict:
    with (Path(root).resolve() / "configs/b7.yaml").open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _spread(items: list[LegalNode], count: int) -> list[LegalNode]:
    if len(items) < count:
        raise ValueError(f"Need {count} candidates, found {len(items)}")
    positions = [round(index * (len(items) - 1) / (count - 1)) for index in range(count)]
    return [items[position] for position in positions]


def _heading_subject(article: LegalNode) -> str:
    heading = article.text.splitlines()[0].strip()
    return re.sub(r"^Điều\s+\w+\.?\s*", "", heading, flags=re.IGNORECASE).strip()


def _dataset_digest(base: Path) -> str:
    return hashlib.sha256(
        (base / "queries.jsonl").read_bytes() + (base / "gold.jsonl").read_bytes()
    ).hexdigest()


def load_b7_dataset(root: str | Path) -> tuple[list[QueryRecord], list[GoldRecord]]:
    base = Path(root).resolve() / B7_DATA_DIR
    queries = [QueryRecord.model_validate(row) for row in read_jsonl(base / "queries.jsonl")]
    gold = [GoldRecord.model_validate(row) for row in read_jsonl(base / "gold.jsonl")]
    return queries, gold


def _old_gold_ids(root: Path) -> set[str]:
    path = root / "data/08_eval/gold.jsonl"
    if not path.is_file():
        return set()
    return {
        node_id
        for row in read_jsonl(path)
        for field in ("gold_node_ids", "required_node_ids")
        for node_id in row.get(field, [])
    }


def _write_review_sheet(
    base: Path,
    queries: list[QueryRecord],
    gold: list[GoldRecord],
    nodes: dict[str, LegalNode],
) -> None:
    gold_by_id = {item.query_id: item for item in gold}
    with (base / "review.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        fields = [
            "query_id", "split", "category", "query", "document_ids", "request_document_ids",
            "gold_node_ids", "required_node_ids", "acceptable_parent_ids", "distractor_node_ids",
            "parent_contribution", "shared_concepts", "discriminating_fact", "notes",
            "gold_text", "required_text", "acceptable_parent_text", "distractor_text",
            "review_status", "reviewer", "review_notes",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for query in queries:
            item = gold_by_id[query.query_id]
            writer.writerow({
                "query_id": query.query_id,
                "split": query.split,
                "category": query.category,
                "query": query.query,
                "document_ids": "|".join(query.document_ids),
                "request_document_ids": "|".join(query.request_document_ids or []),
                "gold_node_ids": "|".join(item.gold_node_ids),
                "required_node_ids": "|".join(item.required_node_ids),
                "acceptable_parent_ids": "|".join(item.acceptable_parent_ids),
                "distractor_node_ids": "|".join(item.distractor_node_ids),
                "parent_contribution": item.parent_contribution,
                "shared_concepts": "|".join(item.shared_concepts),
                "discriminating_fact": item.discriminating_fact,
                "notes": item.notes,
                "gold_text": "\n\n---\n\n".join(f"[{node_id}]\n{nodes[node_id].text}" for node_id in item.gold_node_ids),
                "required_text": "\n\n---\n\n".join(f"[{node_id}]\n{nodes[node_id].text}" for node_id in item.required_node_ids),
                "acceptable_parent_text": "\n\n---\n\n".join(f"[{node_id}]\n{nodes[node_id].text}" for node_id in item.acceptable_parent_ids),
                "distractor_text": "\n\n---\n\n".join(f"[{node_id}]\n{nodes[node_id].text}" for node_id in item.distractor_node_ids),
                "review_status": item.review_status,
                "reviewer": item.reviewer or "",
                "review_notes": item.review_notes,
            })


def _write_review_context(
    base: Path,
    queries: list[QueryRecord],
    gold: list[GoldRecord],
    nodes: dict[str, LegalNode],
) -> None:
    gold_by_id = {item.query_id: item for item in gold}
    write_jsonl(base / "review_context.jsonl", ({
        **query.model_dump(mode="json"),
        **gold_by_id[query.query_id].model_dump(mode="json"),
        "resolved_nodes": [
            {
                "node_id": node_id,
                "node_type": nodes[node_id].type,
                "hierarchy": nodes[node_id].hierarchy,
                "text": nodes[node_id].text,
            }
            for node_id in dict.fromkeys([
                *gold_by_id[query.query_id].gold_node_ids,
                *gold_by_id[query.query_id].required_node_ids,
                *gold_by_id[query.query_id].acceptable_parent_ids,
                *gold_by_id[query.query_id].distractor_node_ids,
            ])
        ],
    } for query in queries))


def create_b7_benchmark_draft(root: str | Path) -> dict:
    root_path = Path(root).resolve()
    research_config = load_b7_research_config(root_path)
    category_quotas = research_config["benchmark"]["categories"]
    registry = load_registry(root_path)
    old_ids = _old_gold_ids(root_path)
    queries: list[QueryRecord] = []
    gold: list[GoldRecord] = []
    sequence = 1

    def add(
        document_id: str,
        category: str,
        split: str,
        text: str,
        gold_ids: list[str],
        required_ids: list[str],
        *,
        request_scope: bool = False,
        parents: list[str] | None = None,
        distractors: list[str] | None = None,
        parent_contribution: str = "",
        shared_concepts: list[str] | None = None,
        discriminating_fact: str = "",
        notes: str = "",
        tags: list[str] | None = None,
    ) -> None:
        nonlocal sequence
        query_id = f"B7Q{sequence:04d}"
        sequence += 1
        queries.append(QueryRecord(
            query_id=query_id,
            query=text,
            category=category,
            split=split,
            document_ids=[document_id],
            request_document_ids=[document_id] if request_scope else None,
            tags=[*(tags or []), "b7_fresh_draft", "human_review_required"],
        ))
        gold.append(GoldRecord(
            query_id=query_id,
            gold_node_ids=gold_ids,
            required_node_ids=required_ids,
            acceptable_parent_ids=parents or [],
            distractor_node_ids=distractors or [],
            parent_contribution=parent_contribution,
            shared_concepts=shared_concepts or [],
            discriminating_fact=discriminating_fact,
            notes=notes,
        ))

    per_document = {
        category: (values["dev"] + values["test"]) // 2
        for category, values in category_quotas.items()
    }
    for document_id in ("LAW_35_2024", "LAW_36_2024"):
        document_record = registry.document(document_id)
        resolver = LegalTreeResolver(load_legal_document(root_path, document_id).nodes)
        articles = [node for node in resolver.nodes if node.type == "article" and node.id not in old_ids]
        clauses = [node for node in resolver.nodes if node.type == "clause" and node.id not in old_ids]
        points = [node for node in resolver.nodes if node.type == "point" and node.id not in old_ids]
        multi_clauses = [
            node for node in clauses
            if len([child for child in resolver.get_children(node.id) if child.type == "point" and child.id not in old_ids]) >= 2
        ]
        context_points = [node for node in points if resolver.get_clause(node.id) is not None]
        selected = {
            "exact_reference": _spread(clauses, per_document["exact_reference"]),
            "multi_evidence": _spread(multi_clauses, per_document["multi_evidence"]),
            "semantic_paraphrase": _spread(articles, per_document["semantic_paraphrase"]),
            "point_specific": _spread(points, per_document["point_specific"]),
            "clause_intro_point": _spread(context_points, per_document["clause_intro_point"]),
            "hard_distractor": _spread(articles[1:], per_document["hard_distractor"]),
        }
        for category, nodes in selected.items():
            half = len(nodes) // 2
            for index, node in enumerate(nodes):
                split = "dev" if index < half else "test"
                explicit_law = index % 2 == 0
                prefix = f"Theo {document_record.title}, " if explicit_law else "Trong văn bản đang tra cứu, "
                request_scope = not explicit_law and category in {"exact_reference", "multi_evidence"}
                article = resolver.get_article(node.id)
                clause = resolver.get_clause(node.id)
                if category == "exact_reference":
                    text = (
                        f"{prefix}hãy cho biết toàn bộ quy định tại khoản {node.hierarchy['clause']} "
                        f"Điều {node.hierarchy['article']}?"
                    )
                    direct_points = [child.id for child in resolver.get_children(node.id) if child.type == "point"]
                    add(
                        document_id, category, split, text, [node.id], [node.id, *direct_points],
                        request_scope=request_scope,
                        parents=[article.id] if article else [],
                        tags=["explicit_legal_citation", "explicit_law" if explicit_law else "caller_scoped"],
                    )
                elif category == "multi_evidence":
                    child_points = [
                        child for child in resolver.get_children(node.id)
                        if child.type == "point" and child.id not in old_ids
                    ][:2]
                    labels = " và ".join(f"điểm {child.hierarchy['point']}" for child in child_points)
                    text = (
                        f"{prefix}{labels} khoản {node.hierarchy['clause']} Điều "
                        f"{node.hierarchy['article']} quy định hai nội dung nào?"
                    )
                    ids = [child.id for child in child_points]
                    add(
                        document_id, category, split, text, ids, ids,
                        request_scope=request_scope,
                        parents=[node.id, *([article.id] if article else [])],
                        tags=["explicit_multi_reference", "explicit_law" if explicit_law else "caller_scoped"],
                    )
                elif category == "semantic_paraphrase":
                    subject = _heading_subject(node)
                    text = (
                        f"Theo {document_record.title}, pháp luật điều chỉnh ra sao đối với vấn đề "
                        f"{subject.casefold()} trong thực tiễn?"
                    )
                    add(document_id, category, split, text, [node.id], [node.id], tags=["semantic_fresh"])
                elif category == "point_specific":
                    snippet = re.sub(r"^[a-zđ]\)\s*", "", node.text, flags=re.IGNORECASE).split(";")[0][:100]
                    text = f"Quy tắc chi tiết nào áp dụng cho trường hợp “{snippet}”?"
                    add(
                        document_id, category, split, text, [node.id], [node.id],
                        parents=[ancestor.id for ancestor in resolver.get_ancestors(node.id) if ancestor.type in {"clause", "article"}],
                        tags=["leaf_evidence"],
                    )
                elif category == "clause_intro_point":
                    assert article is not None and clause is not None
                    text = (
                        f"Khi xét điểm {node.hierarchy['point']} khoản {node.hierarchy['clause']} "
                        f"Điều {node.hierarchy['article']}, phần dẫn của khoản ảnh hưởng thế nào đến nội dung điểm?"
                    )
                    add(
                        document_id, category, split, text, [node.id], [clause.id, node.id],
                        parents=[article.id],
                        parent_contribution="Clause intro establishes the scope applied to the requested Point.",
                        tags=["parent_context_required"],
                    )
                else:
                    position = articles.index(node)
                    distractor = articles[position - 1 if position > 0 else position + 1]
                    subject = _heading_subject(node)
                    concepts = [token for token in re.findall(r"\w+", subject.casefold()) if len(token) > 3][:3]
                    text = (
                        f"Trong {document_record.title}, quy định nào trực tiếp điều chỉnh "
                        f"{subject.casefold()}, thay vì vấn đề gần kề dễ nhầm lẫn?"
                    )
                    add(
                        document_id, category, split, text, [node.id], [node.id],
                        distractors=[distractor.id], shared_concepts=concepts or ["quy định"],
                        discriminating_fact=f"Gold heading is '{subject}'; distractor is the adjacent Article.",
                        tags=["hard_negative_review_required"],
                    )

    base = root_path / B7_DATA_DIR
    base.mkdir(parents=True, exist_ok=True)
    write_jsonl(base / "queries.jsonl", (item.model_dump(mode="json") for item in queries))
    write_jsonl(base / "gold.jsonl", (item.model_dump(mode="json") for item in gold))
    all_nodes = {
        node.id: node
        for document_id in registry.documents
        for node in load_legal_document(root_path, document_id).nodes
    }
    _write_review_sheet(base, queries, gold, all_nodes)
    _write_review_context(base, queries, gold, all_nodes)
    create_routing_golden_suite(root_path)
    manifest = {
        "benchmark_version": B7_DRAFT_VERSION,
        "query_count": len(queries),
        "split_counts": dict(Counter(item.split for item in queries)),
        "category_counts": dict(Counter(item.category for item in queries)),
        "document_counts": {
            document_id: sum(document_id in item.document_ids for item in queries)
            for document_id in registry.documents
        },
        "review_counts": dict(Counter(item.review_status for item in gold)),
        "official_ready": False,
        "dataset_digest": _dataset_digest(base),
        "selection_backend": "postgres_dense",
        "v0_1_role": "regression_only",
        "heldout_authorized": False,
        "b7_config_sha256": hashlib.sha256((root_path / "configs/b7.yaml").read_bytes()).hexdigest(),
    }
    write_json(base / "benchmark_manifest.json", manifest)
    errors = validate_b7_dataset(root_path)
    if errors:
        raise ValueError("B7 draft validation failed: " + "; ".join(errors[:30]))
    return manifest


def apply_b7_review_sheet(root: str | Path, review_sheet: str | Path) -> dict:
    root_path = Path(root).resolve()
    queries, old_gold = load_b7_dataset(root_path)
    query_by_id = {item.query_id: item for item in queries}
    old_by_id = {item.query_id: item for item in old_gold}
    with Path(review_sheet).resolve().open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != len(queries) or {row.get("query_id") for row in rows} != {item.query_id for item in queries}:
        raise ValueError("Review sheet must contain every B7 query exactly once")
    updated_queries: list[QueryRecord] = []
    updated: list[GoldRecord] = []
    for row in rows:
        old_query = query_by_id[row["query_id"]]
        old = old_by_id[row["query_id"]]
        if row.get("split", old_query.split) != old_query.split:
            raise ValueError(f"{old_query.query_id}: split cannot be changed during review")
        if row.get("category", old_query.category) != old_query.category:
            raise ValueError(f"{old_query.query_id}: category cannot be changed during review")
        incoming_documents = [value for value in row.get("document_ids", "").split("|") if value]
        if incoming_documents != old_query.document_ids:
            raise ValueError(f"{old_query.query_id}: document_ids cannot be changed during review")
        incoming_scope = [value for value in row.get("request_document_ids", "").split("|") if value]
        if incoming_scope != (old_query.request_document_ids or []):
            raise ValueError(f"{old_query.query_id}: request_document_ids cannot be changed during review")
        updated_queries.append(old_query.model_copy(update={"query": row.get("query") or old_query.query}))
        updated.append(GoldRecord.model_validate(old.model_copy(update={
            "gold_node_ids": [value for value in row.get("gold_node_ids", "").split("|") if value],
            "required_node_ids": [value for value in row.get("required_node_ids", "").split("|") if value],
            "acceptable_parent_ids": [value for value in row.get("acceptable_parent_ids", "").split("|") if value],
            "distractor_node_ids": [value for value in row.get("distractor_node_ids", "").split("|") if value],
            "parent_contribution": row.get("parent_contribution", old.parent_contribution),
            "shared_concepts": [value for value in row.get("shared_concepts", "").split("|") if value],
            "discriminating_fact": row.get("discriminating_fact", old.discriminating_fact),
            "notes": row.get("notes", old.notes),
            "review_status": row.get("review_status", "draft").strip().casefold(),
            "reviewer": row.get("reviewer") or None,
            "review_notes": row.get("review_notes", ""),
        }).model_dump()))
    base = root_path / B7_DATA_DIR
    write_jsonl(base / "queries.jsonl", (item.model_dump(mode="json") for item in updated_queries))
    write_jsonl(base / "gold.jsonl", (item.model_dump(mode="json") for item in updated))
    errors = validate_b7_dataset(root_path)
    if errors:
        write_jsonl(base / "queries.jsonl", (item.model_dump(mode="json") for item in queries))
        write_jsonl(base / "gold.jsonl", (item.model_dump(mode="json") for item in old_gold))
        raise ValueError("Reviewed B7 benchmark is invalid: " + "; ".join(errors[:30]))
    registry = load_registry(root_path)
    all_nodes = {
        node.id: node
        for document_id in registry.documents
        for node in load_legal_document(root_path, document_id).nodes
    }
    _write_review_sheet(base, updated_queries, updated, all_nodes)
    _write_review_context(base, updated_queries, updated, all_nodes)
    manifest = json.loads((base / "benchmark_manifest.json").read_text(encoding="utf-8"))
    manifest["review_counts"] = dict(Counter(item.review_status for item in updated))
    manifest["official_ready"] = all(item.review_status == "approved" for item in updated)
    manifest["benchmark_version"] = B7_FROZEN_VERSION if manifest["official_ready"] else B7_DRAFT_VERSION
    manifest["dataset_digest"] = _dataset_digest(base)
    write_json(base / "benchmark_manifest.json", manifest)
    return manifest


def validate_b7_dataset(root: str | Path, *, require_approved: bool = False) -> list[str]:
    root_path = Path(root).resolve()
    category_quotas = load_b7_research_config(root_path)["benchmark"]["categories"]
    queries, gold = load_b7_dataset(root_path)
    registry = load_registry(root_path)
    resolvers = {
        document_id: LegalTreeResolver(load_legal_document(root_path, document_id).nodes)
        for document_id in registry.documents
    }
    nodes = {node.id: node for resolver in resolvers.values() for node in resolver.nodes}
    errors: list[str] = []
    query_by_id = {item.query_id: item for item in queries}
    gold_by_id = {item.query_id: item for item in gold}
    if len(query_by_id) != len(queries) or len(gold_by_id) != len(gold):
        errors.append("duplicate query_id")
    if set(query_by_id) != set(gold_by_id):
        errors.append("query/gold IDs differ")
    if len(queries) != 160:
        errors.append(f"expected 160 queries, got {len(queries)}")
    normalized = [normalize_reference_text(item.query) for item in queries]
    if len(normalized) != len(set(normalized)):
        errors.append("duplicate normalized query text")
    old_queries = Path(root_path / "data/08_eval/queries.jsonl")
    if old_queries.is_file():
        old_normalized = {normalize_reference_text(row["query"]) for row in read_jsonl(old_queries)}
        overlap = set(normalized) & old_normalized
        if overlap:
            errors.append(f"{len(overlap)} queries duplicate benchmark v0.1")
    for category, splits in category_quotas.items():
        for split, expected in splits.items():
            actual = sum(item.category == category and item.split == split for item in queries)
            if actual != expected:
                errors.append(f"{category}/{split}: expected {expected}, got {actual}")
    for split in ("dev", "test"):
        if sum(item.split == split for item in queries) != 80:
            errors.append(f"{split}: expected 80 queries")
        for document_id in registry.documents:
            actual = sum(item.split == split and document_id in item.document_ids for item in queries)
            if actual != 40:
                errors.append(f"{split}/{document_id}: expected 40, got {actual}")
    for query_id, query in query_by_id.items():
        item = gold_by_id[query_id]
        if not item.gold_node_ids or not item.required_node_ids:
            errors.append(f"{query_id}: empty gold/required IDs")
            continue
        for node_id in [*item.gold_node_ids, *item.required_node_ids, *item.acceptable_parent_ids, *item.distractor_node_ids]:
            if node_id not in nodes:
                errors.append(f"{query_id}: unknown node {node_id}")
            elif nodes[node_id].document_id not in query.document_ids:
                errors.append(f"{query_id}: node outside query document {node_id}")
        if query.category in {"exact_reference", "multi_evidence"}:
            intent = parse_legal_reference(query.query, registry)
            resolution = resolve_legal_reference(intent, resolvers, query.request_document_ids)
            if resolution.status != "RESOLVED":
                errors.append(f"{query_id}: reference route is {resolution.status}, expected RESOLVED")
            elif query.category == "exact_reference" and resolution.resolved_node_ids != item.gold_node_ids:
                errors.append(f"{query_id}: resolved node differs from exact gold")
            elif query.category == "multi_evidence" and set(resolution.resolved_node_ids) != set(item.gold_node_ids):
                errors.append(f"{query_id}: resolved Point set differs from gold")
            if not intent.explicit_document_ids and not query.request_document_ids:
                errors.append(f"{query_id}: explicit reference has neither law alias nor caller scope")
        if query.category == "multi_evidence" and len(item.required_node_ids) < 2:
            errors.append(f"{query_id}: multi_evidence requires at least two nodes")
        if query.category == "clause_intro_point":
            types = {nodes[node_id].type for node_id in item.required_node_ids if node_id in nodes}
            if not {"clause", "point"}.issubset(types) or not item.parent_contribution.strip():
                errors.append(f"{query_id}: invalid Clause+Point evidence metadata")
        if query.category == "hard_distractor" and (
            not item.distractor_node_ids or not item.shared_concepts or not item.discriminating_fact.strip()
        ):
            errors.append(f"{query_id}: incomplete hard-distractor metadata")
        if require_approved and item.review_status != "approved":
            errors.append(f"{query_id}: gold is not approved")
        if item.review_status in {"approved", "rejected"} and not (item.reviewer or "").strip():
            errors.append(f"{query_id}: reviewer is required for {item.review_status}")
        if item.review_status == "rejected" and not item.review_notes.strip():
            errors.append(f"{query_id}: rejected row requires review_notes")
    return errors


def create_routing_golden_suite(root: str | Path) -> list[dict]:
    root_path = Path(root).resolve()
    registry = load_registry(root_path)
    resolvers = {
        document_id: LegalTreeResolver(load_legal_document(root_path, document_id).nodes)
        for document_id in registry.documents
    }
    cases: list[dict] = []

    def add(query: str, status: str, document_ids=None, node_ids=None, group=""):
        cases.append({
            "case_id": f"R{len(cases) + 1:03d}", "group": group, "query": query,
            "document_ids": document_ids, "expected_status": status,
            "expected_node_ids": node_ids or [],
        })

    for document_id in ("LAW_35_2024", "LAW_36_2024"):
        record = registry.document(document_id)
        clauses = [node for node in resolvers[document_id].nodes if node.type == "clause"][:4]
        for index, node in enumerate(clauses):
            alias = record.reference_aliases[index % len(record.reference_aliases)]
            add(
                f"Theo {alias}, khoản {node.hierarchy['clause']} Điều {node.hierarchy['article']} nói gì?",
                "RESOLVED", node_ids=[node.id], group="alias",
            )
    for document_id in ("LAW_35_2024", "LAW_36_2024"):
        resolver = resolvers[document_id]
        point_nodes = [node for node in resolver.nodes if node.type == "point"]
        chosen = [node for node in point_nodes if node.hierarchy.get("point") == "đ"][:2]
        chosen += [node for node in point_nodes if node.hierarchy.get("point") in {"a", "b"}][:2]
        for node in chosen:
            add(
                f"Điểm {node.hierarchy['point']} khoản {node.hierarchy['clause']} Điều {node.hierarchy['article']} quy định gì?",
                "RESOLVED", document_ids=[document_id], node_ids=[node.id], group="unicode_point",
            )
    common = []
    first = resolvers["LAW_35_2024"]
    second = resolvers["LAW_36_2024"]
    for node in first.nodes:
        if node.type != "clause":
            continue
        intent_query = f"Khoản {node.hierarchy['clause']} Điều {node.hierarchy['article']} quy định gì?"
        intent = parse_legal_reference(intent_query, registry)
        if resolve_legal_reference(intent, {"LAW_36_2024": second}).status == "RESOLVED":
            common.append(intent_query)
        if len(common) == 8:
            break
    for query in common:
        add(query, "AMBIGUOUS", group="ambiguous")
    conflict_nodes = [node for node in first.nodes if node.type == "clause"][:2]
    for node in conflict_nodes:
        add(
            f"Theo Luật Đường bộ, khoản {node.hierarchy['clause']} Điều {node.hierarchy['article']} nói gì?",
            "CONFLICT", document_ids=["LAW_36_2024"], group="conflict",
        )
        add(
            f"Theo Luật số 36/2024/QH15, khoản {node.hierarchy['clause']} Điều {node.hierarchy['article']} nói gì?",
            "CONFLICT", document_ids=["LAW_35_2024"], group="conflict",
        )
    add("Khoản 1 được quy định thế nào?", "PARTIAL", group="invalid")
    add("Điểm đ Điều 10 quy định gì?", "PARTIAL", group="invalid")
    add("Khoản 999 Điều 999 quy định gì?", "NOT_FOUND", document_ids=["LAW_35_2024"], group="invalid")
    add("Điều 9999 của Luật Đường bộ quy định gì?", "NOT_FOUND", group="invalid")
    no_reference = [
        "Điều kiện cấp phép được xác định như thế nào?",
        "Người tham gia giao thông cần tuân thủ nguyên tắc nào?",
        "Cơ quan quản lý có trách nhiệm gì trong trường hợp này?",
        "Quy tắc an toàn được áp dụng ra sao?",
        "Hành vi nào bị nghiêm cấm theo pháp luật?",
        "Trách nhiệm bảo trì công trình thuộc về chủ thể nào?",
        "Việc kiểm tra phương tiện được tiến hành như thế nào?",
        "Quyền và nghĩa vụ của người điều khiển gồm những gì?",
    ]
    for query in no_reference:
        add(query, "NO_REFERENCE", group="no_reference")
    if len(cases) != 40:
        raise ValueError(f"Routing golden suite must contain 40 cases, found {len(cases)}")
    output = root_path / B7_DATA_DIR / "routing_golden.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output, cases)
    return cases


def validate_routing_golden_suite(root: str | Path) -> list[str]:
    root_path = Path(root).resolve()
    registry = load_registry(root_path)
    resolvers = {
        document_id: LegalTreeResolver(load_legal_document(root_path, document_id).nodes)
        for document_id in registry.documents
    }
    cases = read_jsonl(root_path / B7_DATA_DIR / "routing_golden.jsonl")
    errors: list[str] = []
    if len(cases) != 40:
        errors.append(f"expected 40 routing cases, got {len(cases)}")
    for case in cases:
        intent = parse_legal_reference(case["query"], registry)
        resolution = resolve_legal_reference(intent, resolvers, case.get("document_ids"))
        if resolution.status != case["expected_status"]:
            errors.append(f"{case['case_id']}: expected {case['expected_status']}, got {resolution.status}")
        if case.get("expected_node_ids") and resolution.resolved_node_ids != case["expected_node_ids"]:
            errors.append(f"{case['case_id']}: resolved node mismatch")
    return errors
