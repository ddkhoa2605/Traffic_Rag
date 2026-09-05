from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Literal

import orjson
import yaml
from pydantic import BaseModel, Field

from src.legal_tree.loader import load_legal_document
from src.legal_tree.models import LegalNode
from src.legal_tree.resolver import LegalTreeResolver
from src.parser.io import read_jsonl, write_json, write_jsonl


Category = Literal[
    "exact_reference", "semantic_paraphrase", "point_specific",
    "clause_intro_point", "multi_evidence", "hard_distractor",
]
Split = Literal["dev", "test"]


class QueryRecord(BaseModel):
    query_id: str
    query: str = Field(min_length=8)
    category: Category
    split: Split
    document_ids: list[str]
    request_document_ids: list[str] | None = None
    tags: list[str] = Field(default_factory=list)


class GoldRecord(BaseModel):
    query_id: str
    gold_node_ids: list[str]
    required_node_ids: list[str]
    acceptable_parent_ids: list[str] = Field(default_factory=list)
    distractor_node_ids: list[str] = Field(default_factory=list)
    parent_contribution: str = ""
    shared_concepts: list[str] = Field(default_factory=list)
    discriminating_fact: str = ""
    notes: str = ""
    review_status: Literal["draft", "approved", "rejected"] = "draft"
    reviewer: str | None = None
    review_notes: str = ""


def load_dataset(root: str | Path) -> tuple[list[QueryRecord], list[GoldRecord]]:
    base = Path(root).resolve() / "data" / "08_eval"
    queries = [QueryRecord.model_validate(item) for item in read_jsonl(base / "queries.jsonl")]
    gold = [GoldRecord.model_validate(item) for item in read_jsonl(base / "gold.jsonl")]
    return queries, gold


def apply_review_sheet(root: str | Path, review_sheet: str | Path) -> dict:
    """Apply a human-edited review CSV and freeze the version at 120 approvals."""
    root_path = Path(root).resolve()
    queries, current_gold = load_dataset(root_path)
    query_by_id = {item.query_id: item for item in queries}
    by_id = {item.query_id: item for item in current_gold}
    with Path(review_sheet).resolve().open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if {row.get("query_id") for row in rows} != {item.query_id for item in queries}:
        raise ValueError("Review sheet must contain every benchmark query exactly once")
    if len(rows) != len(queries):
        raise ValueError("Review sheet contains duplicate query IDs")
    updated_queries: list[QueryRecord] = []
    updated: list[GoldRecord] = []
    for row in rows:
        old = by_id[row["query_id"]]
        old_query = query_by_id[row["query_id"]]
        if row.get("split", old_query.split) != old_query.split or row.get("category", old_query.category) != old_query.category:
            raise ValueError(f"{old_query.query_id}: split/category cannot be changed by review import")
        if row.get("document_ids") and row["document_ids"].split("|") != old_query.document_ids:
            raise ValueError(f"{old_query.query_id}: document_ids cannot be changed by review import")
        updated_queries.append(old_query.model_copy(update={"query": row.get("query") or old_query.query}))
        updated.append(old.model_copy(update={
            "gold_node_ids": [value for value in row.get("gold_node_ids", "").split("|") if value],
            "required_node_ids": [value for value in row.get("required_node_ids", "").split("|") if value],
            "acceptable_parent_ids": [value for value in row.get("acceptable_parent_ids", "").split("|") if value]
                if "acceptable_parent_ids" in row else old.acceptable_parent_ids,
            "distractor_node_ids": [value for value in row.get("distractor_node_ids", "").split("|") if value]
                if "distractor_node_ids" in row else old.distractor_node_ids,
            "review_status": row.get("review_status", "draft").strip().casefold(),
            "reviewer": row.get("reviewer") or old.reviewer,
            "review_notes": row.get("review_notes", ""),
        }))
    # Validate Pydantic literals before touching the dataset files.
    updated_queries = [QueryRecord.model_validate(item.model_dump()) for item in updated_queries]
    updated = [GoldRecord.model_validate(item.model_dump()) for item in updated]
    output = root_path / "data/08_eval"
    write_jsonl(output / "queries.jsonl", (item.model_dump(mode="json") for item in updated_queries))
    write_jsonl(output / "gold.jsonl", (item.model_dump(mode="json") for item in updated))
    errors = validate_dataset(root_path)
    if errors:
        write_jsonl(output / "queries.jsonl", (item.model_dump(mode="json") for item in queries))
        write_jsonl(output / "gold.jsonl", (item.model_dump(mode="json") for item in current_gold))
        raise ValueError("Reviewed benchmark is invalid: " + "; ".join(errors[:20]))
    manifest = json.loads((output / "benchmark_manifest.json").read_text(encoding="utf-8"))
    manifest["review_counts"] = dict(Counter(item.review_status for item in updated))
    manifest["official_ready"] = all(item.review_status == "approved" for item in updated)
    manifest["benchmark_version"] = "eval-law35-36-v0.1.0" if manifest["official_ready"] else "eval-law35-36-v0.1-draft"
    manifest["dataset_digest"] = hashlib.sha256(
        (output / "queries.jsonl").read_bytes() + (output / "gold.jsonl").read_bytes()
    ).hexdigest()
    write_json(output / "benchmark_manifest.json", manifest)
    return manifest


def export_review_context(root: str | Path, *, output_stem: str = "review_context") -> dict:
    """Export all queries with resolved canonical text and surrounding context."""
    root_path = Path(root).resolve()
    queries, gold_records = load_dataset(root_path)
    gold_by_id = {item.query_id: item for item in gold_records}
    resolvers = {
        document_id: LegalTreeResolver(load_legal_document(root_path, document_id).nodes)
        for document_id in {document_id for query in queries for document_id in query.document_ids}
    }
    output_dir = root_path / "data/08_eval"
    rich_rows = []
    csv_rows = []
    for query in queries:
        gold = gold_by_id[query.query_id]
        requested_ids = list(dict.fromkeys(
            gold.gold_node_ids + gold.required_node_ids + gold.acceptable_parent_ids + gold.distractor_node_ids
        ))
        resolved_nodes = []
        for node_id in requested_ids:
            node = None
            resolver = None
            for document_id in query.document_ids:
                candidate = resolvers[document_id]
                try:
                    node = candidate.get_node(node_id)
                    resolver = candidate
                    break
                except KeyError:
                    continue
            if node is None or resolver is None:
                resolved_nodes.append({"node_id": node_id, "error": "UNRESOLVED"})
                continue
            parent = resolver.get_parent(node.id)
            article = resolver.get_article(node.id)
            clause = resolver.get_clause(node.id)
            resolved_nodes.append({
                "node_id": node.id,
                "type": node.type,
                "hierarchy": node.hierarchy,
                "text": node.text,
                "parent_id": parent.id if parent else None,
                "parent_text": parent.text if parent else None,
                "article_id": article.id if article else None,
                "article_heading": article.text.splitlines()[0] if article and article.text else None,
                "clause_id": clause.id if clause else None,
                "clause_intro": clause.text if clause else None,
            })
        rich_rows.append({
            **query.model_dump(mode="json"),
            **gold.model_dump(mode="json"),
            "resolved_nodes": resolved_nodes,
        })

        def format_nodes(node_ids: list[str]) -> str:
            wanted = set(node_ids)
            parts = []
            for item in resolved_nodes:
                if item["node_id"] not in wanted:
                    continue
                if item.get("error"):
                    parts.append(f"[{item['node_id']}] UNRESOLVED")
                else:
                    parts.append(
                        f"[{item['node_id']}] ({item['type']})\n{item['text']}"
                    )
            return "\n\n---\n\n".join(parts)

        parent_parts = []
        seen_parents = set()
        for item in resolved_nodes:
            parent_id = item.get("parent_id")
            if parent_id and parent_id not in seen_parents:
                seen_parents.add(parent_id)
                parent_parts.append(f"[{parent_id}]\n{item.get('parent_text') or ''}")
        csv_rows.append({
            "query_id": query.query_id,
            "split": query.split,
            "category": query.category,
            "query": query.query,
            "document_ids": "|".join(query.document_ids),
            "gold_node_ids": "|".join(gold.gold_node_ids),
            "gold_node_text": format_nodes(gold.gold_node_ids),
            "required_node_ids": "|".join(gold.required_node_ids),
            "required_node_text": format_nodes(gold.required_node_ids),
            "acceptable_parent_ids": "|".join(gold.acceptable_parent_ids),
            "distractor_node_ids": "|".join(gold.distractor_node_ids),
            "distractor_node_text": format_nodes(gold.distractor_node_ids),
            "parent_contribution": gold.parent_contribution,
            "shared_concepts": "|".join(gold.shared_concepts),
            "discriminating_fact": gold.discriminating_fact,
            "lexical_overlap_ratio": round(_token_overlap_ratio(query.query, format_nodes(gold.gold_node_ids)), 4),
            "longest_common_token_run": _longest_common_token_run(query.query, format_nodes(gold.gold_node_ids)),
            "parent_context": "\n\n---\n\n".join(parent_parts),
            "review_status": gold.review_status,
            "reviewer": gold.reviewer or "",
            "review_notes": gold.review_notes,
        })
    jsonl_path = output_dir / f"{output_stem}.jsonl"
    csv_path = output_dir / f"{output_stem}.csv"
    write_jsonl(jsonl_path, rich_rows)
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    query_csv_path = output_dir / f"{output_stem}_queries.csv"
    query_fields = [
        "query_id", "split", "category", "query", "document_ids",
        "gold_node_ids", "gold_node_text", "required_node_ids", "required_node_text",
        "distractor_node_ids", "distractor_node_text", "parent_contribution",
        "shared_concepts", "discriminating_fact", "lexical_overlap_ratio",
        "longest_common_token_run", "review_status", "reviewer", "review_notes",
    ]
    with query_csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=query_fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in query_fields} for row in csv_rows)
    return {
        "query_count": len(queries),
        "csv": str(csv_path),
        "queries_csv": str(query_csv_path),
        "jsonl": str(jsonl_path),
    }


def apply_benchmark_revisions(root: str | Path, revision_file: str | Path) -> dict:
    """Apply curated post-review revisions and reset changed records to draft."""
    root_path = Path(root).resolve()
    revision_path = Path(revision_file).resolve()
    with revision_path.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    revisions = payload["revisions"]
    queries, gold_records = load_dataset(root_path)
    query_by_id = {item.query_id: item for item in queries}
    gold_by_id = {item.query_id: item for item in gold_records}
    if not set(revisions).issubset(query_by_id):
        raise ValueError(f"Unknown revision IDs: {sorted(set(revisions) - set(query_by_id))}")
    resolvers = {
        document_id: LegalTreeResolver(load_legal_document(root_path, document_id).nodes)
        for document_id in ("LAW_35_2024", "LAW_36_2024")
    }
    all_nodes = {node.id: node for resolver in resolvers.values() for node in resolver.nodes}
    history = root_path / "data/08_eval/history"
    history.mkdir(parents=True, exist_ok=True)
    for source_name, target_name in (
        ("queries.jsonl", "review_round1_queries.jsonl"),
        ("gold.jsonl", "review_round1_gold.jsonl"),
        ("benchmark_manifest.json", "review_round1_manifest.json"),
        ("review.csv", "review_round1.csv"),
    ):
        source = root_path / "data/08_eval" / source_name
        target = history / target_name
        if source.is_file() and not target.exists():
            shutil.copy2(source, target)
    if payload.get("version") == "review-revision-v2":
        for source_name, target_name in (
            ("queries.jsonl", "review_v2_pre_generation_plan_queries.jsonl"),
            ("gold.jsonl", "review_v2_pre_generation_plan_gold.jsonl"),
            ("benchmark_manifest.json", "review_v2_pre_generation_plan_manifest.json"),
            ("review_v2.csv", "review_v2_pre_generation_plan.csv"),
        ):
            source = root_path / "data/08_eval" / source_name
            target = history / target_name
            if source.is_file() and not target.exists():
                shutil.copy2(source, target)
    updated_queries = []
    updated_gold = []
    for query in queries:
        revision = revisions.get(query.query_id)
        gold = gold_by_id[query.query_id]
        if revision is None:
            updated_queries.append(query)
            updated_gold.append(gold)
            continue
        query = query.model_copy(update={
            "query": revision.get("query", query.query),
            "tags": list(dict.fromkeys([*query.tags, "review_round2_revision"])),
        })
        gold_ids = revision.get("gold_node_ids", gold.gold_node_ids)
        required_ids = revision.get("required_node_ids", gold.required_node_ids)
        if revision.get("required_mode") == "clause_and_points":
            clause = all_nodes[gold_ids[0]]
            resolver = resolvers[clause.document_id]
            required_ids = [clause.id, *[
                child.id for child in resolver.get_children(clause.id) if child.type == "point"
            ]]
        primary = all_nodes[gold_ids[0]]
        resolver = resolvers[primary.document_id]
        acceptable = revision.get("acceptable_parent_ids")
        if acceptable is None and gold_ids != gold.gold_node_ids:
            acceptable = [
                node.id for node in resolver.get_ancestors(primary.id)
                if node.type in {"clause", "article"}
            ]
        gold = gold.model_copy(update={
            "gold_node_ids": gold_ids,
            "required_node_ids": required_ids,
            "acceptable_parent_ids": acceptable if acceptable is not None else gold.acceptable_parent_ids,
            "distractor_node_ids": revision.get("distractor_node_ids", gold.distractor_node_ids),
            "parent_contribution": revision.get("parent_contribution", gold.parent_contribution),
            "shared_concepts": revision.get("shared_concepts", gold.shared_concepts),
            "discriminating_fact": revision.get("discriminating_fact", gold.discriminating_fact),
            "notes": revision.get("notes", "Curated after round-1 human review."),
            "review_status": "draft",
            "reviewer": None,
            "review_notes": "Revised after round-1 rejection; requires independent round-2 review.",
        })
        updated_queries.append(QueryRecord.model_validate(query.model_dump()))
        updated_gold.append(GoldRecord.model_validate(gold.model_dump()))
    base = root_path / "data/08_eval"
    write_jsonl(base / "queries.jsonl", (item.model_dump(mode="json") for item in updated_queries))
    write_jsonl(base / "gold.jsonl", (item.model_dump(mode="json") for item in updated_gold))
    errors = validate_dataset(root_path)
    if errors:
        write_jsonl(base / "queries.jsonl", (item.model_dump(mode="json") for item in queries))
        write_jsonl(base / "gold.jsonl", (item.model_dump(mode="json") for item in gold_records))
        raise ValueError("Curated revisions are invalid: " + "; ".join(errors[:50]))
    manifest = json.loads((base / "benchmark_manifest.json").read_text(encoding="utf-8"))
    manifest["benchmark_version"] = "eval-law35-36-v0.1-draft"
    manifest["review_counts"] = dict(Counter(item.review_status for item in updated_gold))
    manifest["official_ready"] = False
    manifest["dataset_digest"] = hashlib.sha256(
        (base / "queries.jsonl").read_bytes() + (base / "gold.jsonl").read_bytes()
    ).hexdigest()
    manifest["revision_source"] = revision_path.name
    manifest["revision_version"] = payload.get("version")
    write_json(base / "benchmark_manifest.json", manifest)
    bundle = export_review_context(root_path, output_stem="review_v2")
    return {
        "revised_count": len(revisions),
        "review_counts": manifest["review_counts"],
        "dataset_digest": manifest["dataset_digest"],
        "history_dir": str(history),
        "review_sheet": str(base / "review_v2.csv"),
        "review_queries_sheet": bundle["queries_csv"],
    }


def _normalize_query(value: str) -> str:
    return re.sub(r"\s+", " ", value.casefold()).strip()


def _tokens(value: str) -> list[str]:
    return re.findall(r"[^\W_]+", _normalize_query(value), flags=re.UNICODE)


def _longest_common_token_run(left: str, right: str) -> int:
    a, b = _tokens(left), _tokens(right)
    previous = [0] * (len(b) + 1)
    longest = 0
    for token_a in a:
        current = [0]
        for index, token_b in enumerate(b, start=1):
            value = previous[index - 1] + 1 if token_a == token_b else 0
            current.append(value)
            longest = max(longest, value)
        previous = current
    return longest


def _token_overlap_ratio(query: str, source: str) -> float:
    query_tokens = set(_tokens(query))
    if not query_tokens:
        return 0.0
    return len(query_tokens & set(_tokens(source))) / len(query_tokens)


def validate_dataset(root: str | Path, *, require_approved: bool = False) -> list[str]:
    root_path = Path(root).resolve()
    queries, gold = load_dataset(root_path)
    errors: list[str] = []
    query_by_id = {item.query_id: item for item in queries}
    gold_by_id = {item.query_id: item for item in gold}
    if len(query_by_id) != len(queries):
        errors.append("duplicate query_id")
    if len(gold_by_id) != len(gold):
        errors.append("duplicate gold query_id")
    if set(query_by_id) != set(gold_by_id):
        errors.append("query/gold IDs differ")
    normalized = [_normalize_query(item.query) for item in queries]
    if len(normalized) != len(set(normalized)):
        errors.append("duplicate normalized query text")
    with (root_path / "data/08_eval/categories.yaml").open(encoding="utf-8") as handle:
        category_config = yaml.safe_load(handle)
    for category, expected in category_config["categories"].items():
        for split in ("dev", "test"):
            count = sum(item.category == category and item.split == split for item in queries)
            if count != expected[split]:
                errors.append(f"{category}/{split}: expected {expected[split]}, got {count}")
    for document_id in ("LAW_35_2024", "LAW_36_2024"):
        count = sum(document_id in item.document_ids for item in queries)
        if count != 60:
            errors.append(f"{document_id}: expected 60 queries, got {count}")
    resolvers = {
        document_id: LegalTreeResolver(load_legal_document(root_path, document_id).nodes)
        for document_id in ("LAW_35_2024", "LAW_36_2024")
    }
    all_nodes = {node.id: node for resolver in resolvers.values() for node in resolver.nodes}
    for query_id, query in query_by_id.items():
        item = gold_by_id[query_id]
        if not item.gold_node_ids or not item.required_node_ids:
            errors.append(f"{query_id}: empty gold/required IDs")
        for node_id in item.gold_node_ids + item.required_node_ids + item.acceptable_parent_ids + item.distractor_node_ids:
            if node_id not in all_nodes:
                errors.append(f"{query_id}: unknown node {node_id}")
            elif all_nodes[node_id].document_id not in query.document_ids:
                errors.append(f"{query_id}: node {node_id} is outside query documents")
        if query.category == "point_specific" and not all_nodes[item.gold_node_ids[0]].type == "point":
            errors.append(f"{query_id}: point_specific gold is not Point")
        if query.category == "clause_intro_point":
            types = {all_nodes[node_id].type for node_id in item.required_node_ids}
            if not {"clause", "point"}.issubset(types):
                errors.append(f"{query_id}: clause_intro_point requires Clause and Point")
            if not any(all_nodes[node_id].type == "point" for node_id in item.gold_node_ids):
                errors.append(f"{query_id}: clause_intro_point exact gold must include a Point")
            point = next((all_nodes[node_id] for node_id in item.gold_node_ids if all_nodes[node_id].type == "point"), None)
            clause = resolvers[point.document_id].get_clause(point.id) if point else None
            if clause is None or clause.id not in item.required_node_ids:
                errors.append(f"{query_id}: clause_intro_point must require the immediate parent Clause")
            if not item.parent_contribution.strip():
                errors.append(f"{query_id}: clause_intro_point requires parent_contribution metadata")
        if query.category == "multi_evidence" and len(set(item.required_node_ids)) < 2:
            errors.append(f"{query_id}: multi_evidence requires >=2 nodes")
        if query.category == "hard_distractor":
            if not item.distractor_node_ids:
                errors.append(f"{query_id}: hard_distractor requires distractor_node_ids")
            if set(item.distractor_node_ids) & set(item.gold_node_ids + item.required_node_ids):
                errors.append(f"{query_id}: distractor overlaps gold/required IDs")
            if not item.shared_concepts or not item.discriminating_fact.strip():
                errors.append(f"{query_id}: hard_distractor requires shared concepts and a discriminating fact")
            gold_types = {all_nodes[node_id].type for node_id in item.gold_node_ids}
            distractor_types = {all_nodes[node_id].type for node_id in item.distractor_node_ids}
            if gold_types != distractor_types:
                errors.append(f"{query_id}: hard_distractor gold/distractor types differ")
        normalized_query = _normalize_query(query.query)
        if "trong thực tế, pháp luật quy định như thế nào về" in normalized_query:
            errors.append(f"{query_id}: generic semantic template")
        if "điều kiện chung và nội dung tại điểm" in normalized_query:
            errors.append(f"{query_id}: generic clause-point template")
        if query.category in {"semantic_paraphrase", "point_specific"}:
            gold_text = " ".join(all_nodes[node_id].text for node_id in item.gold_node_ids)
            if _longest_common_token_run(query.query, gold_text) >= 9:
                errors.append(f"{query_id}: excessive contiguous overlap with gold")
            overlap_limit = 0.65 if query.category == "semantic_paraphrase" else 0.70
            if _token_overlap_ratio(query.query, gold_text) > overlap_limit:
                errors.append(f"{query_id}: excessive lexical overlap with gold")
        if query.category == "exact_reference" and "nội dung gì" in normalized_query:
            for node_id in item.gold_node_ids:
                node = all_nodes[node_id]
                if node.type != "clause":
                    continue
                resolver = resolvers[node.document_id]
                point_ids = {child.id for child in resolver.get_children(node.id) if child.type == "point"}
                if point_ids and not point_ids.issubset(item.required_node_ids):
                    errors.append(f"{query_id}: broad Clause query omits direct Point evidence")
        if require_approved and item.review_status != "approved":
            errors.append(f"{query_id}: gold is not approved")
    return errors


def _spread(items: list[LegalNode], count: int) -> list[LegalNode]:
    if len(items) < count:
        raise ValueError(f"Need {count} candidates, found {len(items)}")
    if count == 1:
        return [items[len(items) // 2]]
    indexes = [round(index * (len(items) - 1) / (count - 1)) for index in range(count)]
    return [items[index] for index in indexes]


def _article_title(article: LegalNode) -> str:
    heading = article.text.splitlines()[0]
    return re.sub(r"^Điều\s+\w+\.?\s*", "", heading, flags=re.IGNORECASE).strip()


def _parents(resolver: LegalTreeResolver, node: LegalNode) -> list[str]:
    return [item.id for item in resolver.get_ancestors(node.id) if item.type in {"clause", "article"}]


def create_benchmark_draft(root: str | Path) -> dict:
    root_path = Path(root).resolve()
    queries: list[QueryRecord] = []
    gold: list[GoldRecord] = []
    sequence = 1
    per_doc = {
        "exact_reference": (6, 3),
        "semantic_paraphrase": (12, 6),
        "point_specific": (8, 4),
        "clause_intro_point": (6, 3),
        "multi_evidence": (4, 2),
        "hard_distractor": (4, 2),
    }

    def add(document_id, category, split, text, gold_ids, required_ids, parents=(), notes="", tags=()):
        nonlocal sequence
        if _normalize_query(text) in {_normalize_query(item.query) for item in queries}:
            number = "35/2024/QH15" if document_id == "LAW_35_2024" else "36/2024/QH15"
            text = f"Theo Luật số {number}, {text[0].lower()}{text[1:]}"
        query_id = f"Q{sequence:04d}"
        sequence += 1
        queries.append(QueryRecord(
            query_id=query_id, query=text, category=category, split=split,
            document_ids=[document_id], tags=[*tags, "draft_generated_for_review"],
        ))
        gold.append(GoldRecord(
            query_id=query_id, gold_node_ids=list(gold_ids), required_node_ids=list(required_ids),
            acceptable_parent_ids=list(parents), notes=notes,
        ))

    for document_id in ("LAW_35_2024", "LAW_36_2024"):
        document = load_legal_document(root_path, document_id)
        resolver = LegalTreeResolver(document.nodes)
        articles = [node for node in resolver.nodes if node.type == "article"]
        clauses = [node for node in resolver.nodes if node.type == "clause"]
        points = [node for node in resolver.nodes if node.type == "point"]
        context_points = [node for node in points if (resolver.get_clause(node.id) and len(resolver.get_clause(node.id).text.split()) >= 8)]
        multi_clauses = [node for node in clauses if len([child for child in resolver.get_children(node.id) if child.type == "point"]) >= 2]

        selected = {
            "exact_reference": _spread(clauses, 9),
            "semantic_paraphrase": _spread(articles, 18),
            "point_specific": _spread(sorted(points, key=lambda n: (n.hierarchy.get("point") != "đ", n.id)), 12),
            "clause_intro_point": _spread(context_points, 9),
            "multi_evidence": _spread(multi_clauses, 6),
            "hard_distractor": _spread(articles[1:], 6),
        }
        for category, (dev_count, test_count) in per_doc.items():
            for local_index, item in enumerate(selected[category]):
                split = "dev" if local_index < dev_count else "test"
                article = resolver.get_article(item.id)
                clause = resolver.get_clause(item.id)
                if category == "exact_reference":
                    query = f"Khoản {item.hierarchy['clause']} Điều {item.hierarchy['article']} quy định nội dung gì?"
                    add(document_id, category, split, query, [item.id], [item.id], _parents(resolver, item), tags=["explicit_legal_citation"])
                elif category == "semantic_paraphrase":
                    title = _article_title(item)
                    query = f"Trong thực tế, pháp luật quy định như thế nào về {title.casefold()}?"
                    add(document_id, category, split, query, [item.id], [item.id], notes="Draft semantic query; verify wording does not copy the heading too closely.", tags=["semantic_draft"])
                elif category == "point_specific":
                    snippet = re.sub(r"^[a-zđ]\)\s*", "", item.text, flags=re.IGNORECASE).split(";")[0][:120]
                    query = f"Quy định cụ thể nào áp dụng đối với nội dung “{snippet}”?"
                    add(document_id, category, split, query, [item.id], [item.id], _parents(resolver, item), tags=["leaf_evidence"])
                elif category == "clause_intro_point":
                    query = f"Khi áp dụng khoản {item.hierarchy['clause']} Điều {item.hierarchy['article']}, điều kiện chung và nội dung tại điểm {item.hierarchy['point']} được hiểu thế nào?"
                    add(document_id, category, split, query, [item.id], [clause.id, item.id], [article.id] if article else [], tags=["parent_context_required"])
                elif category == "multi_evidence":
                    child_points = [child for child in resolver.get_children(item.id) if child.type == "point"][:2]
                    labels = " và ".join(f"điểm {child.hierarchy['point']}" for child in child_points)
                    query = f"{labels.capitalize()} khoản {item.hierarchy['clause']} Điều {item.hierarchy['article']} quy định đồng thời những nội dung nào?"
                    ids = [child.id for child in child_points]
                    add(document_id, category, split, query, ids, ids, [item.id, article.id] if article else [item.id], tags=["two_leaf_evidence"])
                else:
                    title = _article_title(item)
                    nearby = articles[max(0, articles.index(item) - 1)]
                    query = f"Quy định nào điều chỉnh {title.casefold()} khi có các nội dung gần nghĩa dễ gây nhầm lẫn?"
                    add(document_id, category, split, query, [item.id], [item.id], notes=f"Candidate distractor: {nearby.id}", tags=["hard_negative_review_required"])

    output = root_path / "data" / "08_eval"
    write_jsonl(output / "queries.jsonl", (item.model_dump(mode="json") for item in queries))
    write_jsonl(output / "gold.jsonl", (item.model_dump(mode="json") for item in gold))
    review_path = output / "review.csv"
    review_path.parent.mkdir(parents=True, exist_ok=True)
    with review_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["query_id", "split", "category", "query", "gold_node_ids", "required_node_ids", "review_status", "reviewer", "review_notes"])
        gold_by_id = {item.query_id: item for item in gold}
        for query in queries:
            item = gold_by_id[query.query_id]
            writer.writerow([query.query_id, query.split, query.category, query.query, "|".join(item.gold_node_ids), "|".join(item.required_node_ids), item.review_status, item.reviewer or "", item.review_notes])
    digest = hashlib.sha256((output / "queries.jsonl").read_bytes() + (output / "gold.jsonl").read_bytes()).hexdigest()
    canonical = {}
    for document_id in ("LAW_35_2024", "LAW_36_2024"):
        manifest = json.loads((root_path / "data/05_validated" / document_id / "build_manifest.json").read_text(encoding="utf-8"))
        canonical[document_id] = manifest["canonical_digest"]
    manifest = {
        "benchmark_version": "eval-law35-36-v0.1-draft",
        "query_count": len(queries),
        "split_counts": dict(Counter(item.split for item in queries)),
        "category_counts": dict(Counter(item.category for item in queries)),
        "review_counts": dict(Counter(item.review_status for item in gold)),
        "official_ready": all(item.review_status == "approved" for item in gold),
        "dataset_digest": digest,
        "canonical_digests": canonical,
    }
    write_json(output / "benchmark_manifest.json", manifest)
    revision_path = output / "revisions_after_review_v1.yaml"
    if revision_path.is_file():
        apply_benchmark_revisions(root_path, revision_path)
        return json.loads((output / "benchmark_manifest.json").read_text(encoding="utf-8"))
    errors = validate_dataset(root_path)
    if errors:
        raise ValueError("Draft benchmark validation failed: " + "; ".join(errors))
    return manifest
