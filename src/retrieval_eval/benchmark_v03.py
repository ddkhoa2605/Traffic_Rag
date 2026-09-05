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

from .b7_dataset import _write_review_context, _write_review_sheet
from .dataset import (
    GoldRecord,
    QueryRecord,
    _longest_common_token_run,
    _token_overlap_ratio,
)


CONFIG_PATH = Path("configs/benchmark_v03.yaml")


def load_v03_config(root: str | Path) -> dict:
    root_path = Path(root).resolve()
    return yaml.safe_load((root_path / CONFIG_PATH).read_text(encoding="utf-8"))


def _base(root: Path, config: dict) -> Path:
    return root / config["data_dir"]


def _digest(base: Path) -> str:
    return hashlib.sha256(
        (base / "queries.jsonl").read_bytes() + (base / "gold.jsonl").read_bytes()
    ).hexdigest()


def _old_queries_and_nodes(root: Path, config: dict) -> tuple[set[str], set[str]]:
    normalized_queries: set[str] = set()
    node_ids: set[str] = set()
    for relative in config["novelty"]["exclude_query_text_versions"]:
        path = root / relative / "queries.jsonl"
        if path.is_file():
            normalized_queries.update(normalize_reference_text(row["query"]) for row in read_jsonl(path))
    for relative in config["novelty"]["exclude_gold_required_versions"]:
        path = root / relative / "gold.jsonl"
        if path.is_file():
            for row in read_jsonl(path):
                node_ids.update(row.get("gold_node_ids", []))
                node_ids.update(row.get("required_node_ids", []))
    return normalized_queries, node_ids


def _spread(items: list[LegalNode], count: int) -> list[LegalNode]:
    if len(items) < count:
        raise ValueError(f"Need {count} candidates, found {len(items)}")
    if count == 1:
        return [items[len(items) // 2]]
    positions = [round(index * (len(items) - 1) / (count - 1)) for index in range(count)]
    return [items[position] for position in positions]


def _heading_subject(article: LegalNode) -> str:
    heading = article.text.splitlines()[0].strip()
    return re.sub(r"^Điều\s+\w+\.?\s*", "", heading, flags=re.IGNORECASE).strip()


def _point_body(point: LegalNode) -> str:
    value = re.sub(r"^[a-zđ]\)\s*", "", point.text.strip(), flags=re.IGNORECASE)
    first_sentence = re.split(r"[.;]", value, maxsplit=1)[0].strip()
    return first_sentence[:130].rstrip()


def load_v03_dataset(root: str | Path) -> tuple[list[QueryRecord], list[GoldRecord]]:
    root_path = Path(root).resolve()
    base = _base(root_path, load_v03_config(root_path))
    return (
        [QueryRecord.model_validate(row) for row in read_jsonl(base / "queries.jsonl")],
        [GoldRecord.model_validate(row) for row in read_jsonl(base / "gold.jsonl")],
    )


def create_v03_draft(root: str | Path) -> dict:
    root_path = Path(root).resolve()
    config = load_v03_config(root_path)
    base = _base(root_path, config)
    if base.exists() and any(base.iterdir()):
        raise FileExistsError(f"v0.3 dataset directory already contains artifacts: {base}")
    registry = load_registry(root_path)
    old_queries, old_nodes = _old_queries_and_nodes(root_path, config)
    queries: list[QueryRecord] = []
    gold: list[GoldRecord] = []
    selected_gold_required: set[str] = set()
    sequence = 1

    def add(
        document_id: str,
        category: str,
        split: str,
        query: str,
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
        normalized = normalize_reference_text(query)
        if normalized in old_queries or normalized in {
            normalize_reference_text(item.query) for item in queries
        }:
            raise ValueError(f"Generated query is not novel: {query}")
        if (set(gold_ids) | set(required_ids)) & old_nodes:
            raise ValueError(f"v0.3 gold/required overlaps an older benchmark: {gold_ids}/{required_ids}")
        if set(gold_ids) & selected_gold_required:
            raise ValueError(f"v0.3 exact gold reused: {gold_ids}")
        selected_gold_required.update(required_ids)
        query_id = f"V3Q{sequence:04d}"
        sequence += 1
        queries.append(QueryRecord(
            query_id=query_id,
            query=query,
            category=category,
            split=split,
            document_ids=[document_id],
            request_document_ids=[document_id] if request_scope else None,
            tags=["benchmark_v03", "fresh_gold", "human_review_required", *(tags or [])],
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

    quotas = config["categories"]
    for document_id in ("LAW_35_2024", "LAW_36_2024"):
        record = registry.document(document_id)
        resolver = LegalTreeResolver(load_legal_document(root_path, document_id).nodes)

        def available(node: LegalNode) -> bool:
            return node.id not in old_nodes and node.id not in selected_gold_required

        category_counts = {
            category: (values["dev"] + values["test"]) // 2
            for category, values in quotas.items()
        }

        exact_pool = []
        for node in resolver.nodes:
            if node.type != "clause" or not available(node):
                continue
            direct_points = [child for child in resolver.get_children(node.id) if child.type == "point"]
            if all(available(child) for child in direct_points):
                exact_pool.append(node)
        exact_nodes = _spread(exact_pool, category_counts["exact_reference"])
        for index, node in enumerate(exact_nodes):
            split = "dev" if index < len(exact_nodes) // 2 else "test"
            explicit_law = index % 2 == 0
            prefix = f"Theo Luật số {record.document_number}, " if explicit_law else "Trong văn bản đang được tra cứu, "
            direct_points = [child.id for child in resolver.get_children(node.id) if child.type == "point"]
            article = resolver.get_article(node.id)
            add(
                document_id, "exact_reference", split,
                f"{prefix}hãy trích đầy đủ khoản {node.hierarchy['clause']} Điều {node.hierarchy['article']}, kể cả các điểm trực thuộc nếu có?",
                [node.id], [node.id, *direct_points], request_scope=not explicit_law,
                parents=[article.id] if article else [],
                tags=["explicit_reference", "explicit_law" if explicit_law else "caller_scoped"],
            )

        multi_pool = []
        for node in resolver.nodes:
            if node.type != "clause" or not available(node):
                continue
            children = [child for child in resolver.get_children(node.id) if child.type == "point" and available(child)]
            if len(children) >= 2:
                multi_pool.append(node)
        multi_nodes = _spread(multi_pool, category_counts["multi_evidence"])
        for index, node in enumerate(multi_nodes):
            split = "dev" if index < len(multi_nodes) // 2 else "test"
            explicit_law = index % 2 == 0
            points = [child for child in resolver.get_children(node.id) if child.type == "point" and available(child)][:2]
            labels = " và ".join(f"điểm {point.hierarchy['point']}" for point in points)
            prefix = f"Căn cứ Luật số {record.document_number}, " if explicit_law else "Trong văn bản đang mở, "
            point_ids = [point.id for point in points]
            article = resolver.get_article(node.id)
            add(
                document_id, "multi_evidence", split,
                f"{prefix}hãy đối chiếu {labels} khoản {node.hierarchy['clause']} Điều {node.hierarchy['article']} và nêu riêng nội dung của từng điểm?",
                point_ids, point_ids, request_scope=not explicit_law,
                parents=[node.id, *([article.id] if article else [])],
                tags=["explicit_multi_reference", "explicit_law" if explicit_law else "caller_scoped"],
            )

        article_pool = [node for node in resolver.nodes if node.type == "article" and available(node)]
        semantic_nodes = _spread(article_pool, category_counts["semantic_paraphrase"])
        for index, node in enumerate(semantic_nodes):
            split = "dev" if index < len(semantic_nodes) // 2 else "test"
            subject = _heading_subject(node)
            templates = (
                "Theo {law}, cơ chế pháp lý áp dụng cho {subject} được xác định như thế nào?",
                "Khi phát sinh vấn đề về {subject}, {law} đặt ra những yêu cầu hoặc trách nhiệm nào?",
                "Nội dung điều chỉnh cốt lõi đối với {subject} trong {law} là gì?",
                "Pháp luật đường bộ xử lý vấn đề {subject} theo nguyên tắc nào?",
            )
            query = templates[index % len(templates)].format(law=record.title, subject=subject.casefold())
            add(document_id, "semantic_paraphrase", split, query, [node.id], [node.id], tags=["semantic_novel"])

        point_pool = [node for node in resolver.nodes if node.type == "point" and available(node)]
        point_nodes = _spread(point_pool, category_counts["point_specific"])
        for index, node in enumerate(point_nodes):
            split = "dev" if index < len(point_nodes) // 2 else "test"
            body = _point_body(node)
            templates = (
                "Quy định chi tiết nào trực tiếp áp dụng cho tình huống “{body}”?",
                "Trường hợp “{body}” phải tuân theo yêu cầu cụ thể nào?",
                "Nội dung pháp lý dành riêng cho “{body}” được quy định ra sao?",
            )
            add(
                document_id, "point_specific", split,
                templates[index % len(templates)].format(body=body),
                [node.id], [node.id],
                parents=[ancestor.id for ancestor in resolver.get_ancestors(node.id) if ancestor.type in {"clause", "article"}],
                tags=["leaf_evidence", "lexical_overlap_review"],
            )

        context_pool = [
            node for node in resolver.nodes
            if node.type == "point" and available(node)
            and (resolver.get_clause(node.id) is not None)
            and available(resolver.get_clause(node.id))
        ]
        context_nodes = _spread(context_pool, category_counts["clause_intro_point"])
        for index, node in enumerate(context_nodes):
            split = "dev" if index < len(context_nodes) // 2 else "test"
            clause = resolver.get_clause(node.id)
            article = resolver.get_article(node.id)
            assert clause is not None
            add(
                document_id, "clause_intro_point", split,
                f"Để hiểu đúng điểm {node.hierarchy['point']} khoản {node.hierarchy['clause']} Điều {node.hierarchy['article']}, cần kết hợp phạm vi chung của khoản với quy định tại điểm như thế nào?",
                [node.id], [clause.id, node.id], parents=[article.id] if article else [],
                parent_contribution="Clause intro defines the scope that must be combined with the requested Point.",
                tags=["parent_context_required", "explicit_hierarchy"],
            )

        hard_pool = [node for node in resolver.nodes if node.type == "article" and available(node)]
        hard_nodes = _spread(hard_pool, category_counts["hard_distractor"])
        all_articles = [node for node in resolver.nodes if node.type == "article"]
        for index, node in enumerate(hard_nodes):
            split = "dev" if index < len(hard_nodes) // 2 else "test"
            position = all_articles.index(node)
            candidates = [
                article for article in all_articles
                if article.id != node.id and abs(all_articles.index(article) - position) <= 2
            ]
            distractor = candidates[0] if candidates else all_articles[(position + 1) % len(all_articles)]
            subject = _heading_subject(node)
            distractor_subject = _heading_subject(distractor)
            shared = [token for token in re.findall(r"[^\W_]+", subject.casefold()) if len(token) >= 4][:3]
            add(
                document_id, "hard_distractor", split,
                f"Trong {record.title}, điều khoản nào trực tiếp giải quyết {subject.casefold()}, không phải nội dung gần nghĩa về {distractor_subject.casefold()}?",
                [node.id], [node.id], distractors=[distractor.id],
                shared_concepts=shared or ["quy định"],
                discriminating_fact=f"Gold Article heading is '{subject}'; nearby distractor heading is '{distractor_subject}'.",
                tags=["hard_negative", "paired_article"],
            )

    base.mkdir(parents=True, exist_ok=False)
    write_jsonl(base / "queries.jsonl", (item.model_dump(mode="json") for item in queries))
    write_jsonl(base / "gold.jsonl", (item.model_dump(mode="json") for item in gold))
    all_nodes = {
        node.id: node
        for document_id in registry.documents
        for node in load_legal_document(root_path, document_id).nodes
    }
    _write_review_sheet(base, queries, gold, all_nodes)
    _write_review_context(base, queries, gold, all_nodes)
    manifest = {
        "benchmark_version": config["version_draft"],
        "query_count": len(queries),
        "split_counts": dict(Counter(item.split for item in queries)),
        "category_counts": dict(Counter(item.category for item in queries)),
        "document_counts": {
            document_id: sum(document_id in item.document_ids for item in queries)
            for document_id in registry.documents
        },
        "review_counts": dict(Counter(item.review_status for item in gold)),
        "official_ready": False,
        "dataset_digest": _digest(base),
        "selection_backend": config["selection_backend"],
        "novel_query_overlap_v01_v02": 0,
        "gold_required_overlap_v01_v02": 0,
        "heldout_authorized": False,
        "heldout_consumed": False,
        "test_output_count": 0,
        "source_benchmarks": ["eval-law35-36-v0.1.0", "eval-law35-36-v0.2.0"],
    }
    write_json(base / "benchmark_manifest.json", manifest)
    errors = validate_v03_dataset(root_path)
    if errors:
        raise ValueError("v0.3 draft validation failed: " + "; ".join(errors[:50]))
    return manifest


def validate_v03_dataset(
    root: str | Path,
    *,
    require_approved: bool = False,
    queries_override: list[QueryRecord] | None = None,
    gold_override: list[GoldRecord] | None = None,
    check_artifacts: bool = True,
) -> list[str]:
    root_path = Path(root).resolve()
    config = load_v03_config(root_path)
    base = _base(root_path, config)
    if (queries_override is None) != (gold_override is None):
        raise ValueError("queries_override and gold_override must be provided together")
    if queries_override is None:
        queries, gold = load_v03_dataset(root_path)
    else:
        queries, gold = queries_override, gold_override
    old_queries, old_nodes = _old_queries_and_nodes(root_path, config)
    registry = load_registry(root_path)
    resolvers = {
        document_id: LegalTreeResolver(load_legal_document(root_path, document_id).nodes)
        for document_id in registry.documents
    }
    nodes = {node.id: node for resolver in resolvers.values() for node in resolver.nodes}
    errors: list[str] = []
    query_by_id = {item.query_id: item for item in queries}
    gold_by_id = {item.query_id: item for item in gold}
    if len(queries) != 160 or len(gold) != 160:
        errors.append(f"expected 160 queries/gold, found {len(queries)}/{len(gold)}")
    if len(query_by_id) != len(queries) or len(gold_by_id) != len(gold):
        errors.append("duplicate query_id")
    if set(query_by_id) != set(gold_by_id):
        errors.append("query/gold IDs differ")
    if any(not re.fullmatch(r"V3Q\d{4}", item.query_id) for item in queries):
        errors.append("query IDs must use V3Qxxxx namespace")
    normalized = [normalize_reference_text(item.query) for item in queries]
    if len(normalized) != len(set(normalized)):
        errors.append("duplicate normalized query text inside v0.3")
    overlap_queries = set(normalized) & old_queries
    if overlap_queries:
        errors.append(f"{len(overlap_queries)} query texts overlap v0.1/v0.2")
    current_gold_required = {
        node_id for item in gold for node_id in [*item.gold_node_ids, *item.required_node_ids]
    }
    overlap_nodes = current_gold_required & old_nodes
    if overlap_nodes:
        errors.append(f"{len(overlap_nodes)} gold/required nodes overlap v0.1/v0.2")
    for category, splits in config["categories"].items():
        for split, expected in splits.items():
            actual = sum(item.category == category and item.split == split for item in queries)
            if actual != expected:
                errors.append(f"{category}/{split}: expected {expected}, found {actual}")
            for document_id in registry.documents:
                document_actual = sum(
                    item.category == category and item.split == split and document_id in item.document_ids
                    for item in queries
                )
                if document_actual != expected // 2:
                    errors.append(f"{category}/{split}/{document_id}: expected {expected // 2}, found {document_actual}")
    for split in ("dev", "test"):
        if sum(item.split == split for item in queries) != 80:
            errors.append(f"{split}: expected 80 queries")
        for document_id in registry.documents:
            if sum(item.split == split and document_id in item.document_ids for item in queries) != 40:
                errors.append(f"{split}/{document_id}: expected 40 queries")
    for query_id, query in query_by_id.items():
        item = gold_by_id[query_id]
        if not item.gold_node_ids or not item.required_node_ids:
            errors.append(f"{query_id}: empty gold/required IDs")
            continue
        all_ids = [*item.gold_node_ids, *item.required_node_ids, *item.acceptable_parent_ids, *item.distractor_node_ids]
        for node_id in all_ids:
            if node_id not in nodes:
                errors.append(f"{query_id}: unknown node {node_id}")
            elif nodes[node_id].document_id not in query.document_ids:
                errors.append(f"{query_id}: node outside query document {node_id}")
        if query.category in {"exact_reference", "multi_evidence"}:
            intent = parse_legal_reference(query.query, registry)
            resolution = resolve_legal_reference(intent, resolvers, query.request_document_ids)
            if resolution.status != "RESOLVED":
                errors.append(f"{query_id}: explicit reference resolved as {resolution.status}")
            elif query.category == "exact_reference" and resolution.resolved_node_ids != item.gold_node_ids:
                errors.append(f"{query_id}: exact reference resolution differs from gold")
            elif query.category == "multi_evidence" and set(resolution.resolved_node_ids) != set(item.gold_node_ids):
                errors.append(f"{query_id}: multi-reference resolution differs from gold")
            if not intent.explicit_document_ids and not query.request_document_ids:
                errors.append(f"{query_id}: explicit reference has neither law alias nor request scope")
        if query.category == "point_specific" and nodes[item.gold_node_ids[0]].type != "point":
            errors.append(f"{query_id}: point_specific gold is not Point")
        if query.category == "multi_evidence" and len(set(item.required_node_ids)) < 2:
            errors.append(f"{query_id}: multi_evidence requires at least two nodes")
        if query.category == "clause_intro_point":
            types = {nodes[node_id].type for node_id in item.required_node_ids}
            if not {"clause", "point"}.issubset(types) or not item.parent_contribution.strip():
                errors.append(f"{query_id}: invalid Clause+Point evidence")
        if query.category == "hard_distractor":
            if not item.distractor_node_ids or not item.shared_concepts or not item.discriminating_fact.strip():
                errors.append(f"{query_id}: incomplete hard-distractor metadata")
            if set(item.distractor_node_ids) & set(item.gold_node_ids + item.required_node_ids):
                errors.append(f"{query_id}: distractor overlaps gold/required")
        if require_approved and query.category in {"semantic_paraphrase", "point_specific"}:
            gold_text = " ".join(nodes[node_id].text for node_id in item.gold_node_ids)
            longest_run = _longest_common_token_run(query.query, gold_text)
            overlap_ratio = _token_overlap_ratio(query.query, gold_text)
            overlap_limit = 0.65 if query.category == "semantic_paraphrase" else 0.70
            if longest_run >= 9:
                errors.append(
                    f"{query_id}: excessive contiguous overlap with gold ({longest_run} tokens)"
                )
            if overlap_ratio > overlap_limit:
                errors.append(
                    f"{query_id}: excessive lexical overlap with gold "
                    f"({overlap_ratio:.4f} > {overlap_limit:.2f})"
                )
        if require_approved and item.review_status != "approved":
            errors.append(f"{query_id}: gold is not approved")
        if item.review_status in {"approved", "rejected"} and not (item.reviewer or "").strip():
            errors.append(f"{query_id}: reviewer required for {item.review_status}")
        if item.review_status == "rejected" and not item.review_notes.strip():
            errors.append(f"{query_id}: rejected row requires review_notes")
    if check_artifacts:
        manifest = json.loads((base / "benchmark_manifest.json").read_text(encoding="utf-8"))
        if manifest.get("dataset_digest") != _digest(base):
            errors.append("dataset digest mismatch")
        heldout_authorized = bool(manifest.get("heldout_authorized"))
        heldout_consumed = bool(manifest.get("heldout_consumed"))
        if heldout_consumed and not heldout_authorized:
            errors.append("held-out cannot be consumed before authorization")
        if heldout_authorized:
            lock_path = root_path / "reports/b7_v03/b7_policy_lock.json"
            if not manifest.get("official_ready"):
                errors.append("held-out authorization requires an approved frozen dataset")
            if not lock_path.is_file():
                errors.append("held-out authorization requires the v0.3 policy lock")
            else:
                lock = json.loads(lock_path.read_text(encoding="utf-8"))
                if lock.get("dataset_digest") != manifest.get("dataset_digest"):
                    errors.append("held-out policy lock dataset digest mismatch")
                if lock.get("winner") != manifest.get("locked_winner"):
                    errors.append("held-out policy lock winner mismatch")
        test_runs = list((root_path / "reports/b7_v03/runs").glob("*__test")) if (root_path / "reports/b7_v03/runs").exists() else []
        if not heldout_consumed and test_runs:
            errors.append(f"unconsumed held-out has test artifacts: {len(test_runs)}")
        if heldout_consumed and len(test_runs) != 1:
            errors.append(f"consumed held-out must have exactly one test artifact, found {len(test_runs)}")
        if int(manifest.get("test_output_count", 0)) != len(test_runs):
            errors.append("held-out test_output_count does not match test artifacts")
    return errors


def _review_candidates(
    root_path: Path,
    review_path: str | Path,
) -> tuple[list[QueryRecord], list[GoldRecord]]:
    base = _base(root_path, load_v03_config(root_path))
    queries, current_gold = load_v03_dataset(root_path)
    query_by_id = {item.query_id: item for item in queries}
    gold_by_id = {item.query_id: item for item in current_gold}
    with (base / "review.csv").open(encoding="utf-8-sig", newline="") as handle:
        canonical_rows = list(csv.DictReader(handle))
    with Path(review_path).resolve().open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        incoming_fields = reader.fieldnames or []
        rows = list(reader)
    canonical_fields = list(canonical_rows[0]) if canonical_rows else []
    if incoming_fields != canonical_fields:
        raise ValueError("Review sheet columns/order differ from the canonical v0.3 review sheet")
    if len(rows) != len(queries) or [row.get("query_id") for row in rows] != [item.query_id for item in queries]:
        raise ValueError("Review sheet must preserve every v0.3 query and its order")
    mutable = {
        "query", "gold_node_ids", "required_node_ids", "acceptable_parent_ids",
        "distractor_node_ids", "parent_contribution", "shared_concepts",
        "discriminating_fact", "notes", "review_status", "reviewer", "review_notes",
    }
    for row_number, (canonical, incoming) in enumerate(zip(canonical_rows, rows), start=2):
        for field in canonical_fields:
            if field not in mutable and canonical[field] != incoming[field]:
                raise ValueError(f"row {row_number}: immutable field changed: {field}")
    updated_queries: list[QueryRecord] = []
    updated_gold: list[GoldRecord] = []
    for row in rows:
        query = query_by_id[row["query_id"]]
        item = gold_by_id[row["query_id"]]
        updated_queries.append(QueryRecord.model_validate(query.model_copy(update={
            "query": row.get("query") or query.query,
        }).model_dump(mode="json")))
        updated_gold.append(GoldRecord.model_validate(item.model_copy(update={
            "gold_node_ids": [value for value in row.get("gold_node_ids", "").split("|") if value],
            "required_node_ids": [value for value in row.get("required_node_ids", "").split("|") if value],
            "acceptable_parent_ids": [value for value in row.get("acceptable_parent_ids", "").split("|") if value],
            "distractor_node_ids": [value for value in row.get("distractor_node_ids", "").split("|") if value],
            "parent_contribution": row.get("parent_contribution", item.parent_contribution),
            "shared_concepts": [value for value in row.get("shared_concepts", "").split("|") if value],
            "discriminating_fact": row.get("discriminating_fact", item.discriminating_fact),
            "notes": row.get("notes", item.notes),
            "review_status": row.get("review_status", "draft").strip().casefold(),
            "reviewer": row.get("reviewer") or None,
            "review_notes": row.get("review_notes", ""),
        }).model_dump(mode="json")))
    return updated_queries, updated_gold


def validate_v03_review(root: str | Path, review_path: str | Path) -> dict:
    """Validate a reviewed CSV without mutating canonical benchmark artifacts."""
    root_path = Path(root).resolve()
    try:
        queries, gold = _review_candidates(root_path, review_path)
    except (OSError, KeyError, TypeError, ValueError) as exc:
        return {"status": "ERROR", "errors": [str(exc)]}
    errors = validate_v03_dataset(
        root_path,
        require_approved=True,
        queries_override=queries,
        gold_override=gold,
        check_artifacts=False,
    )
    return {
        "status": "PASS" if not errors else "ERROR",
        "errors": errors,
        "review_count": len(gold),
        "approved_count": sum(item.review_status == "approved" for item in gold),
        "rejected_count": sum(item.review_status == "rejected" for item in gold),
        "query_edit_count": sum(
            incoming.query != current.query
            for incoming, current in zip(queries, load_v03_dataset(root_path)[0])
        ),
    }


def apply_v03_review(
    root: str | Path,
    review_path: str | Path,
    *,
    reviewer_override: str | None = None,
) -> dict:
    root_path = Path(root).resolve()
    config = load_v03_config(root_path)
    base = _base(root_path, config)
    queries, current_gold = load_v03_dataset(root_path)
    manifest_path = base / "benchmark_manifest.json"
    current_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    updated_queries, updated_gold = _review_candidates(root_path, review_path)
    if reviewer_override is not None:
        reviewer = reviewer_override.strip()
        if not reviewer:
            raise ValueError("reviewer_override cannot be blank")
        updated_gold = [item.model_copy(update={"reviewer": reviewer}) for item in updated_gold]
    write_jsonl(base / "queries.jsonl", (item.model_dump(mode="json") for item in updated_queries))
    write_jsonl(base / "gold.jsonl", (item.model_dump(mode="json") for item in updated_gold))
    candidate_manifest = {
        **current_manifest,
        "dataset_digest": _digest(base),
    }
    write_json(manifest_path, candidate_manifest)
    errors = validate_v03_dataset(root_path)
    if errors:
        write_jsonl(base / "queries.jsonl", (item.model_dump(mode="json") for item in queries))
        write_jsonl(base / "gold.jsonl", (item.model_dump(mode="json") for item in current_gold))
        write_json(manifest_path, current_manifest)
        raise ValueError("Reviewed v0.3 is invalid: " + "; ".join(errors[:50]))
    registry = load_registry(root_path)
    all_nodes = {
        node.id: node
        for document_id in registry.documents
        for node in load_legal_document(root_path, document_id).nodes
    }
    _write_review_sheet(base, updated_queries, updated_gold, all_nodes)
    _write_review_context(base, updated_queries, updated_gold, all_nodes)
    manifest = candidate_manifest
    manifest["review_counts"] = dict(Counter(item.review_status for item in updated_gold))
    manifest["official_ready"] = all(item.review_status == "approved" for item in updated_gold)
    manifest["benchmark_version"] = config["version_frozen"] if manifest["official_ready"] else config["version_draft"]
    manifest["dataset_digest"] = _digest(base)
    manifest["review_source_sha256"] = hashlib.sha256(Path(review_path).resolve().read_bytes()).hexdigest()
    manifest["reviewer_override"] = reviewer_override.strip() if reviewer_override else None
    write_json(manifest_path, manifest)
    return manifest
