from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

from src.parser.io import read_json, read_jsonl, write_json, write_jsonl


REVIEW_FIELDS = [
    "provider",
    "candidate_kind",
    "candidate_id",
    "article_node_id",
    "document_id",
    "node_or_relation_type",
    "label_or_description",
    "source_node_id",
    "target_node_id",
    "canonical_node_ids",
    "evidence_text",
    "review_status",
    "verdict",
    "entity_type_correct",
    "relation_type_correct",
    "direction_correct",
    "source_valid",
    "canonical_mapping_valid",
    "missing_important_relation",
    "taxonomy_codes",
    "reviewer",
    "notes",
]
REVIEW_EDITABLE_FIELDS = {
    "review_status",
    "verdict",
    "entity_type_correct",
    "relation_type_correct",
    "direction_correct",
    "source_valid",
    "canonical_mapping_valid",
    "missing_important_relation",
    "taxonomy_codes",
    "reviewer",
    "notes",
}
TAXONOMY_CODES = {f"KG_F{index}" for index in range(1, 13)}


def _round_robin(rows: list[dict], limit: int) -> list[dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row["node_or_relation_type"]].append(row)
    for values in groups.values():
        values.sort(key=lambda item: (item["article_node_id"], item["candidate_id"]))
    result: list[dict] = []
    keys = sorted(groups)
    while len(result) < limit and any(groups.values()):
        for key in keys:
            if groups[key] and len(result) < limit:
                result.append(groups[key].pop(0))
    return result


def export_review(provider: str, pilot_dir: str | Path, output_csv: str | Path) -> dict:
    pilot_path = Path(pilot_dir)
    nodes = read_jsonl(pilot_path / "semantic_nodes.jsonl")
    edges = read_jsonl(pilot_path / "semantic_edges.jsonl")
    node_rows: list[dict] = []
    for node in nodes:
        source_ids = node.get("properties", {}).get("source_node_ids", [])
        provenance = node.get("properties", {}).get("provenance", [])
        article_id = next((value.split("__C", 1)[0].split("__P", 1)[0] for value in source_ids), "")
        node_rows.append(
            {
                "provider": provider,
                "candidate_kind": "entity",
                "candidate_id": node["graph_node_id"],
                "article_node_id": article_id,
                "document_id": node.get("document_id") or "",
                "node_or_relation_type": node["node_type"],
                "label_or_description": json.dumps(
                    {
                        "label": node["label"],
                        "description": node.get("properties", {}).get("description", ""),
                        "aliases": node.get("properties", {}).get("aliases", []),
                    },
                    ensure_ascii=False,
                ),
                "source_node_id": "",
                "target_node_id": "",
                "canonical_node_ids": json.dumps(source_ids, ensure_ascii=False),
                "evidence_text": " | ".join(item["evidence_text"] for item in provenance),
            }
        )
    edge_rows: list[dict] = []
    for edge in edges:
        provenance = edge.get("provenance", [])
        canonical_ids = list(dict.fromkeys(item["canonical_node_id"] for item in provenance))
        article_id = next((value.split("__C", 1)[0].split("__P", 1)[0] for value in canonical_ids), "")
        document_id = article_id.split("__A", 1)[0] if "__A" in article_id else ""
        edge_rows.append(
            {
                "provider": provider,
                "candidate_kind": "relationship",
                "candidate_id": edge["edge_id"],
                "article_node_id": article_id,
                "document_id": document_id,
                "node_or_relation_type": edge["relation_type"],
                "label_or_description": edge.get("description", ""),
                "source_node_id": edge["source_node_id"],
                "target_node_id": edge["target_node_id"],
                "canonical_node_ids": json.dumps(canonical_ids, ensure_ascii=False),
                "evidence_text": " | ".join(item["evidence_text"] for item in provenance),
            }
        )
    rejected_path = pilot_path / "rejected.jsonl"
    if rejected_path.is_file():
        for index, rejected in enumerate(read_jsonl(rejected_path)):
            if rejected.get("reason") != "OTHER_RELATION":
                continue
            value = rejected.get("value", {})
            provenance = value.get("provenance", [])
            canonical_ids = list(
                dict.fromkeys(item.get("canonical_node_id", "") for item in provenance if item.get("canonical_node_id"))
            )
            article_id = next(
                (value.split("__C", 1)[0].split("__P", 1)[0] for value in canonical_ids),
                "",
            )
            edge_rows.append(
                {
                    "provider": provider,
                    "candidate_kind": "relationship",
                    "candidate_id": f"REJECTED::OTHER::{index:06d}",
                    "article_node_id": article_id,
                    "document_id": article_id.split("__A", 1)[0] if "__A" in article_id else "",
                    "node_or_relation_type": "OTHER",
                    "label_or_description": value.get("description", ""),
                    "source_node_id": value.get("source_ref", ""),
                    "target_node_id": value.get("target_ref", ""),
                    "canonical_node_ids": json.dumps(canonical_ids, ensure_ascii=False),
                    "evidence_text": " | ".join(item.get("evidence_text", "") for item in provenance),
                }
            )
    rows = [*_round_robin(node_rows, min(100, len(node_rows))), *_round_robin(edge_rows, min(100, len(edge_rows)))]
    articles = sorted({row["article_node_id"] for row in [*node_rows, *edge_rows] if row["article_node_id"]})
    for article_id in articles:
        rows.append(
            {
                "provider": provider,
                "candidate_kind": "article_coverage",
                "candidate_id": f"COVERAGE::{provider}::{article_id}",
                "article_node_id": article_id,
                "document_id": article_id.split("__A", 1)[0],
                "node_or_relation_type": "COVERAGE",
                "label_or_description": "Đánh dấu nếu thiếu một quan hệ quan trọng rõ ràng trong Article",
                "source_node_id": "",
                "target_node_id": "",
                "canonical_node_ids": json.dumps([article_id]),
                "evidence_text": "",
            }
        )
    for row in rows:
        row.update(
            {
                "review_status": "draft",
                "verdict": "",
                "entity_type_correct": "",
                "relation_type_correct": "",
                "direction_correct": "",
                "source_valid": "",
                "canonical_mapping_valid": "",
                "missing_important_relation": "",
                "taxonomy_codes": "",
                "reviewer": "",
                "notes": "",
            }
        )
    output = Path(output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    baseline_output = output.with_suffix(".baseline.jsonl")
    if output.exists() or baseline_output.exists():
        raise FileExistsError(f"Refusing to overwrite audited review queue: {output}")
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    write_jsonl(baseline_output, rows)
    return {
        "provider": provider,
        "review_count": len(rows),
        "entity_count": sum(row["candidate_kind"] == "entity" for row in rows),
        "relationship_count": sum(row["candidate_kind"] == "relationship" for row in rows),
        "coverage_count": sum(row["candidate_kind"] == "article_coverage" for row in rows),
    }


def load_and_validate_review(
    path: str | Path,
    *,
    baseline_path: str | Path | None = None,
) -> tuple[list[dict], list[str]]:
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = reader.fieldnames or []
    errors: list[str] = []
    missing = [field for field in REVIEW_FIELDS if field not in fieldnames]
    if missing:
        errors.append(f"missing review columns: {missing}")
        return rows, errors
    ids = [row["candidate_id"] for row in rows]
    if len(ids) != len(set(ids)):
        errors.append("duplicate candidate_id")
    if baseline_path is not None:
        baseline_file = Path(baseline_path)
        if not baseline_file.is_file():
            errors.append(f"missing immutable review baseline: {baseline_file}")
        else:
            baseline = read_jsonl(baseline_file)
            if len(rows) != len(baseline):
                errors.append(
                    f"review row count changed: expected {len(baseline)}, found {len(rows)}"
                )
            for line, (row, expected) in enumerate(zip(rows, baseline), start=2):
                for field in REVIEW_FIELDS:
                    if field not in REVIEW_EDITABLE_FIELDS and row.get(field, "") != str(expected.get(field, "")):
                        errors.append(f"line {line}: protected field changed: {field}")
    bool_fields = {
        "entity_type_correct",
        "relation_type_correct",
        "direction_correct",
        "source_valid",
        "canonical_mapping_valid",
        "missing_important_relation",
    }
    for line, row in enumerate(rows, start=2):
        if row["review_status"] != "approved":
            errors.append(f"line {line}: review_status must be approved")
        if row["verdict"] not in {"correct", "strategy_failure", "benchmark_issue"}:
            errors.append(f"line {line}: invalid verdict")
        if not row["reviewer"].strip():
            errors.append(f"line {line}: reviewer is required")
        for field in bool_fields:
            if row[field].strip().casefold() not in {"", "true", "false"}:
                errors.append(f"line {line}: {field} must be true/false or empty")
        kind = row["candidate_kind"]
        required = {
            "entity": ("entity_type_correct", "source_valid", "canonical_mapping_valid"),
            "relationship": (
                "relation_type_correct",
                "direction_correct",
                "source_valid",
                "canonical_mapping_valid",
            ),
            "article_coverage": ("missing_important_relation",),
        }.get(kind)
        if required is None:
            errors.append(f"line {line}: invalid candidate_kind")
        else:
            for field in required:
                if row[field].strip().casefold() not in {"true", "false"}:
                    errors.append(f"line {line}: {field} is required for {kind}")
        taxonomy = {
            value.strip()
            for value in row["taxonomy_codes"].replace(";", ",").split(",")
            if value.strip()
        }
        invalid_taxonomy = taxonomy - TAXONOMY_CODES
        if invalid_taxonomy:
            errors.append(f"line {line}: invalid taxonomy codes: {sorted(invalid_taxonomy)}")
        if row["verdict"] == "correct":
            if taxonomy:
                errors.append(f"line {line}: correct verdict cannot have failure taxonomy")
            if kind == "article_coverage":
                success = row["missing_important_relation"].casefold() == "false"
            else:
                success = all(row[field].casefold() == "true" for field in required or ())
            if not success:
                errors.append(f"line {line}: correct verdict conflicts with review booleans")
        elif row["verdict"] == "strategy_failure":
            if not taxonomy:
                errors.append(f"line {line}: strategy_failure requires KG_F taxonomy")
            if kind == "article_coverage":
                failed = row["missing_important_relation"].casefold() == "true"
            else:
                failed = any(row[field].casefold() == "false" for field in required or ())
            if not failed:
                errors.append(f"line {line}: strategy_failure conflicts with review booleans")
    return rows, errors


def review_metrics(path: str | Path, pilot_manifest: str | Path) -> dict:
    rows, errors = load_and_validate_review(path)
    if errors:
        raise ValueError("; ".join(errors))
    manifest = read_json(Path(pilot_manifest))
    entities = [row for row in rows if row["candidate_kind"] == "entity"]
    relations = [row for row in rows if row["candidate_kind"] == "relationship"]
    coverage = [row for row in rows if row["candidate_kind"] == "article_coverage"]

    def ratio(values: list[dict], predicate) -> float:
        return sum(predicate(row) for row in values) / len(values) if values else 0.0

    accepted_edges = sum(row["verdict"] == "correct" for row in relations)
    return {
        "provider": manifest["provider"],
        "model": manifest["model"],
        "schema_failure_count": manifest.get(
            "schema_failure_count", manifest["failure_count"]
        ),
        "provider_failure_count": manifest["failure_count"],
        "article_count": manifest.get("article_count", 0),
        "entity_precision": ratio(entities, lambda row: row["verdict"] == "correct"),
        "relationship_precision": ratio(relations, lambda row: row["verdict"] == "correct"),
        "relationship_direction_accuracy": ratio(
            relations, lambda row: row["direction_correct"].casefold() == "true"
        ),
        "source_attribution_accuracy": ratio(
            [*entities, *relations], lambda row: row["source_valid"].casefold() == "true"
        ),
        "canonical_mapping_accuracy": ratio(
            [*entities, *relations],
            lambda row: row["canonical_mapping_valid"].casefold() == "true",
        ),
        "missing_important_relation_rate": ratio(
            coverage, lambda row: row["missing_important_relation"].casefold() == "true"
        ),
        "estimated_cost_usd": manifest["estimated_cost_usd"],
        "cost_per_accepted_edge": (
            manifest["estimated_cost_usd"] / accepted_edges if accepted_edges else float("inf")
        ),
        "mean_latency_seconds": (
            manifest["elapsed_seconds"] / manifest["article_count"]
            if manifest["article_count"] else float("inf")
        ),
        "benchmark_issue_count": sum(row["verdict"] == "benchmark_issue" for row in rows),
        "review_count": len(rows),
    }


def select_provider(metrics: list[dict], config: dict) -> dict:
    gates = config["selection"]

    def passes(item: dict) -> bool:
        return (
            item["schema_failure_count"] == 0
            and item.get("provider_failure_count", item["schema_failure_count"]) == 0
            and item["benchmark_issue_count"] == 0
            and item["canonical_mapping_accuracy"] >= gates["canonical_mapping_accuracy"]
            and item["source_attribution_accuracy"] >= gates["source_attribution_accuracy"]
            and item["entity_precision"] >= gates["entity_precision_min"]
            and item["relationship_precision"] >= gates["relationship_precision_min"]
            and item["relationship_direction_accuracy"] >= gates["relationship_direction_accuracy_min"]
            and item["missing_important_relation_rate"] <= gates["missing_important_relation_rate_max"]
        )

    eligible = [item for item in metrics if passes(item)]
    if not eligible:
        return {"winner": None, "status": "NO_MODEL_PASSED", "metrics": metrics}
    eligible.sort(key=lambda item: item["provider"])
    best_precision = max(item["relationship_precision"] for item in eligible)
    delta = gates["meaningful_relationship_precision_delta"]
    materially_best = [
        item for item in eligible if best_precision - item["relationship_precision"] < delta
    ]
    materially_best.sort(
        key=lambda item: (
            item["cost_per_accepted_edge"],
            -item["entity_precision"],
            item["mean_latency_seconds"],
            0 if item["provider"] == "openai" else 1,
            item["provider"],
        )
    )
    return {"winner": materially_best[0]["provider"], "status": "SELECTED", "metrics": metrics}
