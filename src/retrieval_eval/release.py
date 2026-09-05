from __future__ import annotations

import hashlib
import json
import csv
from datetime import datetime, timezone
from pathlib import Path

import yaml

from src.chunking.builder import OUTPUT_NAMES, _code_digest
from src.parser.io import write_json
from src.parser.io import read_jsonl


SELECTION_KEYS = (
    ("recall@5", True),
    ("mrr", True),
    ("evidence_coverage@5", True),
    ("avg_evidence_tokens@5", False),
    ("duplicate_token_ratio@5", False),
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def create_release_candidate_lock(
    root: str | Path,
    *,
    artifact_root: str | Path,
    report_root: str | Path,
    run_version: str,
    output_path: str | Path,
) -> dict:
    root_path = Path(root).resolve()
    artifacts = Path(artifact_root)
    reports = Path(report_root)
    output = Path(output_path)
    if not artifacts.is_absolute():
        artifacts = root_path / artifacts
    if not reports.is_absolute():
        reports = root_path / reports
    if not output.is_absolute():
        output = root_path / output
    if output.exists():
        raise FileExistsError(f"Release candidate lock already exists: {output}")

    candidates = ("B4a", "B4b", "B4c", "B4d", "B4e", "B6")
    summaries: dict[str, dict[str, dict]] = {}
    means: dict[str, dict[str, float]] = {}
    summary_hashes: dict[str, dict[str, str]] = {}
    for strategy in candidates:
        summaries[strategy] = {}
        summary_hashes[strategy] = {}
        for retriever in ("BM25", "DENSE"):
            path = reports / f"{strategy}__{retriever}__{run_version}__dev" / "summary.json"
            value = json.loads(path.read_text(encoding="utf-8"))
            if value.get("provisional", True):
                raise ValueError(f"Release run is provisional: {value.get('run_id')}")
            if value.get("metrics_schema_version") != "0.2.0":
                raise ValueError(f"Release run has stale metric schema: {value.get('run_id')}")
            summaries[strategy][retriever.casefold()] = value
            summary_hashes[strategy][retriever.casefold()] = _sha256(path)
        means[strategy] = {
            key: sum(summaries[strategy][backend]["metrics"][key] for backend in ("bm25", "dense")) / 2
            for key, _ in SELECTION_KEYS
        }

    def selection_tuple(strategy: str) -> tuple:
        values = [means[strategy][key] if maximize else -means[strategy][key] for key, maximize in SELECTION_KEYS]
        return (*values, tuple(-ord(char) for char in strategy))

    winner = max(candidates, key=selection_tuple)
    if winner != "B6":
        raise ValueError(f"PostgreSQL B6 release candidate cannot be created; strict winner is {winner}")
    retrieval_config = yaml.safe_load((root_path / "configs/retrieval.yaml").read_text(encoding="utf-8"))
    dense = summaries["B6"]["dense"]["retriever_config"]
    benchmark = json.loads((root_path / "data/08_eval/benchmark_manifest.json").read_text(encoding="utf-8"))
    payload = {
        "version": "0.3.0-candidate",
        "strategies": ["B6"],
        "winner": winner,
        "selection_scope": list(candidates),
        "selection_metrics": means,
        "selection_rule": [key if maximize else f"lower_{key}" for key, maximize in SELECTION_KEYS] + ["strategy_id"],
        "dataset_digest": benchmark["dataset_digest"],
        "artifact_root": artifacts.relative_to(root_path).as_posix(),
        "report_root": reports.relative_to(root_path).as_posix(),
        "run_version": run_version,
        "chunking_config_sha256": _sha256(root_path / "configs/chunking.yaml"),
        "retrieval_config_sha256": _sha256(root_path / "configs/retrieval.yaml"),
        "hybrid_b6_config": retrieval_config["hybrid_b6"],
        "code_digest": _code_digest(root_path),
        "dev_summary_sha256": summary_hashes["B6"],
        "all_dev_summary_sha256": summary_hashes,
        "passage_manifest_sha256": {
            source: _sha256(artifacts / OUTPUT_NAMES[source] / "manifest.json")
            for source in ("B1", "B4e")
        },
        "dense_runtime": {
            "model": dense["model"],
            "revision": dense["resolved_revision"],
            "device": dense["resolved_device"],
            "dtype": dense["resolved_dtype"],
        },
        "production_frozen": False,
        "manual_review_status": "required_after_projection_change",
        "heldout_status": "v0.1-and-v0.2-are-regression-only",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(output, payload)
    return payload


def export_release_v2_review(
    root: str | Path,
    *,
    old_run_dir: str | Path = "reports/b7/runs/B7c__POSTGRES_DENSE__v002__dev",
    new_run_dir: str | Path = "reports/b7/release_v2/runs/B7c__POSTGRES_DENSE__v003__dev",
    output_path: str | Path = "reports/b7/release_v2/release_v2_manual_review.csv",
) -> dict:
    root_path = Path(root).resolve()
    old_path = root_path / old_run_dir
    new_path = root_path / new_run_dir
    output = root_path / output_path
    old = {row["query_id"]: row for row in read_jsonl(old_path / "per_query.jsonl")}
    new = {row["query_id"]: row for row in read_jsonl(new_path / "per_query.jsonl")}

    def exact5(row: dict) -> bool:
        return bool(row["metrics"]["5"]["recall"])

    candidates = []
    for query_id, row in new.items():
        if row["route"] != "B6_FALLBACK":
            continue
        before = old[query_id]
        if not exact5(before) and not exact5(row):
            bucket = "persistent_failure"
        elif not exact5(before) and exact5(row):
            bucket = "recovered_win"
        elif exact5(before) and not exact5(row):
            bucket = "regression"
        else:
            bucket = "changed_fallback"
        old_ids = [item["passage_id"] for item in before["results"][:5]]
        new_ids = [item["passage_id"] for item in row["results"][:5]]
        rank_delta = sum(left != right for left, right in zip(old_ids, new_ids))
        candidates.append((bucket, row["category"], -rank_delta, query_id, before, row))

    priority = {"regression": 0, "persistent_failure": 1, "recovered_win": 2, "changed_fallback": 3}
    candidates.sort(key=lambda item: (priority[item[0]], item[1], item[2], item[3]))
    selected = []
    by_category: dict[str, list[tuple]] = {}
    for item in candidates:
        by_category.setdefault(item[1], []).append(item)
    while len(selected) < min(20, len(candidates)):
        progressed = False
        for category in sorted(by_category):
            if by_category[category] and len(selected) < 20:
                selected.append(by_category[category].pop(0))
                progressed = True
        if not progressed:
            break

    fields = [
        "query_id", "query", "category", "bucket", "gold_node_ids", "required_node_ids",
        "old_top5", "new_top5", "new_evidence_preview", "review_status", "verdict",
        "taxonomy_codes", "reviewer", "notes",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for bucket, _, _, _, before, row in selected:
            writer.writerow({
                "query_id": row["query_id"],
                "query": row["query"],
                "category": row["category"],
                "bucket": bucket,
                "gold_node_ids": "|".join(row["gold_node_ids"]),
                "required_node_ids": "|".join(row["required_node_ids"]),
                "old_top5": json.dumps(before["results"][:5], ensure_ascii=False, separators=(",", ":")),
                "new_top5": json.dumps(row["results"][:5], ensure_ascii=False, separators=(",", ":")),
                "new_evidence_preview": " || ".join(
                    item.get("evidence_text", "")[:500].replace("\n", " ") for item in row["results"][:3]
                ),
                "review_status": "draft",
                "verdict": "",
                "taxonomy_codes": "",
                "reviewer": "",
                "notes": "",
            })
    return {
        "output": str(output),
        "record_count": len(selected),
        "bucket_counts": {
            bucket: sum(item[0] == bucket for item in selected)
            for bucket in sorted({item[0] for item in selected})
        },
        "direct_route_carry_forward_count": sum(
            row["route"].startswith("DIRECT_") and old[query_id]["results"] == row["results"]
            for query_id, row in new.items()
        ),
    }


def validate_release_v2_review(
    root: str | Path,
    review_path: str | Path,
    *,
    output_path: str | Path = "reports/b7/release_v2/release_v2_review_validation.json",
) -> dict:
    root_path = Path(root).resolve()
    review = Path(review_path)
    if not review.is_absolute():
        review = root_path / review
    original = root_path / "reports/b7/release_v2/release_v2_manual_review.csv"
    output = root_path / output_path

    with original.open(encoding="utf-8-sig", newline="") as handle:
        source_rows = list(csv.DictReader(handle))
    with review.open(encoding="utf-8-sig", newline="") as handle:
        reviewed_rows = list(csv.DictReader(handle))
    errors: list[str] = []
    if len(source_rows) != 20 or len(reviewed_rows) != 20:
        errors.append(f"expected 20 source/review rows, found {len(source_rows)}/{len(reviewed_rows)}")
    protected = (
        "query_id", "query", "category", "bucket", "gold_node_ids", "required_node_ids",
        "old_top5", "new_top5", "new_evidence_preview",
    )
    if [row.get("query_id") for row in source_rows] != [row.get("query_id") for row in reviewed_rows]:
        errors.append("query order or membership changed")
    for index, (source, reviewed) in enumerate(zip(source_rows, reviewed_rows), start=2):
        changed = [field for field in protected if source.get(field) != reviewed.get(field)]
        if changed:
            errors.append(f"row {index}: protected fields changed: {changed}")

    run_rows = {
        row["query_id"]: row
        for row in read_jsonl(
            root_path / "reports/b7/release_v2/runs/B7c__POSTGRES_DENSE__v003__dev/per_query.jsonl"
        )
    }
    allowed_verdicts = {"expected_win", "strategy_failure", "benchmark_issue", "ambiguous"}
    allowed_taxonomy = {f"F{index}" for index in range(1, 11)}
    verdict_counts: dict[str, int] = {}
    taxonomy_counts: dict[str, int] = {}
    for index, row in enumerate(reviewed_rows, start=2):
        query_id = row.get("query_id", "")
        status = row.get("review_status", "").strip()
        verdict = row.get("verdict", "").strip()
        if status != "approved":
            errors.append(f"row {index}: review_status must be approved")
        if verdict not in allowed_verdicts:
            errors.append(f"row {index}: invalid verdict {verdict!r}")
        verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1
        if not row.get("reviewer", "").strip():
            errors.append(f"row {index}: reviewer is empty")
        if not row.get("notes", "").strip():
            errors.append(f"row {index}: notes are empty")
        codes = {value for value in row.get("taxonomy_codes", "").split("|") if value}
        invalid_codes = codes - allowed_taxonomy
        if invalid_codes:
            errors.append(f"row {index}: invalid taxonomy codes {sorted(invalid_codes)}")
        for code in codes:
            taxonomy_counts[code] = taxonomy_counts.get(code, 0) + 1
        if verdict == "strategy_failure" and not codes:
            errors.append(f"row {index}: strategy_failure requires taxonomy_codes")
        try:
            old_top5 = json.loads(row["old_top5"])
            new_top5 = json.loads(row["new_top5"])
        except (KeyError, json.JSONDecodeError) as exc:
            errors.append(f"row {index}: invalid embedded top-5 JSON: {exc}")
            continue
        if len(old_top5) != 5 or len(new_top5) != 5:
            errors.append(f"row {index}: old/new top-5 must each contain five results")
        if [item.get("rank") for item in new_top5] != [1, 2, 3, 4, 5]:
            errors.append(f"row {index}: new_top5 ranks are not 1..5")
        artifact = run_rows.get(query_id)
        if artifact is None:
            errors.append(f"row {index}: query missing from v003 artifact")
            continue
        if new_top5 != artifact["results"][:5]:
            errors.append(f"row {index}: new_top5 differs from v003 artifact")
        gold = {value for value in row["gold_node_ids"].split("|") if value}
        retrieved = {
            node_id
            for item in new_top5
            for node_id in [item.get("primary_node_id"), *item.get("member_node_ids", [])]
            if node_id
        }
        exact_or_member_hit = bool(gold & retrieved)
        if verdict == "expected_win" and not exact_or_member_hit:
            errors.append(f"row {index}: expected_win has no gold/member hit in top-5")
        if verdict == "strategy_failure" and exact_or_member_hit:
            errors.append(f"row {index}: strategy_failure has a gold/member hit in top-5")

    benchmark_issue_count = verdict_counts.get("benchmark_issue", 0)
    report = {
        "schema_version": "1.0.0",
        "review_file": review.relative_to(root_path).as_posix() if review.is_relative_to(root_path) else str(review),
        "review_sha256": _sha256(review),
        "source_queue_sha256": _sha256(original),
        "record_count": len(reviewed_rows),
        "approved_count": sum(row.get("review_status", "").strip() == "approved" for row in reviewed_rows),
        "verdict_counts": verdict_counts,
        "taxonomy_counts": taxonomy_counts,
        "benchmark_issue_count": benchmark_issue_count,
        "errors": errors,
        "valid": not errors,
        "manual_review_gate_passed": not errors and benchmark_issue_count == 0,
        "validated_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(output, report)
    return report
