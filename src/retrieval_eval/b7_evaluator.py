from __future__ import annotations

import csv
import hashlib
import json
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path

from src.parser.io import read_jsonl, write_json, write_jsonl
from src.legal_tree.loader import load_legal_document
from src.registry.loader import load_registry
from src.retrieval_runtime.config import (
    ApplicationConfig,
    MultiEvidenceConfig,
    ReferenceRoutingConfig,
    load_application_config,
)
from src.retrieval_runtime.reference import parse_legal_reference
from src.retrieval_runtime.service import PostgresB6Application

from .b7_dataset import B7_DATA_DIR, load_b7_dataset, load_b7_research_config, validate_b7_dataset


B7_REPORT_DIR = Path("reports/b7")
POLICIES = ("B6", "B7a", "B7b", "B7c")
TOP_K = (1, 3, 5, 10)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _dataset_digest(root: Path) -> str:
    return json.loads((root / B7_DATA_DIR / "benchmark_manifest.json").read_text(encoding="utf-8"))["dataset_digest"]


def _run_version(root: Path) -> str:
    value = str(load_b7_research_config(root).get("run_version", "v001"))
    if not value.startswith("v") or not value[1:].isdigit():
        raise ValueError(f"Invalid B7 run_version: {value}")
    return value


def _run_path(root: Path, policy: str, split: str) -> Path:
    return root / B7_REPORT_DIR / "runs" / f"{policy}__POSTGRES_DENSE__{_run_version(root)}__{split}"


def _code_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for base in (root / "src/retrieval_runtime", root / "src/retrieval_eval"):
        for path in sorted(base.glob("*.py")):
            digest.update(path.relative_to(root).as_posix().encode("utf-8"))
            digest.update(path.read_bytes())
    return digest.hexdigest()


def _policy_config(root: Path, policy: str) -> ApplicationConfig:
    if policy not in POLICIES:
        raise ValueError(f"Unsupported ungated B7 policy: {policy}")
    base = load_application_config(root)
    return base.model_copy(update={
        "routing_policy": policy,
        "reference_routing": ReferenceRoutingConfig(enabled=policy in {"B7a", "B7c"}),
        "multi_evidence": MultiEvidenceConfig(enabled=policy in {"B7b", "B7c"}),
        "reranker": None,
    })


def _redundancy(results: list[dict], top_k: int) -> tuple[float, int, int]:
    components = [
        component
        for result in results[:top_k]
        for component in result.get("evidence_components", [])
    ]
    total = sum(component["token_count"] for component in components)
    unique: dict[str, int] = {}
    for component in components:
        unique.setdefault(component["token_sequence_hash"], component["token_count"])
    duplicate = total - sum(unique.values())
    return (duplicate / total if total else 0.0), duplicate, total


def _query_metrics(results: list[dict], gold: dict, top_k: int) -> dict:
    selected = results[:top_k]
    gold_ids = set(gold["gold_node_ids"])
    required = set(gold["required_node_ids"])
    exact_rank = next(
        (row["rank"] for row in results if row["primary_node_id"] in gold_ids), None,
    )
    member_union = {node_id for row in selected for node_id in row.get("member_node_ids", [])}
    included_union = {node_id for row in selected for node_id in row.get("included_node_ids", [])}
    ratio, duplicate, total = _redundancy(results, top_k)
    return {
        "recall": float(exact_rank is not None and exact_rank <= top_k),
        "mrr": 1.0 / exact_rank if exact_rank else 0.0,
        "bundle_exact_hit": float(gold_ids.issubset(member_union)),
        "evidence_coverage": len(required & included_union) / len(required) if required else 0.0,
        "evidence_tokens": sum(row["evidence_tokens"] for row in selected),
        "duplicate_token_ratio": ratio,
        "duplicate_token_count": duplicate,
        "total_component_tokens": total,
    }


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def _aggregate(records: list[dict]) -> dict:
    result: dict[str, float] = {}
    for top_k in TOP_K:
        values = [row["metrics"][str(top_k)] for row in records]
        for field in (
            "recall", "bundle_exact_hit", "evidence_coverage", "evidence_tokens",
            "duplicate_token_ratio", "duplicate_token_count", "total_component_tokens",
        ):
            result[f"{field}@{top_k}"] = _mean([value[field] for value in values])
    result["mrr"] = _mean([row["metrics"]["10"]["mrr"] for row in records])
    latencies = sorted(row["latency_ms"] for row in records)
    result["latency_mean_ms"] = _mean(latencies)
    result["latency_p50_ms"] = statistics.median(latencies) if latencies else 0.0
    if latencies:
        result["latency_p95_ms"] = latencies[min(len(latencies) - 1, int(0.95 * len(latencies)))]
    else:
        result["latency_p95_ms"] = 0.0
    return result


def evaluate_b7_policy(
    root: str | Path,
    policy: str,
    *,
    split: str = "dev",
    application_factory=PostgresB6Application,
    application_kwargs: dict | None = None,
    report_dir: str | Path = B7_REPORT_DIR,
    run_version: str | None = None,
    strategy_lock_path: str | Path = "reports/chunk_ablation/strategy_lock.json",
    dataset_loader=load_b7_dataset,
    dataset_validator=validate_b7_dataset,
    dataset_dir: str | Path = B7_DATA_DIR,
    research_config_loader=load_b7_research_config,
) -> dict:
    root_path = Path(root).resolve()
    if split not in {"dev", "test"}:
        raise ValueError("split must be dev or test")
    errors = dataset_validator(root_path, require_approved=True)
    if errors:
        raise ValueError("B7 benchmark is not approved: " + "; ".join(errors[:20]))
    report_dir = Path(report_dir)
    if not report_dir.is_absolute():
        report_dir = root_path / report_dir
    resolved_run_version = run_version or _run_version(root_path)
    if not resolved_run_version.startswith("v") or not resolved_run_version[1:].isdigit():
        raise ValueError("run_version must look like v003")
    lock_path = report_dir / "b7_policy_lock.json"
    dataset_dir = Path(dataset_dir)
    if not dataset_dir.is_absolute():
        dataset_dir = root_path / dataset_dir
    dataset_digest = json.loads(
        (dataset_dir / "benchmark_manifest.json").read_text(encoding="utf-8")
    )["dataset_digest"]
    run_dir = report_dir / "runs" / f"{policy}__POSTGRES_DENSE__{resolved_run_version}__{split}"
    if split == "test":
        if not lock_path.is_file():
            raise ValueError("Held-out test is unauthorized: b7_policy_lock.json is missing")
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        if lock["winner"] != policy:
            raise ValueError("Held-out may run only for the locked B7 winner")
        if lock.get("heldout_run_count") != 0:
            raise ValueError("Held-out has already been consumed")
        if lock["dataset_digest"] != dataset_digest:
            raise ValueError("B7 dataset changed after policy lock")
        requested_dataset_id = (application_kwargs or {}).get("dataset_id")
        if requested_dataset_id != lock.get("dataset_id"):
            raise ValueError("Held-out PostgreSQL dataset ID differs from the policy lock")
        resolved_strategy_lock = (
            Path(strategy_lock_path) if Path(strategy_lock_path).is_absolute()
            else root_path / strategy_lock_path
        )
        if lock["b6_strategy_lock_sha256"] != _sha256(resolved_strategy_lock):
            raise ValueError("Frozen B6 strategy lock changed after B7 lock")
        if lock["code_digest"] != _code_digest(root_path):
            raise ValueError("B7 source code changed after policy lock")
        if lock["b7_config_sha256"] != _sha256(root_path / "configs/b7.yaml"):
            raise ValueError("B7 research configuration changed after policy lock")
        manifest = json.loads((dataset_dir / "benchmark_manifest.json").read_text(encoding="utf-8"))
        if not manifest.get("heldout_authorized"):
            raise ValueError("Held-out authorization flag is false")
        if run_dir.exists():
            raise ValueError("Held-out artifact already exists; refusing a second test run")
    queries, gold = dataset_loader(root_path)
    selected = [item for item in queries if item.split == split]
    gold_by_id = {item.query_id: item.model_dump(mode="json") for item in gold}
    registry = load_registry(root_path)
    canonical_nodes = {
        node.id: node
        for document_id in registry.documents
        for node in load_legal_document(root_path, document_id).nodes
    }
    config = _policy_config(root_path, policy)
    records: list[dict] = []
    with application_factory(
        root_path,
        config=config,
        allow_unlocked_policy=True,
        **(application_kwargs or {}),
    ) as application:
        for query in selected:
            started = time.perf_counter()
            response = application.search(
                query.query, top_k=10, document_ids=query.request_document_ids,
            )
            latency_ms = (time.perf_counter() - started) * 1000.0
            response_value = response.model_dump(mode="json")
            results = response_value["results"]
            oracle_rank = None
            if policy == "B6" and split == "dev":
                candidate_pool = application.retriever.search_candidate_pool(query.query)
                gold_ids = set(gold_by_id[query.query_id]["gold_node_ids"])
                oracle_rank = next((
                    hit.rank for hit in candidate_pool
                    if application.passage_by_id[hit.passage_id].primary_node_id in gold_ids
                ), None)
            intent = parse_legal_reference(query.query, registry)
            metrics = {
                str(top_k): _query_metrics(results, gold_by_id[query.query_id], top_k)
                for top_k in TOP_K
            }
            records.append({
                "query_id": query.query_id,
                "query": query.query,
                "category": query.category,
                "split": query.split,
                "document_ids": query.document_ids,
                "request_document_ids": query.request_document_ids,
                "parser_detected_reference": bool(intent.article or intent.clause or intent.points),
                "route": response.route,
                "resolution_status": response.resolution_status,
                "latency_ms": latency_ms,
                "gold_node_ids": gold_by_id[query.query_id]["gold_node_ids"],
                "required_node_ids": gold_by_id[query.query_id]["required_node_ids"],
                "candidate_oracle_first_exact_rank": oracle_rank,
                "metrics": metrics,
                "results": results,
            })
    metrics = _aggregate(records)
    category_metrics = {
        category: _aggregate([row for row in records if row["category"] == category])
        for category in sorted({row["category"] for row in records})
    }
    expected_detection = [
        row for row in records
        if row["category"] in {"exact_reference", "multi_evidence", "clause_intro_point"}
    ]
    expected_reference = [
        row for row in records if row["category"] in {"exact_reference", "multi_evidence"}
    ]
    detected = [row for row in records if row["parser_detected_reference"]]
    true_detected = [row for row in detected if row in expected_detection]
    resolved = [row for row in expected_reference if row["resolution_status"] == "RESOLVED"]
    direct_rows = [row for row in expected_detection if row["route"].startswith("DIRECT_")]
    wrong_law = sum(
        bool(row["results"] and row["results"][0]["document_id"] not in row["document_ids"])
        for row in direct_rows
    )
    wrong_article = 0
    wrong_clause = 0
    direct_count = 0
    for row in expected_detection:
        if not row["route"].startswith("DIRECT_") or not row["results"]:
            continue
        direct_count += 1
        expected_node = canonical_nodes[row["gold_node_ids"][0]]
        actual = row["results"][0]["hierarchy"]
        wrong_article += actual.get("article") != expected_node.hierarchy.get("article")
        if expected_node.hierarchy.get("clause") is not None:
            wrong_clause += actual.get("clause") != expected_node.hierarchy.get("clause")
    routing_metrics = {
        "reference_detection_precision": len(true_detected) / len(detected) if detected else 1.0,
        "reference_detection_recall": len(true_detected) / len(expected_detection) if expected_detection else 1.0,
        "resolution_accuracy": len(resolved) / len(expected_reference) if expected_reference else 1.0,
        "wrong_law_rate": wrong_law / len(direct_rows) if direct_rows else 0.0,
        "wrong_article_rate": wrong_article / direct_count if direct_count else 0.0,
        "wrong_clause_rate": wrong_clause / direct_count if direct_count else 0.0,
        "route_counts": dict(Counter(row["route"] for row in records)),
    }
    if policy == "B6" and split == "dev":
        oracle_ranks = [row["candidate_oracle_first_exact_rank"] for row in records]
        gate_config = research_config_loader(root_path)["gates"]
        oracle_recall30 = sum(rank is not None and rank <= 30 for rank in oracle_ranks) / len(records)
        ordering_failures = sum(rank is not None and 6 <= rank <= 30 for rank in oracle_ranks)
        reranker_gate = {
            "oracle_recall@30": oracle_recall30,
            "ordering_failure_count_rank6_30": ordering_failures,
            "eligible": (
                oracle_recall30 >= gate_config["reranker_oracle_recall30_min"]
                and ordering_failures >= gate_config["reranker_min_ordering_failures"]
            ),
        }
    else:
        reranker_gate = None
    ranking_rows = [
        (row["query_id"], result["rank"], result["passage_id"], result["primary_node_id"])
        for row in records for result in row["results"]
    ]
    no_reference_rows = [
        item for row in records if not row["parser_detected_reference"]
        for item in [(row["query_id"], [(r["rank"], r["passage_id"]) for r in row["results"]])]
    ]
    summary = {
        "run_id": run_dir.name,
        "policy": policy,
        "retriever": "postgres_dense",
        "split": split,
        "query_count": len(records),
        "metrics_schema_version": "0.3.0",
        "run_version": resolved_run_version,
        "dataset_digest": dataset_digest,
        "b6_strategy_lock_sha256": _sha256(
            Path(strategy_lock_path) if Path(strategy_lock_path).is_absolute()
            else root_path / strategy_lock_path
        ),
        "metrics": metrics,
        "category_metrics": category_metrics,
        "routing_metrics": routing_metrics,
        "reranker_gate": reranker_gate,
        "ranking_digest": hashlib.sha256(json.dumps(ranking_rows, ensure_ascii=False).encode("utf-8")).hexdigest(),
        "no_reference_ranking_digest": hashlib.sha256(json.dumps(no_reference_rows, ensure_ascii=False).encode("utf-8")).hexdigest(),
        "routing_policy_config": config.model_dump(mode="json"),
        "b7_config_sha256": _sha256(root_path / "configs/b7.yaml"),
    }
    run_dir.mkdir(parents=True, exist_ok=False)
    write_jsonl(run_dir / "per_query.jsonl", records)
    write_json(run_dir / "summary.json", summary)
    if split == "test":
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        lock["heldout_run_count"] = 1
        lock["test_summary_sha256"] = _sha256(run_dir / "summary.json")
        write_json(lock_path, lock)
        manifest_path = dataset_dir / "benchmark_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["heldout_consumed"] = True
        manifest["test_output_count"] = 1
        manifest["test_run_id"] = run_dir.name
        manifest["test_summary_sha256"] = lock["test_summary_sha256"]
        write_json(manifest_path, manifest)
    return summary


def export_b7_manual_review(
    root: str | Path,
    *,
    report_dir: str | Path = B7_REPORT_DIR,
    run_version: str | None = None,
    output_filename: str = "b7_manual_review.csv",
) -> dict:
    root_path = Path(root).resolve()
    resolved_report_dir = Path(report_dir)
    if not resolved_report_dir.is_absolute():
        resolved_report_dir = root_path / resolved_report_dir
    resolved_run_version = run_version or _run_version(root_path)
    rows: list[dict] = []
    for policy in ("B7a", "B7b", "B7c"):
        path = (
            resolved_report_dir / "runs"
            / f"{policy}__POSTGRES_DENSE__{resolved_run_version}__dev"
            / "per_query.jsonl"
        )
        if not path.is_file():
            raise ValueError(f"Missing dev artifact for {policy}")
        for record in read_jsonl(path):
            if not record["route"].startswith("DIRECT_"):
                continue
            rows.append({
                "review_id": f"{policy}__{record['query_id']}",
                "policy": policy,
                "query_id": record["query_id"],
                "category": record["category"],
                "query": record["query"],
                "route": record["route"],
                "gold_node_ids": "|".join(record["gold_node_ids"]),
                "required_node_ids": "|".join(record["required_node_ids"]),
                "top_result": json.dumps(record["results"][:1], ensure_ascii=False, separators=(",", ":")),
                "recall@1": record["metrics"]["1"]["recall"],
                "bundle_exact_hit@1": record["metrics"]["1"]["bundle_exact_hit"],
                "evidence_coverage@1": record["metrics"]["1"]["evidence_coverage"],
                "review_status": "draft",
                "verdict": "",
                "reviewer": "",
                "review_notes": "",
            })
    output = resolved_report_dir / output_filename
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_jsonl(output.with_suffix(".jsonl"), rows)
    return {"review_count": len(rows), "path": str(output), "approved_count": 0}


def assess_b7d_gate(root: str | Path) -> dict:
    root_path = Path(root).resolve()
    summary_path = _run_path(root_path, "B6", "dev") / "summary.json"
    if not summary_path.is_file():
        raise ValueError("Run the approved B6 dev evaluation before assessing B7d")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    gate = summary.get("reranker_gate")
    if gate is None:
        raise ValueError("B6 summary does not contain reranker eligibility metrics")
    result = {
        **gate,
        "status": "ELIGIBLE_FOR_IMPLEMENTATION" if gate["eligible"] else "GATED_OFF",
        "model": "BAAI/bge-reranker-v2-m3" if gate["eligible"] else None,
        "revision": None,
        "note": (
            "Resolve and pin a model revision before implementing/running B7d."
            if gate["eligible"] else "B7d must not be implemented or run for this benchmark release."
        ),
    }
    write_json(root_path / B7_REPORT_DIR / "b7d_gate.json", result)
    return result


def apply_b7_manual_review(
    root: str | Path,
    review_csv: str | Path,
    *,
    report_dir: str | Path = B7_REPORT_DIR,
    review_filename: str = "b7_manual_review.csv",
) -> dict:
    root_path = Path(root).resolve()
    resolved_report_dir = Path(report_dir)
    if not resolved_report_dir.is_absolute():
        resolved_report_dir = root_path / resolved_report_dir
    canonical_path = resolved_report_dir / review_filename
    with canonical_path.open(encoding="utf-8-sig", newline="") as handle:
        canonical = list(csv.DictReader(handle))
    with Path(review_csv).resolve().open(encoding="utf-8-sig", newline="") as handle:
        reviewed = list(csv.DictReader(handle))
    if [row["review_id"] for row in canonical] != [row["review_id"] for row in reviewed]:
        raise ValueError("Reviewed rows/order differ from the canonical B7 review queue")
    mutable = {"review_status", "verdict", "reviewer", "review_notes"}
    errors: list[str] = []
    for index, (old, new) in enumerate(zip(canonical, reviewed), start=2):
        for field in old:
            if field not in mutable and old[field] != new.get(field, ""):
                errors.append(f"row {index}: immutable field changed: {field}")
        if new.get("review_status", "").casefold() not in {"approved", "rejected", "draft"}:
            errors.append(f"row {index}: invalid review_status")
        if new.get("verdict", "").casefold() not in {"correct", "strategy_failure", "benchmark_issue", "ambiguous"}:
            errors.append(f"row {index}: invalid verdict")
        if not new.get("reviewer", "").strip():
            errors.append(f"row {index}: reviewer is required")
    if errors:
        raise ValueError("Invalid B7 manual review: " + "; ".join(errors[:20]))
    canonical_path.write_text(Path(review_csv).resolve().read_text(encoding="utf-8-sig"), encoding="utf-8-sig")
    write_jsonl(canonical_path.with_suffix(".jsonl"), reviewed)
    approved = sum(row["review_status"].casefold() == "approved" for row in reviewed)
    issues = sum(row["verdict"].casefold() == "benchmark_issue" for row in reviewed)
    return {"review_count": len(reviewed), "approved_count": approved, "benchmark_issue_count": issues}


def select_b7_candidate(
    root: str | Path,
    *,
    report_dir: str | Path = B7_REPORT_DIR,
    run_version: str | None = None,
    review_filename: str = "b7_manual_review.csv",
    decision_filename: str = "b7_decision.json",
) -> dict:
    root_path = Path(root).resolve()
    resolved_report_dir = Path(report_dir)
    if not resolved_report_dir.is_absolute():
        resolved_report_dir = root_path / resolved_report_dir
    resolved_run_version = run_version or _run_version(root_path)
    summaries = {}
    for policy in POLICIES:
        path = (
            resolved_report_dir / "runs"
            / f"{policy}__POSTGRES_DENSE__{resolved_run_version}__dev"
            / "summary.json"
        )
        if not path.is_file():
            raise ValueError(f"Missing dev summary for {policy}")
        summaries[policy] = json.loads(path.read_text(encoding="utf-8"))
    review_path = resolved_report_dir / review_filename
    if not review_path.is_file():
        raise ValueError("B7 manual review has not been exported")
    with review_path.open(encoding="utf-8-sig", newline="") as handle:
        reviews = list(csv.DictReader(handle))
    if not reviews or any(row["review_status"].casefold() != "approved" for row in reviews):
        raise ValueError("B7 selection requires every manual review row approved")
    if any(row["verdict"].casefold() == "benchmark_issue" for row in reviews):
        raise ValueError("B7 selection blocked by benchmark_issue")
    baseline = summaries["B6"]
    research_config = load_b7_research_config(root_path)
    configured_gates = research_config["gates"]
    gates: dict[str, list[str]] = defaultdict(list)
    for policy, summary in summaries.items():
        if policy == "B6":
            continue
        metrics = summary["metrics"]
        base_metrics = baseline["metrics"]
        if summary["routing_metrics"]["wrong_law_rate"] != 0:
            gates[policy].append("wrong_law_rate must be zero")
        if summary["routing_metrics"]["wrong_article_rate"] != 0:
            gates[policy].append("wrong_article_rate must be zero")
        if summary["routing_metrics"]["wrong_clause_rate"] != 0:
            gates[policy].append("wrong_clause_rate must be zero")
        if summary["routing_metrics"]["resolution_accuracy"] != 1.0:
            gates[policy].append("supported-reference resolution accuracy must be 100%")
        if summary["no_reference_ranking_digest"] != baseline["no_reference_ranking_digest"]:
            gates[policy].append("NO_REFERENCE ranking digest changed")
        for field in ("recall@5", "mrr", "evidence_coverage@5"):
            if metrics[field] < base_metrics[field]:
                gates[policy].append(f"global {field} regressed")
        if metrics["evidence_tokens@5"] > base_metrics["evidence_tokens@5"] * configured_gates["evidence_tokens_max_multiplier"]:
            gates[policy].append("evidence token cost increased by more than 25%")
        if metrics["duplicate_token_ratio@5"] > base_metrics["duplicate_token_ratio@5"] + configured_gates["duplicate_token_ratio_max_increase"]:
            gates[policy].append("duplicate-token ratio increased by more than 0.05")
        if metrics["latency_p95_ms"] > base_metrics["latency_p95_ms"] + configured_gates["latency_p95_max_increase_ms"]:
            gates[policy].append("latency p95 increased by more than 10 ms")
        if policy in {"B7a", "B7c"}:
            gain = (
                summary["category_metrics"]["exact_reference"]["recall@1"]
                - baseline["category_metrics"]["exact_reference"]["recall@1"]
            )
            if gain < configured_gates["exact_reference_recall1_min_gain"]:
                gates[policy].append("Exact-reference Recall@1 gain is below 0.20")
        if policy in {"B7b", "B7c"}:
            gain = (
                summary["category_metrics"]["multi_evidence"]["evidence_coverage@5"]
                - baseline["category_metrics"]["multi_evidence"]["evidence_coverage@5"]
            )
            if gain < configured_gates["multi_evidence_coverage5_min_gain"]:
                gates[policy].append("Multi-evidence Coverage@5 gain is below 0.20")
    eligible = [policy for policy in ("B7a", "B7b", "B7c") if not gates[policy]]
    if not eligible:
        winner = "B6"
    else:
        def selection_tuple(policy: str):
            summary = summaries[policy]
            metrics = summary["metrics"]
            return (
                summary["category_metrics"]["exact_reference"]["recall@1"],
                summary["category_metrics"]["multi_evidence"]["evidence_coverage@5"],
                metrics["recall@5"], metrics["mrr"], metrics["evidence_coverage@5"],
                -metrics["evidence_tokens@5"], -metrics["duplicate_token_ratio@5"],
                -metrics["latency_p95_ms"], policy,
            )
        winner = max(eligible, key=selection_tuple)
    decision = {
        "winner": winner,
        "eligible": eligible,
        "gate_failures": dict(gates),
        "selection_rule": [
            "exact_reference_recall@1", "multi_evidence_coverage@5", "global_recall@5",
            "mrr", "global_evidence_coverage@5", "lower_evidence_tokens@5",
            "lower_duplicate_token_ratio@5", "lower_latency_p95_ms", "policy_id",
        ],
        "run_version": resolved_run_version,
        "dataset_digest": baseline["dataset_digest"],
    }
    write_json(resolved_report_dir / decision_filename, decision)
    return decision


def lock_b7_policy(
    root: str | Path,
    *,
    report_dir: str | Path = B7_REPORT_DIR,
    run_version: str | None = None,
    decision_filename: str = "b7_decision.json",
    lock_filename: str = "b7_policy_lock.json",
    dataset_dir: str | Path = B7_DATA_DIR,
    dataset_id: str | None = None,
    strategy_lock_path: str | Path = "reports/chunk_ablation/strategy_lock.json",
) -> dict:
    root_path = Path(root).resolve()
    resolved_report_dir = Path(report_dir)
    if not resolved_report_dir.is_absolute():
        resolved_report_dir = root_path / resolved_report_dir
    resolved_dataset_dir = Path(dataset_dir)
    if not resolved_dataset_dir.is_absolute():
        resolved_dataset_dir = root_path / resolved_dataset_dir
    resolved_strategy_lock = Path(strategy_lock_path)
    if not resolved_strategy_lock.is_absolute():
        resolved_strategy_lock = root_path / resolved_strategy_lock
    resolved_run_version = run_version or _run_version(root_path)
    decision_path = resolved_report_dir / decision_filename
    if not decision_path.is_file():
        raise ValueError("Run B7 selection before locking")
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    winner = decision["winner"]
    summary_path = (
        resolved_report_dir / "runs"
        / f"{winner}__POSTGRES_DENSE__{resolved_run_version}__dev"
        / "summary.json"
    )
    if dataset_id is None:
        dataset_id = json.loads(
            (root_path / "reports/chunk_ablation/postgres/postgres_migration_manifest.json").read_text(encoding="utf-8")
        )["dataset_id"]
    dataset_digest = json.loads(
        (resolved_dataset_dir / "benchmark_manifest.json").read_text(encoding="utf-8")
    )["dataset_digest"]
    lock = {
        "schema_version": "0.3.0",
        "run_version": resolved_run_version,
        "winner": winner,
        "dataset_id": dataset_id,
        "dataset_digest": dataset_digest,
        "b6_strategy_lock_sha256": _sha256(resolved_strategy_lock),
        "dev_summary_sha256": _sha256(summary_path),
        "base_application_config_sha256": _sha256(root_path / "configs/application.yaml"),
        "winner_runtime_config": _policy_config(root_path, winner).model_dump(mode="json"),
        "code_digest": _code_digest(root_path),
        "b7_config_sha256": _sha256(root_path / "configs/b7.yaml"),
        "heldout_run_count": 0,
    }
    lock_path = resolved_report_dir / lock_filename
    if lock_path.exists():
        raise ValueError("Policy lock already exists; refusing to overwrite it")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("dataset_digest") != dataset_digest:
        raise ValueError("Winner dev summary dataset digest does not match the benchmark")
    if summary.get("b6_strategy_lock_sha256") != _sha256(resolved_strategy_lock):
        raise ValueError("Winner dev summary strategy lock differs from the requested lock")
    write_json(lock_path, lock)
    manifest_path = resolved_dataset_dir / "benchmark_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["heldout_authorized"] = True
    manifest["heldout_consumed"] = False
    manifest["test_output_count"] = 0
    manifest["locked_winner"] = winner
    manifest["policy_lock_sha256"] = _sha256(lock_path)
    write_json(manifest_path, manifest)
    return lock
