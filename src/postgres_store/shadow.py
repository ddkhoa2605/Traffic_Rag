from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

import numpy as np
import yaml

from src.chunking.builder import OUTPUT_NAMES
from src.legal_tree.loader import load_legal_document
from src.legal_tree.resolver import LegalTreeResolver
from src.parser.io import write_json
from src.retrieval_eval.dataset import load_dataset
from src.retrieval_eval.dense import DenseEncoder, DenseRetriever
from src.retrieval_eval.evaluator import _hybrid_passages
from src.retrieval_eval.hybrid import DualGranularityRetriever, HybridConfig, HybridSearchTrace

from .artifacts import FrozenRelease
from .config import PostgresSettings
from .dense import PostgresArticleRetriever, PostgresFineRetriever, PostgresVectorSession
from .repository import REPORT_DIR, validate_database


SCORE_TOLERANCE = 1e-6


def _compare_hits(label: str, expected, actual, mismatches: list[dict], query_id: str) -> None:
    expected_ids = [hit.passage_id for hit in expected]
    actual_ids = [hit.passage_id for hit in actual]
    if expected_ids != actual_ids:
        mismatches.append({
            "query_id": query_id, "stage": label,
            "expected_ids": expected_ids, "actual_ids": actual_ids,
        })
        return
    for left, right in zip(expected, actual):
        if abs(float(left.score) - float(right.score)) > SCORE_TOLERANCE:
            mismatches.append({
                "query_id": query_id, "stage": f"{label}_score",
                "passage_id": left.passage_id,
                "expected": float(left.score), "actual": float(right.score),
            })


def _compare_trace(
    query_id: str,
    expected: HybridSearchTrace,
    actual: HybridSearchTrace,
    mismatches: list[dict],
) -> None:
    _compare_hits("article_top10", expected.article_hits, actual.article_hits, mismatches, query_id)
    _compare_hits("fine_global_top50", expected.fine_global_hits, actual.fine_global_hits, mismatches, query_id)
    for article_id in sorted(expected.guided_hits):
        _compare_hits(
            f"guided_top3:{article_id}",
            expected.guided_hits[article_id], actual.guided_hits.get(article_id, ()),
            mismatches, query_id,
        )
    _compare_hits("b6_final_top10", expected.hits, actual.hits, mismatches, query_id)


def shadow_compare(
    root: str | Path,
    settings: PostgresSettings,
    release: FrozenRelease,
) -> dict:
    root_path = Path(root).resolve()
    database_validation = validate_database(settings, release)
    if not database_validation["valid"]:
        raise ValueError("Database release invalid: " + "; ".join(database_validation["errors"]))
    queries, _ = load_dataset(root_path)
    dev_queries = [query for query in queries if query.split == "dev"]
    if len(dev_queries) != 80:
        raise ValueError(f"Shadow comparison requires exactly 80 dev queries, found {len(dev_queries)}")

    retrieval_config = yaml.safe_load((root_path / "configs/retrieval.yaml").read_text(encoding="utf-8"))
    hybrid_config = retrieval_config["hybrid_b6"]
    artifact_root = release.release_manifest.get("artifact_root", "data/07_retrieval_ablation")
    article_passages, fine_passages, passage_manifest = _hybrid_passages(
        root_path, hybrid_config, artifact_root
    )
    artifact_path = Path(artifact_root)
    if not artifact_path.is_absolute():
        artifact_path = root_path / artifact_path
    resolvers = {
        document_id: LegalTreeResolver(load_legal_document(root_path, document_id).nodes)
        for document_id in passage_manifest["documents"]
    }
    dense = retrieval_config["dense"]
    encoder = DenseEncoder(
        dense["model"], dense["revision"], max_length=int(dense["max_length"]),
        preferred_device=release.strategy_lock["dense_runtime"]["device"],
    )
    if encoder.revision != release.release_manifest["embedding_revision"]:
        raise ValueError("Resolved encoder revision differs from frozen release")
    numpy_retriever = DualGranularityRetriever(
        DenseRetriever(
            article_passages, encoder,
            cache_dir=artifact_path / OUTPUT_NAMES["B1"] / "embeddings",
            passage_manifest_hash=passage_manifest["article_manifest_hash"], batch_size=4,
        ),
        DenseRetriever(
            fine_passages, encoder,
            cache_dir=artifact_path / OUTPUT_NAMES["B4e"] / "embeddings",
            passage_manifest_hash=passage_manifest["fine_manifest_hash"], batch_size=4,
        ),
        article_passages, fine_passages, resolvers, HybridConfig.from_dict(hybrid_config),
    )

    dsn = settings.runtime_dsn or settings.ingest_dsn
    if not dsn:
        raise ValueError("Missing runtime or ingest DSN")
    mismatches: list[dict] = []
    latencies_ms: list[float] = []
    with PostgresVectorSession(
        dsn=dsn,
        dataset_id=release.dataset_id,
        model_name=release.release_manifest["embedding_model"],
        model_revision=release.release_manifest["embedding_revision"],
        encoder=encoder,
        allow_building=True,
    ) as session:
        postgres_retriever = DualGranularityRetriever(
            PostgresArticleRetriever(session), PostgresFineRetriever(session),
            article_passages, fine_passages, resolvers, HybridConfig.from_dict(hybrid_config),
        )
        for query in dev_queries:
            expected = numpy_retriever.search_with_trace(query.query)
            started = time.perf_counter()
            actual = postgres_retriever.search_with_trace(query.query)
            latencies_ms.append((time.perf_counter() - started) * 1000)
            _compare_trace(query.query_id, expected, actual, mismatches)

    report_root = release.strategy_lock.get("report_root", "reports/chunk_ablation/runs")
    run_version = release.strategy_lock.get("run_version", "v001")
    baseline_summary_path = root_path / report_root / f"B6__DENSE__{run_version}__dev/summary.json"
    baseline_summary = json.loads(baseline_summary_path.read_text(encoding="utf-8"))
    ranking_match = not mismatches
    report = {
        "schema_version": release.release_manifest["schema_version"],
        "dataset_id": release.dataset_id,
        "query_count": len(dev_queries),
        "score_tolerance": SCORE_TOLERANCE,
        "article_top10_match": not any(item["stage"].startswith("article_top10") for item in mismatches),
        "fine_global_top50_match": not any(item["stage"].startswith("fine_global_top50") for item in mismatches),
        "guided_top3_match": not any(item["stage"].startswith("guided_top3") for item in mismatches),
        "b6_final_top10_match": not any(item["stage"].startswith("b6_final_top10") for item in mismatches),
        "projection_integrity_match": database_validation["valid"],
        "metrics_match_frozen": ranking_match and database_validation["valid"],
        "frozen_metrics": baseline_summary["metrics"],
        "mismatch_count": len(mismatches),
        "mismatches": mismatches[:100],
        "db_latency_ms": {
            "mean": statistics.fmean(latencies_ms),
            "p50": float(np.percentile(latencies_ms, 50)),
            "p95": float(np.percentile(latencies_ms, 95)),
            "max": max(latencies_ms),
        },
        "passed": ranking_match and database_validation["valid"],
    }
    release_report_dir = root_path / REPORT_DIR
    if release.release_manifest["schema_version"] != "0.1.0":
        release_report_dir = release_report_dir / "releases" / release.release_manifest["release_name"]
    output = release_report_dir / "shadow_comparison.json"
    write_json(output, report)
    markdown = [
        "# PostgreSQL B6 Shadow Comparison", "",
        f"Status: **{'PASS' if report['passed'] else 'FAIL'}**", "",
        f"Dataset: `{release.dataset_id}`", "",
        f"Queries: `{len(dev_queries)}`; mismatches: `{len(mismatches)}`.", "",
        "| Gate | Result |", "|---|---|",
        f"| B1 top-10 | {report['article_top10_match']} |",
        f"| B4e global top-50 | {report['fine_global_top50_match']} |",
        f"| Guided top-3/Article | {report['guided_top3_match']} |",
        f"| B6 final top-10 | {report['b6_final_top10_match']} |",
        f"| Frozen metrics equivalence | {report['metrics_match_frozen']} |", "",
        f"DB-only latency p50/p95: `{report['db_latency_ms']['p50']:.2f}` / `{report['db_latency_ms']['p95']:.2f}` ms.",
    ]
    release_report_dir.mkdir(parents=True, exist_ok=True)
    (release_report_dir / "shadow_comparison.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    if not report["passed"]:
        raise ValueError(f"PostgreSQL shadow comparison failed with {len(mismatches)} mismatches")
    return report
