from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from src.retrieval_eval.round2 import (
    METRICS_SCHEMA_VERSION,
    ROUND2_VARIANTS,
    _load_run_artifacts,
    build_round2_review_queue,
    validate_round2_review,
)


ROOT = Path(__file__).resolve().parents[2]


def test_all_round2_runs_migrated_without_ranking_changes():
    manifest = json.loads(
        (ROOT / "reports/chunk_ablation/round2/round2_migration_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["run_count"] == 10
    assert manifest["ranking_or_embedding_recomputed"] is False
    assert manifest["saved_rank_depth"] == 10
    for strategy in ROUND2_VARIANTS:
        for retriever in ("BM25", "DENSE"):
            run = ROOT / "reports/chunk_ablation/runs" / f"{strategy}__{retriever}__v001__dev"
            current = json.loads((run / "summary.json").read_text(encoding="utf-8"))
            legacy = json.loads((run / "summary.metrics-v0.1.json").read_text(encoding="utf-8"))
            assert current["metrics_schema_version"] == METRICS_SCHEMA_VERSION
            assert current["metrics"]["duplicate_token_ratio_legacy@5"] == legacy["metrics"]["duplicate_token_ratio@5"]
            assert current["ranking_digest"] == manifest["ranking_digests"][current["run_id"]]


def test_round2_review_sampling_is_exact_and_deterministic():
    artifacts = _load_run_artifacts(ROOT, "dev")
    first = build_round2_review_queue(artifacts, ROOT)
    second = build_round2_review_queue(artifacts, ROOT)
    assert first == second
    assert len(first) == 100
    assert len({(row["strategy"], row["query_id"]) for row in first}) == 100
    assert Counter((row["strategy"], row["bucket"]) for row in first) == Counter({
        **{(strategy, "failure"): 10 for strategy in ROUND2_VARIANTS},
        **{(strategy, "false_positive_top1"): 5 for strategy in ROUND2_VARIANTS},
        **{(strategy, "clear_win"): 5 for strategy in ROUND2_VARIANTS},
    })
    assert validate_round2_review(
        ROOT / "reports/chunk_ablation/round2/round2_manual_review.csv"
    ) == []
