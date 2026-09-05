from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path


def test_b6_official_dev_matrix_and_review_gate_artifacts():
    root = Path(__file__).resolve().parents[2]
    benchmark = json.loads((root / "data/08_eval/benchmark_manifest.json").read_text(encoding="utf-8"))
    for retriever in ("BM25", "DENSE"):
        run_dir = root / "reports/chunk_ablation/runs" / f"B6__{retriever}__v001__dev"
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        rows = [json.loads(line) for line in (run_dir / "per_query.jsonl").read_text(encoding="utf-8").splitlines()]
        manifest = json.loads((run_dir / "hybrid_manifest.json").read_text(encoding="utf-8"))
        assert summary["provisional"] is False
        assert summary["metrics_schema_version"] == "0.2.0"
        assert summary["dataset_hash"] == benchmark["dataset_digest"]
        assert summary["passage_manifest_hash"] == manifest["passage_digest"]
        assert summary["query_count"] == 80 and len(rows) == 800
        assert Counter(row["rank"] for row in rows) == Counter({rank: 80 for rank in range(1, 11)})
        assert all(row["evidence_components"] for row in rows)
        assert manifest["hybrid_config"]["rrf_k"] == 60
        assert manifest["hybrid_config"]["article_strategy"] == "B1"
        assert manifest["hybrid_config"]["fine_strategy"] == "B4e"

    with (root / "reports/chunk_ablation/b6/b6_manual_review.csv").open(encoding="utf-8-sig", newline="") as handle:
        review = list(csv.DictReader(handle))
    assert len(review) == 20
    assert Counter(row["bucket"] for row in review) == Counter({"failure": 10, "false_positive_top1": 5, "clear_win": 5})
    lock = json.loads((root / "reports/chunk_ablation/strategy_lock.json").read_text(encoding="utf-8"))
    assert lock["version"] == "0.2.0" and lock["winner"] == "B6" and lock["strategies"] == ["B6"]
    test_dirs = sorted(path.name for path in (root / "reports/chunk_ablation/runs").glob("*__test"))
    assert test_dirs == ["B6__BM25__v001__test", "B6__DENSE__v001__test"]
    for run_name in test_dirs:
        summary = json.loads((root / "reports/chunk_ablation/runs" / run_name / "summary.json").read_text(encoding="utf-8"))
        rows = (root / "reports/chunk_ablation/runs" / run_name / "per_query.jsonl").read_text(encoding="utf-8").splitlines()
        assert summary["split"] == "test" and summary["query_count"] == 40 and summary["provisional"] is False
        assert len(rows) == 400
