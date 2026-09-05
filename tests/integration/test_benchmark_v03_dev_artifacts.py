from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports/b7_v03"


def test_v03_official_dev_matrix_and_single_winner_test_use_frozen_dataset():
    manifest = json.loads(
        (ROOT / "data/11_eval_v03/benchmark_manifest.json").read_text(encoding="utf-8")
    )
    for policy in ("B6", "B7a", "B7b", "B7c"):
        run_dir = REPORT / "runs" / f"{policy}__POSTGRES_DENSE__v001__dev"
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        records = [
            json.loads(line)
            for line in (run_dir / "per_query.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        assert summary["dataset_digest"] == manifest["dataset_digest"]
        assert summary["split"] == "dev"
        assert summary["query_count"] == len(records) == 80
        assert summary["retriever"] == "postgres_dense"
        assert {record["query_id"] for record in records} == {
            f"V3Q{index:04d}"
            for index in list(range(1, 9)) + list(range(17, 25))
            + list(range(33, 41)) + list(range(49, 55))
            + list(range(61, 67)) + list(range(73, 77))
            + list(range(81, 89)) + list(range(97, 105))
            + list(range(113, 121)) + list(range(129, 135))
            + list(range(141, 147)) + list(range(153, 157))
        }
    test_dirs = list((REPORT / "runs").glob("*__test"))
    assert [path.name for path in test_dirs] == ["B7c__POSTGRES_DENSE__v001__test"]
    test_summary = json.loads((test_dirs[0] / "summary.json").read_text(encoding="utf-8"))
    assert test_summary["policy"] == "B7c"
    assert test_summary["split"] == "test"
    assert test_summary["query_count"] == 80
    assert test_summary["dataset_digest"] == manifest["dataset_digest"]
    assert manifest["heldout_authorized"] is True
    assert manifest["heldout_consumed"] is True


def test_v03_manual_review_queue_is_complete_and_deterministic():
    with (REPORT / "v03_manual_review.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 88
    assert Counter(row["policy"] for row in rows) == {"B7a": 28, "B7b": 16, "B7c": 44}
    assert Counter(row["category"] for row in rows) == {
        "exact_reference": 32,
        "multi_evidence": 32,
        "clause_intro_point": 24,
    }
    assert all(row["review_status"] == "approved" for row in rows)
    assert all(row["verdict"] == "correct" for row in rows)
    assert all(row["reviewer"] == "review01" for row in rows)
    assert len({row["review_id"] for row in rows}) == 88
