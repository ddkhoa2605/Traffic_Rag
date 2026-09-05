from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

from src.parser.io import read_jsonl
from src.retrieval_eval.benchmark_v03 import validate_v03_dataset
from src.retrieval_runtime.reference import normalize_reference_text


ROOT = Path(__file__).resolve().parents[2]
V03 = ROOT / "data/11_eval_v03"


def _older_query_texts() -> set[str]:
    return {
        normalize_reference_text(row["query"])
        for folder in ("data/08_eval", "data/09_eval_b7")
        for row in read_jsonl(ROOT / folder / "queries.jsonl")
    }


def _older_gold_required_ids() -> set[str]:
    return {
        node_id
        for folder in ("data/08_eval", "data/09_eval_b7")
        for row in read_jsonl(ROOT / folder / "gold.jsonl")
        for field in ("gold_node_ids", "required_node_ids")
        for node_id in row[field]
    }


def test_v03_frozen_dataset_is_balanced_novel_and_heldout_is_consumed_once():
    queries = read_jsonl(V03 / "queries.jsonl")
    gold = read_jsonl(V03 / "gold.jsonl")
    manifest = json.loads((V03 / "benchmark_manifest.json").read_text(encoding="utf-8"))

    assert len(queries) == len(gold) == 160
    assert Counter(row["split"] for row in queries) == {"dev": 80, "test": 80}
    assert Counter(row["document_ids"][0] for row in queries) == {
        "LAW_35_2024": 80,
        "LAW_36_2024": 80,
    }
    assert all(row["query_id"] == f"V3Q{index:04d}" for index, row in enumerate(queries, 1))
    assert not ({normalize_reference_text(row["query"]) for row in queries} & _older_query_texts())
    current_nodes = {
        node_id
        for row in gold
        for field in ("gold_node_ids", "required_node_ids")
        for node_id in row[field]
    }
    assert not (current_nodes & _older_gold_required_ids())
    assert manifest["benchmark_version"] == "eval-law35-36-v0.3.0"
    assert manifest["official_ready"] is True
    assert manifest["heldout_authorized"] is True
    assert manifest["heldout_consumed"] is True
    assert manifest["test_output_count"] == 1
    assert manifest["locked_winner"] == "B7c"
    assert validate_v03_dataset(ROOT) == []


def test_v03_review_sheet_covers_all_rows_and_approval_gate_is_open_for_dev_only():
    with (V03 / "review.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 160
    assert [row["query_id"] for row in rows] == [f"V3Q{index:04d}" for index in range(1, 161)]
    assert all(row["review_status"] == "approved" for row in rows)
    assert all(row["reviewer"] == "review01" for row in rows)
    approval_errors = validate_v03_dataset(ROOT, require_approved=True)
    assert approval_errors == []
