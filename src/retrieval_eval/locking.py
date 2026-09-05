from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import yaml

from src.parser.io import write_json
from src.chunking.builder import OUTPUT_NAMES, _code_digest

from .dataset import validate_dataset


LOCK_PATH = Path("reports/chunk_ablation/strategy_lock.json")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def create_strategy_lock(root: str | Path, strategies: list[str]) -> dict:
    root_path = Path(root).resolve()
    lock_path = root_path / LOCK_PATH
    if lock_path.exists():
        raise FileExistsError(f"Strategy lock already exists: {lock_path}")
    strategies = list(dict.fromkeys(strategies))
    if len(strategies) != 1:
        raise ValueError("Freeze requires exactly one winner")
    from .b6_analysis import _load_summaries, select_b6_candidate, validate_b6_review

    review_path = root_path / "reports/chunk_ablation/b6/b6_manual_review.csv"
    review_errors = validate_b6_review(review_path, require_complete=True) if review_path.exists() else ["B6 review missing"]
    if review_errors:
        raise ValueError("Cannot freeze before B6 20/20 review: " + "; ".join(review_errors[:20]))
    selection = select_b6_candidate(_load_summaries(root_path))
    if strategies != [selection["winner"]]:
        raise ValueError(f"Frozen strategy must equal strict-rule winner {selection['winner']}")
    errors = validate_dataset(root_path, require_approved=True)
    if errors:
        raise ValueError("Cannot freeze with unapproved/invalid gold: " + "; ".join(errors[:20]))
    for strategy in strategies:
        for retriever in ("BM25", "DENSE"):
            summary_path = root_path / "reports/chunk_ablation/runs" / f"{strategy}__{retriever}__v001__dev/summary.json"
            if not summary_path.is_file():
                raise FileNotFoundError(f"Missing official dev run: {summary_path}")
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if summary.get("provisional", True):
                raise ValueError(f"Dev run is provisional: {summary['run_id']}")
    benchmark = json.loads((root_path / "data/08_eval/benchmark_manifest.json").read_text(encoding="utf-8"))
    summary_hashes = {
        retriever.casefold(): _sha256(
            root_path / "reports/chunk_ablation/runs" / f"{strategies[0]}__{retriever}__v001__dev/summary.json"
        ) for retriever in ("BM25", "DENSE")
    }
    if strategies[0] == "B6":
        passage_manifests = {
            source: _sha256(root_path / "data/07_retrieval_ablation" / OUTPUT_NAMES[source] / "manifest.json")
            for source in ("B1", "B4e")
        }
    else:
        passage_manifests = {
            strategies[0]: _sha256(root_path / "data/07_retrieval_ablation" / OUTPUT_NAMES[strategies[0]] / "manifest.json")
        }
    dense_summary = json.loads((root_path / "reports/chunk_ablation/runs" / f"{strategies[0]}__DENSE__v001__dev/summary.json").read_text(encoding="utf-8"))
    retrieval_config = yaml.safe_load((root_path / "configs/retrieval.yaml").read_text(encoding="utf-8"))
    payload = {
        "version": "0.2.0",
        "strategies": strategies,
        "winner": strategies[0],
        "dataset_digest": benchmark["dataset_digest"],
        "chunking_config_sha256": _sha256(root_path / "configs/chunking.yaml"),
        "retrieval_config_sha256": _sha256(root_path / "configs/retrieval.yaml"),
        "hybrid_b6_config": retrieval_config.get("hybrid_b6"),
        "code_digest": _code_digest(root_path),
        "dev_summary_sha256": summary_hashes,
        "passage_manifest_sha256": passage_manifests,
        "dense_runtime": {
            "model": dense_summary["retriever_config"].get("model"),
            "revision": dense_summary["retriever_config"].get("resolved_revision"),
            "device": dense_summary["retriever_config"].get("resolved_device"),
            "dtype": dense_summary["retriever_config"].get("resolved_dtype"),
        },
        "selection_rule": ["recall@5", "mrr", "evidence_coverage@5", "avg_evidence_tokens@5", "duplicate_token_ratio@5", "strategy_id"],
        "locked_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(lock_path, payload)
    return payload


def authorize_test_run(root: str | Path, strategies: list[str], retriever: str, official: bool) -> dict:
    root_path = Path(root).resolve()
    if not official:
        raise ValueError("Held-out test requires --official")
    lock_path = root_path / LOCK_PATH
    if not lock_path.is_file():
        raise FileNotFoundError("Held-out test is locked; create strategy_lock.json after dev selection")
    payload = json.loads(lock_path.read_text(encoding="utf-8"))
    if list(dict.fromkeys(strategies)) != payload["strategies"]:
        raise ValueError("Held-out run must use the single frozen winner")
    benchmark = json.loads((root_path / "data/08_eval/benchmark_manifest.json").read_text(encoding="utf-8"))
    expected = {
        "chunking_config_sha256": _sha256(root_path / "configs/chunking.yaml"),
        "retrieval_config_sha256": _sha256(root_path / "configs/retrieval.yaml"),
        "dataset_digest": benchmark["dataset_digest"],
        "code_digest": _code_digest(root_path),
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"Frozen config changed: {key}")
    for strategy in strategies:
        path = root_path / "reports/chunk_ablation/runs" / f"{strategy}__{retriever.upper()}__v001__test/summary.json"
        if path.exists():
            raise FileExistsError(f"Held-out run already exists and cannot be overwritten: {path}")
    return payload
