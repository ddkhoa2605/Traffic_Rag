from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from src.chunking.builder import OUTPUT_NAMES, ROUND1_STRATEGIES, ROUND2_STRATEGIES, STRATEGIES
from src.chunking.models import RetrievalPassage
from src.parser.io import read_jsonl
from src.retrieval_eval.dense import DenseEncoder
from src.retrieval_eval.evaluator import evaluate_run
from src.retrieval_eval.locking import authorize_test_run


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate chunk strategies")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--strategy", choices=[*STRATEGIES, "B6"])
    target.add_argument("--all", action="store_true")
    target.add_argument("--round2-all", action="store_true")
    parser.add_argument("--retriever", required=True, choices=["bm25", "dense"])
    parser.add_argument("--split", default="dev", choices=["dev", "test"])
    parser.add_argument("--official", action="store_true", help="Require all gold records to be approved")
    parser.add_argument("--root", default=None)
    parser.add_argument("--artifact-root", default="data/07_retrieval_ablation")
    parser.add_argument("--report-root", default="reports/chunk_ablation/runs")
    parser.add_argument("--run-version", default="v001")
    args = parser.parse_args()
    root = Path(args.root or Path.cwd()).resolve()
    selected = list(ROUND2_STRATEGIES) if args.round2_all else (list(ROUND1_STRATEGIES) if args.all else [args.strategy])
    if args.split == "test":
        authorize_test_run(root, selected, args.retriever, args.official)
    encoder = None
    if args.retriever == "dense":
        with (root / "configs/retrieval.yaml").open(encoding="utf-8") as handle:
            config = yaml.safe_load(handle)["dense"]
        encoder = DenseEncoder(config["model"], config["revision"], max_length=config["max_length"], preferred_device=config["preferred_device"])
        # Resolve the device once for the complete selected matrix. If the
        # worst passage does not fit on CUDA, every strategy uses CPU FP32.
        source_strategies = [
            source
            for strategy_id in selected
            for source in (("B1", "B4e") if strategy_id == "B6" else (strategy_id,))
        ]
        longest_text = max((
            RetrievalPassage.model_validate(item).index_text
            for strategy_id in dict.fromkeys(source_strategies)
            for item in read_jsonl(
                (Path(args.artifact_root) if Path(args.artifact_root).is_absolute() else root / args.artifact_root)
                / OUTPUT_NAMES[strategy_id] / "passages.jsonl"
            )
        ), key=len)
        encoder.preflight(longest_text)
    summaries = {}
    for strategy_id in selected:
        summary = evaluate_run(
            root, strategy_id, args.retriever, split=args.split,
            official=args.official, dense_encoder=encoder,
            artifact_root=args.artifact_root,
            report_root=args.report_root,
            run_version=args.run_version,
        )
        summaries[strategy_id] = summary.model_dump(mode="json")
    print(json.dumps(summaries, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
