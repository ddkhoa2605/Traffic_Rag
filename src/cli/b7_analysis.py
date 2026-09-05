from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.retrieval_eval.b7_dataset import (
    apply_b7_review_sheet,
    create_b7_benchmark_draft,
    validate_b7_dataset,
    validate_routing_golden_suite,
)
from src.retrieval_eval.b7_evaluator import (
    POLICIES,
    assess_b7d_gate,
    apply_b7_manual_review,
    evaluate_b7_policy,
    export_b7_manual_review,
    lock_b7_policy,
    select_b7_candidate,
)
from src.retrieval_eval.dense import DenseEncoder
from src.retrieval_runtime.config import load_application_config


def main() -> int:
    parser = argparse.ArgumentParser(description="Build and validate the B7 benchmark and routing suite")
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--create-draft", action="store_true")
    actions.add_argument("--validate", action="store_true")
    actions.add_argument("--validate-routing", action="store_true")
    actions.add_argument("--apply-review", metavar="CSV")
    actions.add_argument("--run-dev", metavar="POLICY")
    actions.add_argument("--run-test", metavar="POLICY")
    actions.add_argument("--export-manual-review", action="store_true")
    actions.add_argument("--assess-reranker", action="store_true")
    actions.add_argument("--apply-manual-review", metavar="CSV")
    actions.add_argument("--select", action="store_true")
    actions.add_argument("--lock", action="store_true")
    parser.add_argument("--require-approved", action="store_true")
    parser.add_argument("--root", default=None)
    parser.add_argument("--dataset-id", default=None, help="Research-only PostgreSQL dataset selector")
    parser.add_argument("--allow-building-dataset", action="store_true")
    parser.add_argument("--strategy-lock", default="reports/chunk_ablation/strategy_lock.json")
    parser.add_argument("--report-dir", default="reports/b7")
    parser.add_argument("--run-version", default=None)
    args = parser.parse_args()
    root = Path(args.root or Path.cwd()).resolve()
    if args.create_draft:
        result = create_b7_benchmark_draft(root)
    elif args.apply_review:
        result = apply_b7_review_sheet(root, args.apply_review)
    elif args.validate_routing:
        errors = validate_routing_golden_suite(root)
        result = {"status": "PASS" if not errors else "ERROR", "errors": errors}
    elif args.run_dev:
        policies = POLICIES if args.run_dev.casefold() == "all" else (args.run_dev,)
        shared_encoder = None
        if len(policies) > 1:
            encoder_config = load_application_config(root).query_encoder
            shared_encoder = DenseEncoder(
                encoder_config.model,
                encoder_config.revision,
                max_length=encoder_config.max_length,
                preferred_device=encoder_config.preferred_device,
            )
        result = {}
        for policy in policies:
            result[policy] = evaluate_b7_policy(
                root, policy, split="dev",
                application_kwargs={
                "dataset_id": args.dataset_id,
                "allow_building_dataset": args.allow_building_dataset,
                "strategy_lock_path": args.strategy_lock,
                "encoder": shared_encoder,
                },
                report_dir=args.report_dir,
                run_version=args.run_version,
                strategy_lock_path=args.strategy_lock,
            )
    elif args.run_test:
        result = evaluate_b7_policy(root, args.run_test, split="test")
    elif args.export_manual_review:
        result = export_b7_manual_review(root)
    elif args.assess_reranker:
        result = assess_b7d_gate(root)
    elif args.apply_manual_review:
        result = apply_b7_manual_review(root, args.apply_manual_review)
    elif args.select:
        result = select_b7_candidate(root)
    elif args.lock:
        result = lock_b7_policy(root)
    else:
        errors = validate_b7_dataset(root, require_approved=args.require_approved)
        result = {"status": "PASS" if not errors else "ERROR", "errors": errors}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if not result.get("errors") else 1


if __name__ == "__main__":
    raise SystemExit(main())
