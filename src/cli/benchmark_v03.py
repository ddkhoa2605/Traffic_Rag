from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.retrieval_eval.benchmark_v03 import (
    apply_v03_review,
    create_v03_draft,
    load_v03_dataset,
    load_v03_config,
    validate_v03_dataset,
    validate_v03_review,
)
from src.retrieval_eval.b7_evaluator import (
    POLICIES,
    apply_b7_manual_review,
    evaluate_b7_policy,
    export_b7_manual_review,
    lock_b7_policy,
    select_b7_candidate,
)
from src.retrieval_eval.dense import DenseEncoder
from src.retrieval_runtime.config import load_application_config


def main() -> int:
    parser = argparse.ArgumentParser(description="Create and validate the unseen v0.3 benchmark")
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--create-draft", action="store_true")
    actions.add_argument("--validate", action="store_true")
    actions.add_argument("--apply-review")
    actions.add_argument("--validate-review")
    actions.add_argument("--run-dev", metavar="POLICY")
    actions.add_argument("--export-manual-review", action="store_true")
    actions.add_argument("--apply-manual-review", metavar="CSV")
    actions.add_argument("--select", action="store_true")
    actions.add_argument("--lock", action="store_true")
    actions.add_argument("--run-test", metavar="POLICY")
    parser.add_argument("--require-approved", action="store_true")
    parser.add_argument("--reviewer", default=None, help="Override reviewer on every imported review row")
    parser.add_argument("--root", default=None)
    parser.add_argument("--dataset-id", default=None, help="Research-only PostgreSQL dataset selector")
    parser.add_argument("--allow-building-dataset", action="store_true")
    parser.add_argument("--strategy-lock", default="reports/chunk_ablation/strategy_lock.json")
    parser.add_argument("--report-dir", default="reports/b7_v03")
    parser.add_argument("--run-version", default="v001")
    args = parser.parse_args()
    root = Path(args.root or Path.cwd()).resolve()
    if args.create_draft:
        result = create_v03_draft(root)
    elif args.select:
        result = select_b7_candidate(
            root,
            report_dir=args.report_dir,
            run_version=args.run_version,
            review_filename="v03_manual_review.csv",
            decision_filename="v03_decision.json",
        )
    elif args.lock:
        if not args.dataset_id:
            parser.error("--lock requires --dataset-id")
        result = lock_b7_policy(
            root,
            report_dir=args.report_dir,
            run_version=args.run_version,
            decision_filename="v03_decision.json",
            dataset_dir=load_v03_config(root)["data_dir"],
            dataset_id=args.dataset_id,
            strategy_lock_path=args.strategy_lock,
        )
    elif args.run_test:
        if not args.dataset_id:
            parser.error("--run-test requires --dataset-id")
        result = evaluate_b7_policy(
            root,
            args.run_test,
            split="test",
            application_kwargs={
                "dataset_id": args.dataset_id,
                "allow_building_dataset": args.allow_building_dataset,
                "strategy_lock_path": args.strategy_lock,
            },
            report_dir=args.report_dir,
            run_version=args.run_version,
            strategy_lock_path=args.strategy_lock,
            dataset_loader=load_v03_dataset,
            dataset_validator=validate_v03_dataset,
            dataset_dir=load_v03_config(root)["data_dir"],
        )
    elif args.export_manual_review:
        result = export_b7_manual_review(
            root,
            report_dir=args.report_dir,
            run_version=args.run_version,
            output_filename="v03_manual_review.csv",
        )
    elif args.apply_manual_review:
        result = apply_b7_manual_review(
            root,
            args.apply_manual_review,
            report_dir=args.report_dir,
            review_filename="v03_manual_review.csv",
        )
    elif args.run_dev:
        policies = POLICIES if args.run_dev.casefold() == "all" else (args.run_dev,)
        invalid = [policy for policy in policies if policy not in POLICIES]
        if invalid:
            parser.error(f"unsupported policy: {', '.join(invalid)}")
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
                root,
                policy,
                split="dev",
                application_kwargs={
                    "dataset_id": args.dataset_id,
                    "allow_building_dataset": args.allow_building_dataset,
                    "strategy_lock_path": args.strategy_lock,
                    "encoder": shared_encoder,
                },
                report_dir=args.report_dir,
                run_version=args.run_version,
                strategy_lock_path=args.strategy_lock,
                dataset_loader=load_v03_dataset,
                dataset_validator=validate_v03_dataset,
                dataset_dir=load_v03_config(root)["data_dir"],
            )
    elif args.validate_review:
        result = validate_v03_review(root, args.validate_review)
    elif args.apply_review:
        result = apply_v03_review(root, args.apply_review, reviewer_override=args.reviewer)
    else:
        errors = validate_v03_dataset(root, require_approved=args.require_approved)
        result = {"status": "PASS" if not errors else "ERROR", "errors": errors}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if not result.get("errors") else 1


if __name__ == "__main__":
    raise SystemExit(main())
