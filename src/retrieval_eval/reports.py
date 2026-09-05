from __future__ import annotations

import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

from src.chunking.builder import OUTPUT_NAMES
from src.chunking.models import RetrievalPassage
from src.parser.io import read_jsonl


def _percentile(values: list[int], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return float(ordered[round((len(ordered) - 1) * fraction)])


def compare_runs(root: str | Path, *, split: str = "dev") -> dict:
    root_path = Path(root).resolve()
    report_root = root_path / "reports/chunk_ablation"
    benchmark_manifest = json.loads(
        (root_path / "data/08_eval/benchmark_manifest.json").read_text(encoding="utf-8")
    )
    current_dataset_hash = benchmark_manifest["dataset_digest"]
    summaries = []
    stale_runs = []
    for path in sorted((report_root / "runs").glob(f"*__{split}/summary.json")):
        summary = json.loads(path.read_text(encoding="utf-8"))
        if summary.get("dataset_hash") != current_dataset_hash:
            stale_runs.append(summary.get("run_id", path.parent.name))
            continue
        summaries.append(summary)
    if not summaries:
        detail = f"; ignored {len(stale_runs)} stale run(s)" if stale_runs else ""
        raise FileNotFoundError(
            f"No {split} run summaries match benchmark dataset {current_dataset_hash}{detail}"
        )
    report_root.mkdir(parents=True, exist_ok=True)
    metric_keys = sorted({key for item in summaries for key in item["metrics"]})
    with (report_root / "results.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        fields = ["run_id", "strategy", "retriever", "split", "provisional", *metric_keys]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in summaries:
            writer.writerow({**{key: item[key] for key in fields[:5]}, **item["metrics"]})
    with (report_root / "per_query.csv").open("w", encoding="utf-8-sig", newline="") as output:
        writer = None
        for item in summaries:
            path = report_root / "runs" / item["run_id"] / "per_query.csv"
            with path.open(encoding="utf-8-sig", newline="") as source:
                reader = csv.DictReader(source)
                if writer is None:
                    writer = csv.DictWriter(output, fieldnames=reader.fieldnames)
                    writer.writeheader()
                writer.writerows(reader)
    chunk_rows = []
    for strategy_id, folder in OUTPUT_NAMES.items():
        path = root_path / "data/07_retrieval_ablation" / folder / "passages.jsonl"
        if not path.is_file():
            continue
        passages = [RetrievalPassage.model_validate(item) for item in read_jsonl(path)]
        for field, output_name in (("token_count_index", "index"), ("token_count_evidence", "evidence")):
            values = [getattr(item, field) for item in passages]
            chunk_rows.append({
                "strategy": strategy_id, "text_type": output_name, "passage_count": len(values),
                "min": min(values), "p25": _percentile(values, .25), "median": statistics.median(values),
                "p75": _percentile(values, .75), "p95": _percentile(values, .95),
                "max": max(values), "mean": round(statistics.mean(values), 3),
            })
    with (report_root / "chunk_statistics.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(chunk_rows[0]))
        writer.writeheader(); writer.writerows(chunk_rows)
    lines = ["# Chunking Ablation Summary", "", f"Split: `{split}`", "", "| Run | R@5 | MRR | EvidenceCov@5 | StructuralCov@5 | Evidence tokens@5 |", "|---|---:|---:|---:|---:|---:|"]
    for item in sorted(summaries, key=lambda value: value["metrics"].get("recall@5", 0), reverse=True):
        m = item["metrics"]
        lines.append(f"| {item['run_id']} | {m.get('recall@5',0):.3f} | {m.get('mrr',0):.3f} | {m.get('evidence_coverage@5',0):.3f} | {m.get('structural_coverage@5',0):.3f} | {m.get('avg_evidence_tokens@5',0):.1f} |")
    (report_root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    by_strategy = defaultdict(dict)
    for item in summaries:
        by_strategy[item["strategy"]][item["retriever"]] = item
    complete = {key: value for key, value in by_strategy.items() if {"bm25", "dense"}.issubset(value)}
    round1_complete = {key: value for key, value in complete.items() if key.startswith("B") and key in {
        "B0_fixed_window", "B1_article", "B2_clause", "B3_point", "B4_point_clause_context", "B5_child_parent"
    }}
    ranked_round1 = sorted(
        round1_complete,
        key=lambda key: -statistics.mean(run["metrics"].get("recall@5", 0) for run in round1_complete[key].values()),
    )
    b3 = round1_complete.get("B3_point")
    context_candidates = [key for key in ("B4_point_clause_context", "B5_child_parent") if key in round1_complete]
    top3_trigger = any(key in ranked_round1[:3] for key in context_candidates)
    evidence_trigger = False
    if b3:
        b3_recall = statistics.mean(run["metrics"]["recall@5"] for run in b3.values())
        b3_evidence = statistics.mean(run["metrics"]["evidence_coverage@5"] for run in b3.values())
        for key in context_candidates:
            candidate = round1_complete[key]
            recall = statistics.mean(run["metrics"]["recall@5"] for run in candidate.values())
            evidence = statistics.mean(run["metrics"]["evidence_coverage@5"] for run in candidate.values())
            evidence_trigger |= evidence - b3_evidence >= 0.03 and b3_recall - recall <= 0.02
    round2_triggered = top3_trigger or evidence_trigger
    lines += ["", "## Round 2 trigger", "", f"B4/B5 in Recall@5 top 3: `{top3_trigger}`  ", f"Evidence-gain rule met: `{evidence_trigger}`  ", f"Run B4a-B4e after official gold approval: `{round2_triggered}`"]
    (report_root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    decision_lines = ["# Chunking Decision", ""]
    provisional = any(item["provisional"] for item in summaries)
    approval_required = False
    retriever_winners: dict[str, str] = {}
    if provisional or len(complete) < 6:
        decision_lines += ["Status: **PROVISIONAL — NO WINNER FROZEN**", "", "Dataset gold is still draft or the 12-run matrix is incomplete."]
        winner = None
    else:
        ranked = sorted(complete.items(), key=lambda pair: (
            -statistics.mean(run["metrics"]["recall@5"] for run in pair[1].values()),
            -statistics.mean(run["metrics"]["mrr"] for run in pair[1].values()),
            -statistics.mean(run["metrics"]["evidence_coverage@5"] for run in pair[1].values()),
            statistics.mean(run["metrics"]["avg_evidence_tokens@5"] for run in pair[1].values()),
            statistics.mean(run["metrics"]["duplicate_token_ratio@5"] for run in pair[1].values()),
        ))
        winner = ranked[0][0]
        runner_up = ranked[1][0]
        winner_runs = complete[winner]
        runner_runs = complete[runner_up]
        for retriever in ("bm25", "dense"):
            retriever_ranked = sorted(
                (
                    (candidate, runs[retriever])
                    for candidate, runs in complete.items()
                    if retriever in runs
                ),
                key=lambda pair: (
                    -pair[1]["metrics"]["recall@5"],
                    -pair[1]["metrics"]["mrr"],
                    -pair[1]["metrics"]["evidence_coverage@5"],
                    pair[1]["metrics"]["avg_evidence_tokens@5"],
                    pair[1]["metrics"]["duplicate_token_ratio@5"],
                ),
            )
            retriever_winners[retriever] = retriever_ranked[0][0]
        approval_required = len(set(retriever_winners.values())) > 1
        decision_lines += [
            "Status: **DEV CANDIDATE — NOT PRODUCTION-FROZEN**" if approval_required else "Status: **DEV CANDIDATE**",
            "",
            f"Winner: **{winner}**", "", f"Runner-up: **{runner_up}**", "",
            "Selection follows Recall@5 → MRR → EvidenceCoverage@5 → context cost → redundancy.", "",
            f"Evaluated split: `{split}`", "",
            "| Candidate | Retriever | Recall@5 | MRR | EvidenceCoverage@5 | Evidence tokens@5 |",
            "|---|---|---:|---:|---:|---:|",
        ]
        for candidate, runs in ((winner, winner_runs), (runner_up, runner_runs)):
            for retriever, run in sorted(runs.items()):
                metric = run["metrics"]
                decision_lines.append(
                    f"| {candidate} | {retriever} | {metric['recall@5']:.3f} | {metric['mrr']:.3f} | "
                    f"{metric['evidence_coverage@5']:.3f} | {metric['avg_evidence_tokens@5']:.1f} |"
                )
        winner_mean = {
            key: statistics.mean(run["metrics"][key] for run in winner_runs.values())
            for key in ("recall@5", "mrr", "evidence_coverage@5", "avg_evidence_tokens@5", "duplicate_token_ratio@5")
        }
        runner_mean = {
            key: statistics.mean(run["metrics"][key] for run in runner_runs.values())
            for key in winner_mean
        }
        if approval_required:
            decision_lines += [
                "", "## Production-freeze gate", "",
                f"BM25 winner: **{retriever_winners['bm25']}**  ",
                f"Dense winner: **{retriever_winners['dense']}**  ",
                "The retrievers disagree, so user approval is required before locking a production strategy or opening the held-out test split.",
            ]
        decision_lines += [
            "", "## Candidate configuration", "", "```json",
            json.dumps({
                candidate: {retriever: run["retriever_config"] for retriever, run in runs.items()}
                for candidate, runs in ((winner, winner_runs), (runner_up, runner_runs))
            }, ensure_ascii=False, indent=2),
            "```", "", "## Trade-offs and known failures", "",
            f"- Mean Recall@5: `{winner_mean['recall@5']:.3f}` vs runner-up `{runner_mean['recall@5']:.3f}`.",
            f"- Mean MRR: `{winner_mean['mrr']:.3f}` vs runner-up `{runner_mean['mrr']:.3f}`.",
            f"- Mean EvidenceCoverage@5: `{winner_mean['evidence_coverage@5']:.3f}` vs runner-up `{runner_mean['evidence_coverage@5']:.3f}`.",
            f"- Mean evidence tokens@5: `{winner_mean['avg_evidence_tokens@5']:.1f}` vs runner-up `{runner_mean['avg_evidence_tokens@5']:.1f}`.",
            f"- Mean duplicate-token ratio@5: `{winner_mean['duplicate_token_ratio@5']:.3f}` vs runner-up `{runner_mean['duplicate_token_ratio@5']:.3f}`.",
            "", "Category Recall@5 exposes complementary failure modes:", "",
            "| Category | Winner BM25 | Winner Dense | Runner-up BM25 | Runner-up Dense |",
            "|---|---:|---:|---:|---:|",
        ]
        categories = sorted(set(winner_runs["bm25"]["category_metrics"]) | set(runner_runs["bm25"]["category_metrics"]))
        for category in categories:
            decision_lines.append(
                f"| {category} | {winner_runs['bm25']['category_metrics'][category]['recall@5']:.3f} | "
                f"{winner_runs['dense']['category_metrics'][category]['recall@5']:.3f} | "
                f"{runner_runs['bm25']['category_metrics'][category]['recall@5']:.3f} | "
                f"{runner_runs['dense']['category_metrics'][category]['recall@5']:.3f} |"
            )
        decision_lines += [
            "", "See `failure_analysis.md` for the automatic failure inventory; manual qualitative review remains required before production freeze.", "",
            "Citation policy: return canonical IDs from `citation_node_ids`; never cite a retrieval passage ID.",
        ]
    (report_root / "chunking_decision.md").write_text("\n".join(decision_lines) + "\n", encoding="utf-8")
    failure_lines = ["# Failure Analysis", "", "Status: automatic inventory complete; manual review pending.", ""]
    for item in sorted(summaries, key=lambda value: value["run_id"]):
        failed = []
        rows_path = report_root / "runs" / item["run_id"] / "per_query.jsonl"
        per_query = defaultdict(list)
        for row in read_jsonl(rows_path):
            if row["rank"] <= 5:
                per_query[row["query_id"]].append(row)
        failed = [query_id for query_id, rows in per_query.items() if not any(row["exact_hit"] for row in rows)]
        failure_lines.append(f"- `{item['run_id']}`: {len(failed)}/{len(per_query)} no ExactHit@5; sample: {', '.join(failed[:10])}")
    failure_lines += ["", "Manual review target per strategy: 10 failures, 5 false-positive top-1, and 5 clear wins.", "", "Taxonomy: F1 short leaf; F2 missing parent; F3 dilution; F4 broad article; F5 repeated parent; F6 paraphrase; F7 exact reference; F8 incomplete evidence; F9 citation granularity; F10 window boundary."]
    (report_root / "failure_analysis.md").write_text("\n".join(failure_lines) + "\n", encoding="utf-8")
    return {
        "run_count": len(summaries),
        "stale_run_count": len(stale_runs),
        "stale_run_ids": stale_runs,
        "dataset_hash": current_dataset_hash,
        "complete_strategy_count": len(complete),
        "provisional": provisional,
        "round2_triggered": round2_triggered,
        "winner": winner,
        "retriever_winners": retriever_winners,
        "approval_required_before_freeze": approval_required,
    }
