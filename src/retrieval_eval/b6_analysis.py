from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from src.parser.io import read_json, read_jsonl, write_json, write_jsonl

from .dataset import load_dataset
from .locking import create_strategy_lock
from .round2 import RETRIEVERS, TAXONOMY_CODES, VERDICTS, REVIEW_STATUSES


B6_CANDIDATES = ("B6", "B4e", "B4d")
B6_DIR = Path("reports/chunk_ablation/b6")
TOP_K = (1, 3, 5, 10)


def _selection_tuple(metrics: dict, strategy: str) -> tuple:
    return (
        -float(metrics["recall@5"]), -float(metrics["mrr"]),
        -float(metrics["evidence_coverage@5"]), float(metrics["avg_evidence_tokens@5"]),
        float(metrics["duplicate_token_ratio@5"]), strategy,
    )


def _dominates(left: dict, right: dict) -> bool:
    maximize = ("recall@5", "mrr", "evidence_coverage@5")
    minimize = ("avg_evidence_tokens@5", "duplicate_token_ratio@5")
    return (
        all(left[key] >= right[key] for key in maximize)
        and all(left[key] <= right[key] for key in minimize)
        and (
            any(left[key] > right[key] for key in maximize)
            or any(left[key] < right[key] for key in minimize)
        )
    )


def select_b6_candidate(summaries: list[dict]) -> dict:
    by_strategy: dict[str, dict[str, dict]] = defaultdict(dict)
    for summary in summaries:
        strategy = summary["run_id"].split("__", 1)[0]
        if strategy in B6_CANDIDATES:
            by_strategy[strategy][summary["retriever"]] = summary
    missing = [strategy for strategy in B6_CANDIDATES if set(by_strategy[strategy]) != set(RETRIEVERS)]
    if missing:
        raise ValueError(f"Incomplete B6 comparison matrix: {missing}")
    keys = ("recall@5", "mrr", "evidence_coverage@5", "avg_evidence_tokens@5", "duplicate_token_ratio@5")
    means = {
        strategy: {
            key: statistics.mean(by_strategy[strategy][retriever]["metrics"][key] for retriever in RETRIEVERS)
            for key in keys
        }
        for strategy in B6_CANDIDATES
    }
    frontier = [
        strategy for strategy in B6_CANDIDATES
        if not any(_dominates(means[other], means[strategy]) for other in B6_CANDIDATES if other != strategy)
    ]
    ranked = sorted(B6_CANDIDATES, key=lambda strategy: _selection_tuple(means[strategy], strategy))
    return {
        "metrics_schema_version": "0.2.0",
        "winner": ranked[0],
        "runner_up": ranked[1],
        "ranked_candidates": ranked,
        "pareto_frontier": frontier,
        "mean_metrics": means,
        "retriever_winners": {
            retriever: min(
                B6_CANDIDATES,
                key=lambda strategy: _selection_tuple(by_strategy[strategy][retriever]["metrics"], strategy),
            )
            for retriever in RETRIEVERS
        },
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    fields.extend(key for row in rows[1:] for key in row if key not in fields)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _redundancy(rows: list[dict], top_k: int) -> dict:
    components = [
        component for row in sorted(rows, key=lambda value: value["rank"])[:top_k]
        for component in row.get("evidence_components", [])
    ]
    total = sum(int(component["token_count"]) for component in components)
    unique: dict[str, int] = {}
    counts: Counter[str] = Counter()
    component_ids: dict[str, set[str]] = defaultdict(set)
    for component in components:
        digest = component["token_sequence_hash"]
        unique.setdefault(digest, int(component["token_count"]))
        counts[digest] += 1
        component_ids[digest].add(f"{component['role']}:{component['node_id']}")
    duplicate = total - sum(unique.values())
    ratio = duplicate / total if total else 0.0
    if not 0 <= ratio <= 1:
        raise ValueError(f"Invalid B6 duplicate-token ratio: {ratio}")
    repeated = sorted(value for digest, values in component_ids.items() if counts[digest] > 1 for value in values)
    return {
        "top_k": top_k, "duplicate_token_ratio": ratio,
        "duplicate_token_count": duplicate, "total_component_tokens": total,
        "unique_component_tokens": total - duplicate,
        "repeated_component_ids": repeated,
    }


def _query_metric(rows: list[dict], required_ids: list[str], redundancy: dict) -> dict:
    top = sorted(rows, key=lambda value: value["rank"])[:5]
    exact_rank = next((row["rank"] for row in top if row["exact_hit"]), None)
    included = {node_id for row in top for node_id in row["included_node_ids"]}
    required = set(required_ids)
    return {
        "exact_hit@5": float(exact_rank is not None),
        "first_exact_rank": exact_rank,
        "structural_hit@5": float(any(row["structural_hit"] for row in top)),
        "evidence_coverage@5": len(required & included) / len(required) if required else 0.0,
        "evidence_tokens@5": sum(row["evidence_tokens"] for row in top),
        "duplicate_token_ratio@5": redundancy["duplicate_token_ratio"],
    }


def _load_summaries(root: Path, split: str = "dev") -> list[dict]:
    return [
        read_json(root / "reports/chunk_ablation/runs" / f"{strategy}__{retriever.upper()}__v001__{split}" / "summary.json")
        for strategy in B6_CANDIDATES for retriever in RETRIEVERS
    ]


def _ranking_digest(rows: list[dict]) -> str:
    digest = hashlib.sha256()
    for row in sorted(rows, key=lambda value: (value["query_id"], value["rank"])):
        value = {key: row[key] for key in ("query_id", "rank", "retrieved_passage_id", "retrieved_primary_node_id", "score")}
        digest.update(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def analyze_b6_dev(root: str | Path) -> dict:
    root_path = Path(root).resolve()
    report = root_path / B6_DIR
    benchmark = read_json(root_path / "data/08_eval/benchmark_manifest.json")
    queries, gold_records = load_dataset(root_path)
    query_by_id = {item.query_id: item for item in queries if item.split == "dev"}
    gold_by_id = {item.query_id: item for item in gold_records}
    summaries = _load_summaries(root_path)
    result_rows = []
    for summary in summaries:
        if summary["dataset_hash"] != benchmark["dataset_digest"]:
            raise ValueError(f"Stale dataset digest: {summary['run_id']}")
        if summary.get("provisional", True):
            raise ValueError(f"B6 comparison requires official dev run: {summary['run_id']}")
        if summary["run_id"].startswith("B6__") and not summary.get("ranking_digest"):
            run_dir = root_path / "reports/chunk_ablation/runs" / summary["run_id"]
            summary["ranking_digest"] = _ranking_digest(read_jsonl(run_dir / "per_query.jsonl"))
            write_json(run_dir / "summary.json", summary)
        result_rows.append({
            "run_id": summary["run_id"], "strategy": summary["run_id"].split("__", 1)[0],
            "retriever": summary["retriever"], "split": summary["split"],
            "metrics_schema_version": summary["metrics_schema_version"], **summary["metrics"],
        })
    _write_csv(report / "b6_results.csv", result_rows)

    redundancy_rows: list[dict] = []
    query_metric_rows: list[dict] = []
    for retriever in RETRIEVERS:
        run_id = f"B6__{retriever.upper()}__v001__dev"
        rows = read_jsonl(root_path / "reports/chunk_ablation/runs" / run_id / "per_query.jsonl")
        per_query: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            per_query[row["query_id"]].append(row)
        if set(per_query) != set(query_by_id):
            raise ValueError(f"{run_id}: expected exact 80-query dev set")
        for query_id in sorted(per_query):
            ranks = sorted(row["rank"] for row in per_query[query_id])
            if ranks != list(range(1, 11)):
                raise ValueError(f"{run_id}/{query_id}: ranks must be 1..10")
            at_five = None
            for top_k in TOP_K:
                value = _redundancy(per_query[query_id], top_k)
                if top_k == 5:
                    at_five = value
                redundancy_rows.append({
                    "run_id": run_id, "strategy": "B6", "retriever": retriever,
                    "query_id": query_id, "category": query_by_id[query_id].category,
                    **{**value, "repeated_component_ids": "|".join(value["repeated_component_ids"])},
                })
            query_metric_rows.append({
                "run_id": run_id, "strategy": "B6", "retriever": retriever,
                "query_id": query_id, "category": query_by_id[query_id].category,
                **_query_metric(per_query[query_id], gold_by_id[query_id].required_node_ids, at_five),
            })
    _write_csv(report / "b6_redundancy.csv", redundancy_rows)
    _write_csv(report / "b6_per_query_metrics.csv", query_metric_rows)
    selection = select_b6_candidate(summaries)
    _write_decision(root_path, selection, review_ready=False, locked=False)
    _write_failure(root_path, [], False)
    return {"selection": selection, "report_dir": str(report), "query_count": len(query_by_id)}


def _query_tuple(metric: dict) -> tuple:
    rank = metric.get("first_exact_rank")
    reciprocal = 1 / int(rank) if rank not in (None, "", "None") else 0.0
    return (
        float(metric["exact_hit@5"]), reciprocal, float(metric["evidence_coverage@5"]),
        -float(metric["evidence_tokens@5"]), -float(metric["duplicate_token_ratio@5"]),
    )


def _mean_tuple(values: list[tuple]) -> tuple:
    return tuple(statistics.mean(parts) for parts in zip(*values))


def _round_robin(candidates: list[dict], count: int) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for candidate in candidates:
        grouped[candidate["category"]].append(candidate)
    for values in grouped.values():
        values.sort(key=lambda value: value["sort_key"])
    selected: list[dict] = []
    while len(selected) < count:
        progressed = False
        for category in sorted(grouped):
            if grouped[category] and len(selected) < count:
                selected.append(grouped[category].pop(0)); progressed = True
        if not progressed:
            break
    if len(selected) != count:
        raise ValueError(f"B6 review bucket has only {len(selected)}/{count} eligible candidates")
    return selected


def _top5(rows: list[dict]) -> list[dict]:
    return [{
        "rank": row["rank"], "passage_id": row["retrieved_passage_id"],
        "primary_node_id": row["retrieved_primary_node_id"], "score": row["score"],
        "exact_hit": row["exact_hit"], "structural_hit": row["structural_hit"],
        "included_node_ids": row["included_node_ids"], "evidence_tokens": row["evidence_tokens"],
    } for row in sorted(rows, key=lambda value: value["rank"])[:5]]


def _suggest_taxonomy(category: str, bucket: str) -> str:
    if bucket == "false_positive_top1":
        return "F9" if category in {"point_specific", "clause_intro_point"} else "F6"
    if bucket == "clear_win":
        return "F7" if category == "exact_reference" else "F2"
    return {"semantic_paraphrase": "F6", "clause_intro_point": "F2", "multi_evidence": "F8",
            "point_specific": "F1", "hard_distractor": "F3", "exact_reference": "F7"}.get(category, "F3")


def export_b6_review(root: str | Path) -> dict:
    root_path = Path(root).resolve()
    report = root_path / B6_DIR
    queries, gold_records = load_dataset(root_path)
    query_by_id = {item.query_id: item for item in queries if item.split == "dev"}
    gold_by_id = {item.query_id: item for item in gold_records}
    with (report / "b6_per_query_metrics.csv").open(encoding="utf-8-sig", newline="") as handle:
        b6_metrics = list(csv.DictReader(handle))
    with (root_path / "reports/chunk_ablation/round2/round2_per_query_metrics.csv").open(encoding="utf-8-sig", newline="") as handle:
        baseline_metrics = list(csv.DictReader(handle))
    metric_map = {(row["strategy"], row["retriever"], row["query_id"]): row for row in [*baseline_metrics, *b6_metrics]}
    run_rows = {}
    for retriever in RETRIEVERS:
        rows = read_jsonl(root_path / "reports/chunk_ablation/runs" / f"B6__{retriever.upper()}__v001__dev" / "per_query.jsonl")
        grouped: dict[str, list[dict]] = defaultdict(list)
        for row in rows: grouped[row["query_id"]].append(row)
        run_rows[retriever] = grouped
    candidates = []
    for query_id, query in query_by_id.items():
        bm = metric_map[("B6", "bm25", query_id)]; de = metric_map[("B6", "dense", query_id)]
        bm_rows = sorted(run_rows["bm25"][query_id], key=lambda row: row["rank"])
        de_rows = sorted(run_rows["dense"][query_id], key=lambda row: row["rank"])
        exact_bm, exact_de = bool(float(bm["exact_hit@5"])), bool(float(de["exact_hit@5"]))
        recovered = any(
            not rows[0]["exact_hit"] and not rows[0]["structural_hit"] and any(row["exact_hit"] for row in rows[1:5])
            for rows in (bm_rows, de_rows)
        )
        b6_tuple = _mean_tuple([_query_tuple(bm), _query_tuple(de)])
        beaten = sum(
            b6_tuple > _mean_tuple([
                _query_tuple(metric_map[(other, retriever, query_id)]) for retriever in RETRIEVERS
            ]) for other in ("B4e", "B4d")
        )
        candidates.append({
            "query_id": query_id, "category": query.category,
            "failure": not exact_bm and not exact_de,
            "no_structural": not any(row["structural_hit"] for row in [*bm_rows[:5], *de_rows[:5]]),
            "coverage": statistics.mean((float(bm["evidence_coverage@5"]), float(de["evidence_coverage@5"]))),
            "false_positive": recovered and (exact_bm or exact_de),
            "clear_win": (exact_bm or exact_de) and beaten > 0,
            "beaten": beaten,
            "best_rank": min([int(value) for value in (bm["first_exact_rank"], de["first_exact_rank"]) if value not in (None, "", "None")], default=99),
        })
    failures = _round_robin([{**item, "sort_key": (not item["no_structural"], item["coverage"], item["query_id"])} for item in candidates if item["failure"]], 10)
    used = {item["query_id"] for item in failures}
    false_positives = _round_robin([{**item, "sort_key": (item["best_rank"], item["query_id"])} for item in candidates if item["false_positive"] and item["query_id"] not in used], 5)
    used.update(item["query_id"] for item in false_positives)
    wins = _round_robin([{**item, "sort_key": (-item["beaten"], item["best_rank"], -item["coverage"], item["query_id"])} for item in candidates if item["clear_win"] and item["query_id"] not in used], 5)
    queue = []
    for bucket, selected in (("failure", failures), ("false_positive_top1", false_positives), ("clear_win", wins)):
        for item in selected:
            query_id = item["query_id"]; query = query_by_id[query_id]; gold = gold_by_id[query_id]
            bm = metric_map[("B6", "bm25", query_id)]; de = metric_map[("B6", "dense", query_id)]
            queue.append({
                "review_id": f"B6__{query_id}", "strategy": "B6", "query_id": query_id,
                "category": query.category, "bucket": bucket, "query": query.query,
                "gold_node_ids": "|".join(gold.gold_node_ids), "required_node_ids": "|".join(gold.required_node_ids),
                "bm25_top5": json.dumps(_top5(run_rows["bm25"][query_id]), ensure_ascii=False, separators=(",", ":")),
                "dense_top5": json.dumps(_top5(run_rows["dense"][query_id]), ensure_ascii=False, separators=(",", ":")),
                "bm25_first_exact_rank": bm["first_exact_rank"], "dense_first_exact_rank": de["first_exact_rank"],
                "bm25_evidence_coverage@5": bm["evidence_coverage@5"], "dense_evidence_coverage@5": de["evidence_coverage@5"],
                "bm25_duplicate_token_ratio@5": bm["duplicate_token_ratio@5"], "dense_duplicate_token_ratio@5": de["duplicate_token_ratio@5"],
                "suggested_taxonomy": _suggest_taxonomy(query.category, bucket),
                "review_status": "draft", "verdict": "", "taxonomy_codes": "", "reviewer": "", "review_notes": "",
            })
    _write_csv(report / "b6_manual_review.csv", queue)
    json_rows = [{**row, "gold_node_ids": row["gold_node_ids"].split("|"), "required_node_ids": row["required_node_ids"].split("|"),
                  "bm25_top5": json.loads(row["bm25_top5"]), "dense_top5": json.loads(row["dense_top5"]), "taxonomy_codes": []} for row in queue]
    write_jsonl(report / "b6_manual_review.jsonl", json_rows)
    errors = validate_b6_review(report / "b6_manual_review.csv")
    if errors: raise ValueError("Generated B6 review is invalid: " + "; ".join(errors))
    return {"record_count": len(queue), "csv": str(report / "b6_manual_review.csv")}


def _read_review(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8-sig", newline="") as handle: return list(csv.DictReader(handle))


def validate_b6_review(path: str | Path, *, require_complete: bool = False) -> list[str]:
    rows = _read_review(path); errors = []
    if len(rows) != 20: errors.append(f"expected 20 records, got {len(rows)}")
    if len({row.get("review_id") for row in rows}) != len(rows): errors.append("duplicate review_id")
    if Counter(row.get("bucket") for row in rows) != Counter({"failure": 10, "false_positive_top1": 5, "clear_win": 5}): errors.append("invalid bucket quota")
    for index, row in enumerate(rows, start=2):
        if row.get("review_id") != f"B6__{row.get('query_id')}" or row.get("strategy") != "B6": errors.append(f"row {index}: invalid identity")
        if row.get("suggested_taxonomy") not in TAXONOMY_CODES: errors.append(f"row {index}: invalid suggested taxonomy")
        try:
            for field in ("bm25_top5", "dense_top5"):
                top = json.loads(row.get(field, ""))
                if [item["rank"] for item in top] != [1, 2, 3, 4, 5]: errors.append(f"row {index}: invalid {field}")
        except (json.JSONDecodeError, KeyError, TypeError): errors.append(f"row {index}: invalid top-5 JSON")
        status = row.get("review_status", "").strip().casefold(); verdict = row.get("verdict", "").strip().casefold()
        codes = {value for value in row.get("taxonomy_codes", "").split("|") if value}
        if status not in REVIEW_STATUSES: errors.append(f"row {index}: invalid review_status")
        if verdict and verdict not in VERDICTS: errors.append(f"row {index}: invalid verdict")
        if not codes.issubset(TAXONOMY_CODES): errors.append(f"row {index}: invalid taxonomy_codes")
        if require_complete:
            if status != "approved": errors.append(f"row {index}: not approved")
            if not verdict: errors.append(f"row {index}: verdict required")
            if verdict == "benchmark_issue": errors.append(f"row {index}: benchmark_issue invalidates runs")
            if not row.get("reviewer", "").strip(): errors.append(f"row {index}: reviewer required")
    return errors


def apply_b6_review(root: str | Path, reviewed_csv: str | Path) -> dict:
    root_path = Path(root).resolve(); canonical = root_path / B6_DIR / "b6_manual_review.csv"
    baseline = {row["review_id"]: row for row in _read_review(canonical)}; incoming = _read_review(reviewed_csv)
    if {row.get("review_id") for row in incoming} != set(baseline): raise ValueError("Reviewed B6 CSV must contain exact exported IDs")
    mutable = {"review_status", "verdict", "taxonomy_codes", "reviewer", "review_notes"}
    for row in incoming:
        changed = [key for key in baseline[row["review_id"]] if key not in mutable and row.get(key, "") != baseline[row["review_id"]].get(key, "")]
        if changed: raise ValueError(f"{row['review_id']}: immutable fields changed: {changed}")
    temporary = canonical.with_name(".b6_review_validating.csv"); _write_csv(temporary, incoming)
    errors = validate_b6_review(temporary); temporary.unlink(missing_ok=True)
    if errors: raise ValueError("B6 review invalid: " + "; ".join(errors[:20]))
    _write_csv(canonical, incoming)
    json_rows = []
    for row in incoming:
        value = dict(row)
        for key in ("gold_node_ids", "required_node_ids", "taxonomy_codes"): value[key] = [item for item in value[key].split("|") if item]
        for key in ("bm25_top5", "dense_top5"): value[key] = json.loads(value[key])
        json_rows.append(value)
    write_jsonl(root_path / B6_DIR / "b6_manual_review.jsonl", json_rows)
    complete = validate_b6_review(canonical, require_complete=True)
    _write_failure(root_path, json_rows, not complete)
    return {"approved_count": sum(row["review_status"].casefold() == "approved" for row in incoming), "ready_to_finalize": not complete, "errors": complete}


def _write_failure(root: Path, rows: list[dict], complete: bool) -> None:
    buckets = Counter(row.get("bucket") for row in rows); verdicts = Counter(row.get("verdict") for row in rows if row.get("verdict"))
    lines = ["# B6 Failure Analysis", "", f"Manual review complete: `{complete}`.", "", f"Records: `{len(rows)}`; buckets: `{dict(buckets)}`.", "", f"Verdicts: `{dict(verdicts)}`.", "", "A `benchmark_issue` invalidates B6 dev runs; legitimate retrieval failures remain in the report."]
    path = root / B6_DIR / "b6_failure_analysis.md"; path.parent.mkdir(parents=True, exist_ok=True); path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_decision(root: Path, selection: dict, *, review_ready: bool, locked: bool) -> None:
    lines = ["# B6 Dual-Granularity Decision", "", f"Status: **{'LOCKED' if locked else ('REVIEW COMPLETE' if review_ready else 'MANUAL REVIEW REQUIRED')}**.", "", "Selection rule is frozen: Recall@5 → MRR → EvidenceCoverage@5 → evidence tokens@5 → corrected duplicate-token ratio@5 → strategy ID.", "", "| Strategy | Recall@5 | MRR | Coverage@5 | Evidence tokens@5 | Duplicate ratio@5 |", "|---|---:|---:|---:|---:|---:|"]
    for strategy in B6_CANDIDATES:
        metric = selection["mean_metrics"][strategy]
        lines.append(f"| {strategy} | {metric['recall@5']:.4f} | {metric['mrr']:.4f} | {metric['evidence_coverage@5']:.4f} | {metric['avg_evidence_tokens@5']:.1f} | {metric['duplicate_token_ratio@5']:.4f} |")
    lines += ["", f"Dev winner: **{selection['winner']}**.", "", f"Runner-up: **{selection['runner_up']}**.", "", f"Pareto frontier: `{', '.join(selection['pareto_frontier'])}`.", "", f"BM25 winner: **{selection['retriever_winners']['bm25']}**; Dense winner: **{selection['retriever_winners']['dense']}**.", "", "Held-out test remains closed until the 20/20 B6 review gate passes and the winner lock is written.", "", "Citation policy: cite canonical `citation_node_ids`; never cite hybrid or passage IDs."]
    path = root / B6_DIR / "b6_decision.md"; path.parent.mkdir(parents=True, exist_ok=True); path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def finalize_b6(root: str | Path) -> dict:
    root_path = Path(root).resolve(); review = root_path / B6_DIR / "b6_manual_review.csv"
    errors = validate_b6_review(review, require_complete=True)
    if errors: raise ValueError("B6 review gate failed: " + "; ".join(errors[:20]))
    selection = select_b6_candidate(_load_summaries(root_path))
    lock = create_strategy_lock(root_path, [selection["winner"]])
    _write_decision(root_path, selection, review_ready=True, locked=True)
    _write_failure(root_path, _read_review(review), True)
    return {"selection": selection, "strategy_lock": lock, "test_opened": False}
