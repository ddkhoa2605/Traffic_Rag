from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import orjson

from src.chunking.builder import OUTPUT_NAMES
from src.chunking.models import RetrievalPassage
from src.chunking.token_counter import BGETokenCounter, TokenCounter
from src.legal_tree.loader import load_legal_document
from src.legal_tree.resolver import LegalTreeResolver
from src.parser.io import read_json, read_jsonl, write_json, write_jsonl

from .dataset import load_dataset


METRICS_SCHEMA_VERSION = "0.2.0"
ROUND2_VARIANTS = ("B4a", "B4b", "B4c", "B4d", "B4e")
RETRIEVERS = ("bm25", "dense")
TOP_K = (1, 3, 5, 10)
TAXONOMY_CODES = {f"F{index}" for index in range(1, 11)}
VERDICTS = {"expected_win", "strategy_failure", "false_positive", "benchmark_issue", "ambiguous"}
REVIEW_STATUSES = {"draft", "approved", "rejected"}
ROUND2_DIR = Path("reports/chunk_ablation/round2")


@dataclass(frozen=True)
class PassageComponent:
    node_id: str
    role: str
    text: str
    token_sequence_hash: str
    token_count: int

    @property
    def component_id(self) -> str:
        return f"{self.role}:{self.node_id}"


def _first_line(text: str) -> str:
    return text.splitlines()[0].strip() if text.strip() else ""


def _token_hash(token_ids: list[int]) -> str:
    return hashlib.sha256(orjson.dumps(token_ids)).hexdigest()


def project_b4_components(
    passage: RetrievalPassage,
    resolver: LegalTreeResolver,
    token_counter: TokenCounter,
) -> list[PassageComponent]:
    """Project the exact text components present in a B4a-e evidence passage."""
    if not passage.passage_id.startswith(ROUND2_VARIANTS):
        raise ValueError(f"Not a B4a-e passage: {passage.passage_id}")
    context_ids = set(passage.context_node_ids)
    result: list[PassageComponent] = []
    for node_id in passage.source_node_ids:
        node = resolver.get_node(node_id)
        if node_id in context_ids and node.type == "document":
            # The document component is a retrieval projection, not canonical
            # Document.text. Read the exact first line from the passage so old
            # and corrected releases can both be audited without substituting
            # the canonical PDF header/body.
            role, component_text = "document_title", _first_line(passage.evidence_text)
        elif node_id in context_ids and node.type == "article":
            role, component_text = "article_heading", _first_line(node.text)
        elif node_id in context_ids and node.type == "clause":
            role, component_text = "clause_intro", node.text.strip()
        else:
            role, component_text = node.type, node.text.strip()
        if not component_text:
            raise ValueError(f"Empty {role} component in {passage.passage_id}: {node_id}")
        encoded = token_counter.encode_with_offsets(component_text)
        if not encoded.token_ids:
            raise ValueError(f"Zero-token component in {passage.passage_id}: {node_id}")
        result.append(PassageComponent(
            node_id=node_id,
            role=role,
            text=component_text,
            token_sequence_hash=_token_hash(encoded.token_ids),
            token_count=len(encoded.token_ids),
        ))
    reconstructed = "\n".join(component.text for component in result).strip()
    if reconstructed != passage.evidence_text:
        raise ValueError(
            f"Component projection does not reconstruct {passage.passage_id}: "
            f"{len(reconstructed)} != {len(passage.evidence_text)} characters"
        )
    projected_tokens = token_counter.count(reconstructed)
    if projected_tokens != passage.token_count_evidence:
        raise ValueError(
            f"Projected token count mismatch for {passage.passage_id}: "
            f"{projected_tokens} != {passage.token_count_evidence}"
        )
    return result


def compute_query_redundancy(
    rows: list[dict],
    passage_map: dict[str, RetrievalPassage],
    components: dict[str, list[PassageComponent]],
    top_k: int,
) -> dict:
    selected = sorted(rows, key=lambda row: row["rank"])[:top_k]
    missing_passages = [
        row["retrieved_passage_id"] for row in selected
        if row["retrieved_passage_id"] not in passage_map
    ]
    if missing_passages:
        raise ValueError(f"Ranking contains unknown passages: {missing_passages}")
    occurrences = [
        component
        for row in selected
        for component in components[row["retrieved_passage_id"]]
    ]
    total = sum(component.token_count for component in occurrences)
    unique_by_hash: dict[str, PassageComponent] = {}
    hash_counts: Counter[str] = Counter()
    for component in occurrences:
        hash_counts[component.token_sequence_hash] += 1
        unique_by_hash.setdefault(component.token_sequence_hash, component)
    unique = sum(component.token_count for component in unique_by_hash.values())
    duplicate = total - unique
    ratio = duplicate / total if total else 0.0
    if not 0.0 <= ratio <= 1.0:
        raise ValueError(f"Invalid duplicate-token ratio {ratio}")
    repeated_hashes = {value for value, count in hash_counts.items() if count > 1}
    repeated_ids = sorted({
        component.component_id for component in occurrences
        if component.token_sequence_hash in repeated_hashes
    })
    return {
        "top_k": top_k,
        "duplicate_token_ratio": ratio,
        "duplicate_token_count": duplicate,
        "total_component_tokens": total,
        "unique_component_tokens": unique,
        "repeated_component_ids": repeated_ids,
    }


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, math.ceil(fraction * len(ordered)) - 1)
    return float(ordered[index])


def _ranking_digest(rows: Iterable[dict]) -> str:
    digest = hashlib.sha256()
    for row in sorted(rows, key=lambda value: (value["query_id"], value["rank"])):
        immutable = {
            "query_id": row["query_id"],
            "rank": row["rank"],
            "retrieved_passage_id": row["retrieved_passage_id"],
            "retrieved_primary_node_id": row["retrieved_primary_node_id"],
            "score": row["score"],
        }
        digest.update(orjson.dumps(immutable, option=orjson.OPT_SORT_KEYS))
        digest.update(b"\n")
    return digest.hexdigest()


def _query_retrieval_metrics(rows: list[dict], gold, redundancy: dict) -> dict:
    selected = sorted(rows, key=lambda row: row["rank"])[:5]
    first_exact = next((row["rank"] for row in selected if row["exact_hit"]), None)
    included = {node_id for row in selected for node_id in row["included_node_ids"]}
    required = set(gold.required_node_ids)
    return {
        "exact_hit@5": float(first_exact is not None),
        "first_exact_rank": first_exact,
        "structural_hit@5": float(any(row["structural_hit"] for row in selected)),
        "evidence_coverage@5": len(required & included) / len(required) if required else 0.0,
        "evidence_tokens@5": sum(row["evidence_tokens"] for row in selected),
        "duplicate_token_ratio@5": redundancy["duplicate_token_ratio"],
    }


def _summary_statistics(records: list[dict], key: str) -> tuple[float, float, float]:
    values = [float(record[key]) for record in records]
    return statistics.mean(values), statistics.median(values), _percentile(values, .95)


def _correct_metric_dict(metric_dict: dict, records_by_k: dict[int, list[dict]], legacy: dict) -> dict:
    result = dict(metric_dict)
    for k in TOP_K:
        legacy_key = f"duplicate_token_ratio_legacy@{k}"
        old_key = f"duplicate_token_ratio@{k}"
        result[legacy_key] = float(legacy.get(legacy_key, legacy.get(old_key, 0.0)))
        mean, median, p95 = _summary_statistics(records_by_k[k], "duplicate_token_ratio")
        result[old_key] = round(mean, 6)
        result[f"duplicate_token_ratio_median@{k}"] = round(median, 6)
        result[f"duplicate_token_ratio_p95@{k}"] = round(p95, 6)
        result[f"avg_duplicate_token_count@{k}"] = round(
            statistics.mean(record["duplicate_token_count"] for record in records_by_k[k]), 6
        )
        result[f"avg_total_component_tokens@{k}"] = round(
            statistics.mean(record["total_component_tokens"] for record in records_by_k[k]), 6
        )
    return result


def _round2_paths(root: Path) -> tuple[Path, Path]:
    report_root = root / ROUND2_DIR
    run_root = root / "reports/chunk_ablation/runs"
    report_root.mkdir(parents=True, exist_ok=True)
    return report_root, run_root


def recompute_round2_metrics(
    root: str | Path,
    *,
    split: str = "dev",
    token_counter: TokenCounter | None = None,
) -> dict:
    """Re-aggregate B4a-e from saved ranks; never rerank or re-embed."""
    root_path = Path(root).resolve()
    report_root, run_root = _round2_paths(root_path)
    benchmark_manifest = read_json(root_path / "data/08_eval/benchmark_manifest.json")
    queries, gold_records = load_dataset(root_path)
    query_by_id = {item.query_id: item for item in queries if item.split == split}
    gold_by_id = {item.query_id: item for item in gold_records}

    passage_maps: dict[str, dict[str, RetrievalPassage]] = {}
    component_maps: dict[str, dict[str, list[PassageComponent]]] = {}
    passage_manifests: dict[str, dict] = {}
    all_document_ids: set[str] = set()
    for strategy_id in ROUND2_VARIANTS:
        passage_dir = root_path / "data/07_retrieval_ablation" / OUTPUT_NAMES[strategy_id]
        manifest = read_json(passage_dir / "manifest.json")
        passages = [RetrievalPassage.model_validate(item) for item in read_jsonl(passage_dir / "passages.jsonl")]
        passage_maps[strategy_id] = {passage.passage_id: passage for passage in passages}
        passage_manifests[strategy_id] = manifest
        all_document_ids.update(manifest["documents"])
    resolvers = {
        document_id: LegalTreeResolver(load_legal_document(root_path, document_id).nodes)
        for document_id in sorted(all_document_ids)
    }
    revision = passage_manifests["B4a"]["tokenizer_revision"]
    counter = token_counter or BGETokenCounter(revision)
    for strategy_id, passage_map in passage_maps.items():
        component_maps[strategy_id] = {
            passage_id: project_b4_components(passage, resolvers[passage.document_id], counter)
            for passage_id, passage in passage_map.items()
        }

    redundancy_rows: list[dict] = []
    query_metric_rows: list[dict] = []
    result_rows: list[dict] = []
    run_artifacts: dict[str, dict] = {}
    invariant_values = defaultdict(set)
    for strategy_id in ROUND2_VARIANTS:
        for retriever in RETRIEVERS:
            run_id = f"{strategy_id}__{retriever.upper()}__v001__{split}"
            run_dir = run_root / run_id
            summary_path = run_dir / "summary.json"
            summary = read_json(summary_path)
            rows = read_jsonl(run_dir / "per_query.jsonl")
            digest_before = _ranking_digest(rows)
            if summary["dataset_hash"] != benchmark_manifest["dataset_digest"]:
                raise ValueError(f"{run_id}: stale dataset digest")
            if summary["passage_manifest_hash"] != passage_manifests[strategy_id]["passage_digest"]:
                raise ValueError(f"{run_id}: passage manifest hash changed")
            for field in ("dataset_hash", "passage_manifest_hash"):
                invariant_values[field].add(summary[field])
            invariant_values["tokenizer_revision"].add(passage_manifests[strategy_id]["tokenizer_revision"])
            if retriever == "dense":
                for field in ("resolved_revision", "resolved_device", "resolved_dtype"):
                    invariant_values[field].add(summary["retriever_config"].get(field))
            per_query = defaultdict(list)
            for row in rows:
                if row["query_id"] in query_by_id:
                    per_query[row["query_id"]].append(row)
            if set(per_query) != set(query_by_id):
                raise ValueError(f"{run_id}: query set differs from {split} benchmark")
            for query_id, query_rows in per_query.items():
                ranks = sorted(row["rank"] for row in query_rows)
                if ranks != list(range(1, 11)):
                    raise ValueError(f"{run_id}/{query_id}: expected ranks 1..10, got {ranks}")
            per_k: dict[int, list[dict]] = {k: [] for k in TOP_K}
            category_per_k: dict[str, dict[int, list[dict]]] = defaultdict(lambda: {k: [] for k in TOP_K})
            for query_id in sorted(per_query):
                query = query_by_id[query_id]
                for k in TOP_K:
                    value = compute_query_redundancy(
                        per_query[query_id], passage_maps[strategy_id], component_maps[strategy_id], k
                    )
                    record = {
                        "run_id": run_id,
                        "strategy": strategy_id,
                        "retriever": retriever,
                        "query_id": query_id,
                        "category": query.category,
                        **value,
                    }
                    per_k[k].append(record)
                    category_per_k[query.category][k].append(record)
                    redundancy_rows.append({
                        **record,
                        "repeated_component_ids": "|".join(value["repeated_component_ids"]),
                    })
                qmetric = _query_retrieval_metrics(
                    per_query[query_id], gold_by_id[query_id], per_k[5][-1]
                )
                query_metric_rows.append({
                    "run_id": run_id,
                    "strategy": strategy_id,
                    "retriever": retriever,
                    "query_id": query_id,
                    "category": query.category,
                    **qmetric,
                })

            archive_path = run_dir / "summary.metrics-v0.1.json"
            if not archive_path.exists():
                shutil.copy2(summary_path, archive_path)
            legacy_summary = read_json(archive_path)
            immutable_summary_fields = (
                "run_id", "strategy", "retriever", "split", "query_count", "retriever_config",
                "corpus_digests", "passage_manifest_hash", "dataset_hash",
            )
            changed_fields = [
                field for field in immutable_summary_fields
                if summary.get(field) != legacy_summary.get(field)
            ]
            if changed_fields:
                raise ValueError(f"{run_id}: immutable summary fields changed: {changed_fields}")
            summary["metrics"] = _correct_metric_dict(summary["metrics"], per_k, legacy_summary["metrics"])
            for category, metric_dict in summary["category_metrics"].items():
                summary["category_metrics"][category] = _correct_metric_dict(
                    metric_dict, category_per_k[category], legacy_summary["category_metrics"][category]
                )
            summary["metrics_schema_version"] = METRICS_SCHEMA_VERSION
            summary["ranking_digest"] = digest_before
            write_json(summary_path, summary)
            if _ranking_digest(read_jsonl(run_dir / "per_query.jsonl")) != digest_before:
                raise RuntimeError(f"{run_id}: ranking changed during metric migration")
            result_rows.append({
                "run_id": run_id,
                "strategy": strategy_id,
                "retriever": retriever,
                "split": split,
                "metrics_schema_version": METRICS_SCHEMA_VERSION,
                "ranking_digest": digest_before,
                **summary["metrics"],
            })
            run_artifacts[run_id] = {
                "summary": summary,
                "rows": rows,
                "query_rows": per_query,
                "passage_map": passage_maps[strategy_id],
                "components": component_maps[strategy_id],
            }

    _write_csv(report_root / "round2_results.csv", result_rows)
    _write_csv(report_root / "round2_redundancy.csv", redundancy_rows)
    _write_csv(report_root / "round2_per_query_metrics.csv", query_metric_rows)
    invariant_report = {
        "metrics_schema_version": METRICS_SCHEMA_VERSION,
        "split": split,
        "run_count": len(run_artifacts),
        "dataset_digest": benchmark_manifest["dataset_digest"],
        "ranking_digests": {run_id: artifact["summary"]["ranking_digest"] for run_id, artifact in run_artifacts.items()},
        "retriever_config_digests": {
            run_id: hashlib.sha256(orjson.dumps(
                artifact["summary"]["retriever_config"], option=orjson.OPT_SORT_KEYS
            )).hexdigest()
            for run_id, artifact in run_artifacts.items()
        },
        "invariants": {key: sorted(value, key=lambda item: str(item)) for key, value in invariant_values.items()},
        "top_k_ranks": list(TOP_K),
        "saved_rank_depth": 10,
        "ranking_or_embedding_recomputed": False,
    }
    write_json(report_root / "round2_migration_manifest.json", invariant_report)
    selection = select_round2_candidate([artifact["summary"] for artifact in run_artifacts.values()])
    _write_decision_report(root_path, selection, review_errors=["manual review not exported or complete"])
    _write_failure_report(root_path, [], review_complete=False)
    return {
        "run_count": len(run_artifacts),
        "selection": selection,
        "report_dir": str(report_root),
        "ranking_unchanged": True,
    }


def _strategy_id(summary: dict) -> str:
    run_id = summary.get("run_id", "")
    return run_id.split("__", 1)[0]


def _selection_tuple(metrics: dict, strategy_id: str) -> tuple:
    return (
        -metrics["recall@5"],
        -metrics["mrr"],
        -metrics["evidence_coverage@5"],
        metrics["avg_evidence_tokens@5"],
        metrics["duplicate_token_ratio@5"],
        strategy_id,
    )


def _dominates(left: dict, right: dict) -> bool:
    maximize = ("recall@5", "mrr", "evidence_coverage@5")
    minimize = ("avg_evidence_tokens@5", "duplicate_token_ratio@5")
    no_worse = all(left[key] >= right[key] for key in maximize) and all(left[key] <= right[key] for key in minimize)
    better = any(left[key] > right[key] for key in maximize) or any(left[key] < right[key] for key in minimize)
    return no_worse and better


def select_round2_candidate(run_summaries: list[dict]) -> dict:
    by_strategy: dict[str, dict[str, dict]] = defaultdict(dict)
    for raw in run_summaries:
        summary = raw.model_dump(mode="json") if hasattr(raw, "model_dump") else raw
        strategy_id = _strategy_id(summary)
        if strategy_id in ROUND2_VARIANTS:
            by_strategy[strategy_id][summary["retriever"]] = summary
    missing = [strategy for strategy in ROUND2_VARIANTS if set(by_strategy[strategy]) != set(RETRIEVERS)]
    if missing:
        raise ValueError(f"Incomplete Round 2 run matrix: {missing}")
    keys = ("recall@5", "mrr", "evidence_coverage@5", "avg_evidence_tokens@5", "duplicate_token_ratio@5")
    means = {
        strategy: {
            key: statistics.mean(by_strategy[strategy][retriever]["metrics"][key] for retriever in RETRIEVERS)
            for key in keys
        }
        for strategy in ROUND2_VARIANTS
    }
    frontier = [
        strategy for strategy in ROUND2_VARIANTS
        if not any(_dominates(means[other], means[strategy]) for other in ROUND2_VARIANTS if other != strategy)
    ]
    ranked = sorted(ROUND2_VARIANTS, key=lambda strategy: _selection_tuple(means[strategy], strategy))
    winner = ranked[0]
    frontier_remaining = [strategy for strategy in frontier if strategy != winner]
    challenger = min(frontier_remaining, key=lambda strategy: _selection_tuple(means[strategy], strategy)) if frontier_remaining else ranked[1]
    retriever_winners = {}
    for retriever in RETRIEVERS:
        retriever_winners[retriever] = min(
            ROUND2_VARIANTS,
            key=lambda strategy: _selection_tuple(by_strategy[strategy][retriever]["metrics"], strategy),
        )
    return {
        "metrics_schema_version": METRICS_SCHEMA_VERSION,
        "winner": winner,
        "challenger": challenger,
        "pareto_frontier": frontier,
        "ranked_candidates": ranked,
        "mean_metrics": means,
        "retriever_winners": retriever_winners,
        "retriever_disagreement": len(set(retriever_winners.values())) > 1,
        "paired_comparison": {strategy: means[strategy] for strategy in ("B4e", "B4d")},
    }


def _round_robin(candidates: list[dict], count: int) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for candidate in candidates:
        grouped[candidate["category"]].append(candidate)
    for values in grouped.values():
        values.sort(key=lambda item: item["sort_key"])
    result: list[dict] = []
    categories = sorted(grouped)
    while len(result) < count:
        progressed = False
        for category in categories:
            if grouped[category] and len(result) < count:
                result.append(grouped[category].pop(0))
                progressed = True
        if not progressed:
            break
    if len(result) != count:
        raise ValueError(f"Review bucket has only {len(result)}/{count} eligible candidates")
    return result


def _top5(rows: list[dict]) -> list[dict]:
    return [{
        "rank": row["rank"],
        "passage_id": row["retrieved_passage_id"],
        "primary_node_id": row["retrieved_primary_node_id"],
        "score": row["score"],
        "exact_hit": row["exact_hit"],
        "structural_hit": row["structural_hit"],
        "included_node_ids": row["included_node_ids"],
        "evidence_tokens": row["evidence_tokens"],
    } for row in sorted(rows, key=lambda item: item["rank"])[:5]]


def _suggest_taxonomy(category: str, bucket: str) -> str:
    if bucket == "false_positive_top1":
        return "F9" if category in {"point_specific", "clause_intro_point"} else "F6"
    if bucket == "clear_win":
        return "F7" if category == "exact_reference" else "F2"
    return {
        "semantic_paraphrase": "F6", "clause_intro_point": "F2",
        "multi_evidence": "F8", "point_specific": "F1",
        "hard_distractor": "F3", "exact_reference": "F7",
    }.get(category, "F3")


def _load_run_artifacts(root: Path, split: str) -> dict[str, dict]:
    artifacts = {}
    metric_path = root / ROUND2_DIR / "round2_per_query_metrics.csv"
    with metric_path.open(encoding="utf-8-sig", newline="") as handle:
        metric_rows = list(csv.DictReader(handle))
    metric_map = {(row["strategy"], row["retriever"], row["query_id"]): row for row in metric_rows}
    for strategy in ROUND2_VARIANTS:
        for retriever in RETRIEVERS:
            run_id = f"{strategy}__{retriever.upper()}__v001__{split}"
            run_dir = root / "reports/chunk_ablation/runs" / run_id
            rows = read_jsonl(run_dir / "per_query.jsonl")
            query_rows = defaultdict(list)
            for row in rows:
                query_rows[row["query_id"]].append(row)
            artifacts[run_id] = {
                "summary": read_json(run_dir / "summary.json"),
                "query_rows": query_rows,
                "query_metrics": {
                    query_id: metric_map[(strategy, retriever, query_id)] for query_id in query_rows
                },
            }
    return artifacts


def _float_metric(row: dict, name: str) -> float:
    return float(row[name])


def _query_tuple(metrics: dict) -> tuple:
    rank = metrics["first_exact_rank"]
    reciprocal = 1 / int(rank) if rank not in (None, "", "None") else 0.0
    return (
        _float_metric(metrics, "exact_hit@5"), reciprocal,
        _float_metric(metrics, "evidence_coverage@5"),
        -_float_metric(metrics, "evidence_tokens@5"),
        -_float_metric(metrics, "duplicate_token_ratio@5"),
    )


def build_round2_review_queue(run_artifacts: dict[str, dict], root: str | Path | None = None) -> list[dict]:
    root_path = Path(root or ".").resolve()
    queries, gold_records = load_dataset(root_path)
    query_by_id = {item.query_id: item for item in queries if item.split == "dev"}
    gold_by_id = {item.query_id: item for item in gold_records}
    by_strategy: dict[str, dict[str, dict]] = defaultdict(dict)
    for run_id, artifact in run_artifacts.items():
        strategy, retriever = run_id.split("__")[:2]
        by_strategy[strategy][retriever.casefold()] = artifact
    combined_metrics: dict[tuple[str, str], tuple] = {}
    for strategy in ROUND2_VARIANTS:
        for query_id in query_by_id:
            tuples = [_query_tuple(by_strategy[strategy][retriever]["query_metrics"][query_id]) for retriever in RETRIEVERS]
            combined_metrics[(strategy, query_id)] = tuple(statistics.mean(values) for values in zip(*tuples))

    queue: list[dict] = []
    for strategy in ROUND2_VARIANTS:
        bm25 = by_strategy[strategy]["bm25"]
        dense = by_strategy[strategy]["dense"]
        candidates = []
        for query_id, query in query_by_id.items():
            bm = bm25["query_metrics"][query_id]
            de = dense["query_metrics"][query_id]
            bm_rows = sorted(bm25["query_rows"][query_id], key=lambda row: row["rank"])
            de_rows = sorted(dense["query_rows"][query_id], key=lambda row: row["rank"])
            exact_bm, exact_de = bool(float(bm["exact_hit@5"])), bool(float(de["exact_hit@5"]))
            structural_bm = any(row["structural_hit"] for row in bm_rows[:5])
            structural_de = any(row["structural_hit"] for row in de_rows[:5])
            recovered = any(
                not rows[0]["exact_hit"] and not rows[0]["structural_hit"]
                and any(row["exact_hit"] for row in rows[1:5])
                for rows in (bm_rows, de_rows)
            )
            beaten = sum(
                combined_metrics[(strategy, query_id)] > combined_metrics[(other, query_id)]
                for other in ROUND2_VARIANTS if other != strategy
            )
            candidates.append({
                "query_id": query_id, "category": query.category,
                "failure": not exact_bm and not exact_de,
                "no_structural": not structural_bm and not structural_de,
                "coverage": statistics.mean((float(bm["evidence_coverage@5"]), float(de["evidence_coverage@5"]))),
                "false_positive": recovered and (exact_bm or exact_de),
                "clear_win": (exact_bm or exact_de) and beaten > 0,
                "beaten": beaten,
                "best_rank": min(
                    [int(value) for value in (bm["first_exact_rank"], de["first_exact_rank"]) if value not in (None, "", "None")],
                    default=99,
                ),
            })
        failures = [{**item, "sort_key": (not item["no_structural"], item["coverage"], item["query_id"])} for item in candidates if item["failure"]]
        selected_failure = _round_robin(failures, 10)
        used = {item["query_id"] for item in selected_failure}
        false_positives = [
            {**item, "sort_key": (item["best_rank"], item["query_id"])}
            for item in candidates if item["false_positive"] and item["query_id"] not in used
        ]
        selected_false = _round_robin(false_positives, 5)
        used.update(item["query_id"] for item in selected_false)
        wins = [
            {**item, "sort_key": (-item["beaten"], item["best_rank"], -item["coverage"], item["query_id"])}
            for item in candidates if item["clear_win"] and item["query_id"] not in used
        ]
        selected_wins = _round_robin(wins, 5)
        for bucket, selected in (
            ("failure", selected_failure),
            ("false_positive_top1", selected_false),
            ("clear_win", selected_wins),
        ):
            for item in selected:
                query_id = item["query_id"]
                query = query_by_id[query_id]
                gold = gold_by_id[query_id]
                bm = bm25["query_metrics"][query_id]
                de = dense["query_metrics"][query_id]
                queue.append({
                    "review_id": f"{strategy}__{query_id}",
                    "strategy": strategy,
                    "query_id": query_id,
                    "category": query.category,
                    "bucket": bucket,
                    "query": query.query,
                    "gold_node_ids": gold.gold_node_ids,
                    "required_node_ids": gold.required_node_ids,
                    "bm25_top5": _top5(bm25["query_rows"][query_id]),
                    "dense_top5": _top5(dense["query_rows"][query_id]),
                    "bm25_first_exact_rank": bm["first_exact_rank"] or None,
                    "dense_first_exact_rank": de["first_exact_rank"] or None,
                    "bm25_evidence_coverage@5": float(bm["evidence_coverage@5"]),
                    "dense_evidence_coverage@5": float(de["evidence_coverage@5"]),
                    "bm25_duplicate_token_ratio@5": float(bm["duplicate_token_ratio@5"]),
                    "dense_duplicate_token_ratio@5": float(de["duplicate_token_ratio@5"]),
                    "suggested_taxonomy": _suggest_taxonomy(query.category, bucket),
                    "review_status": "draft",
                    "verdict": "",
                    "taxonomy_codes": [],
                    "reviewer": "",
                    "review_notes": "",
                })
    return queue


def _review_csv_row(record: dict) -> dict:
    result = dict(record)
    for key in ("gold_node_ids", "required_node_ids", "taxonomy_codes"):
        result[key] = "|".join(result[key]) if isinstance(result[key], list) else result[key]
    for key in ("bm25_top5", "dense_top5"):
        result[key] = json.dumps(result[key], ensure_ascii=False, separators=(",", ":"))
    return result


def export_round2_review(root: str | Path, *, split: str = "dev") -> dict:
    if split != "dev":
        raise ValueError("Round 2 manual review is restricted to dev")
    root_path = Path(root).resolve()
    report_root, _ = _round2_paths(root_path)
    artifacts = _load_run_artifacts(root_path, split)
    queue = build_round2_review_queue(artifacts, root_path)
    csv_rows = [_review_csv_row(record) for record in queue]
    _write_csv(report_root / "round2_manual_review.csv", csv_rows)
    write_jsonl(report_root / "round2_manual_review.jsonl", queue)
    errors = validate_round2_review(report_root / "round2_manual_review.csv", require_complete=False)
    if errors:
        raise ValueError("Generated review queue is invalid: " + "; ".join(errors))
    completion_errors = validate_round2_review(
        report_root / "round2_manual_review.csv", require_complete=True
    )
    selection = select_round2_candidate(_load_round2_summaries(root_path, split))
    _write_decision_report(
        root_path, selection,
        ["manual review requires 100/100 approved records with verdict and reviewer; current approved=0"],
    )
    _write_failure_report(root_path, queue, review_complete=False)
    return {"record_count": len(queue), "csv": str(report_root / "round2_manual_review.csv")}


def _parse_review_csv(path: str | Path) -> list[dict]:
    with Path(path).resolve().open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def validate_round2_review(review_csv: str | Path, *, require_complete: bool = False) -> list[str]:
    rows = _parse_review_csv(review_csv)
    errors: list[str] = []
    if len(rows) != 100:
        errors.append(f"expected 100 records, got {len(rows)}")
    review_ids = [row.get("review_id", "") for row in rows]
    if len(review_ids) != len(set(review_ids)):
        errors.append("duplicate review_id")
    pairs = [(row.get("strategy"), row.get("query_id")) for row in rows]
    if len(pairs) != len(set(pairs)):
        errors.append("duplicate strategy/query_id")
    expected_buckets = {"failure": 10, "false_positive_top1": 5, "clear_win": 5}
    for strategy in ROUND2_VARIANTS:
        strategy_rows = [row for row in rows if row.get("strategy") == strategy]
        if len(strategy_rows) != 20:
            errors.append(f"{strategy}: expected 20 records, got {len(strategy_rows)}")
        counts = Counter(row.get("bucket") for row in strategy_rows)
        if counts != Counter(expected_buckets):
            errors.append(f"{strategy}: invalid bucket quota {dict(counts)}")
    unknown_strategies = sorted({row.get("strategy", "") for row in rows} - set(ROUND2_VARIANTS))
    if unknown_strategies:
        errors.append(f"unknown strategies: {unknown_strategies}")
    for index, row in enumerate(rows, start=2):
        if row.get("review_id") != f"{row.get('strategy')}__{row.get('query_id')}":
            errors.append(f"row {index}: review_id does not match strategy/query_id")
        if not row.get("gold_node_ids", "").strip() or not row.get("required_node_ids", "").strip():
            errors.append(f"row {index}: gold/required IDs must not be empty")
        if row.get("suggested_taxonomy") not in TAXONOMY_CODES:
            errors.append(f"row {index}: invalid suggested_taxonomy")
        for field in ("bm25_top5", "dense_top5"):
            try:
                top_rows = json.loads(row.get(field, ""))
                if len(top_rows) != 5 or [item["rank"] for item in top_rows] != [1, 2, 3, 4, 5]:
                    errors.append(f"row {index}: {field} must contain ranks 1..5")
            except (json.JSONDecodeError, KeyError, TypeError):
                errors.append(f"row {index}: invalid {field} JSON")
        for field in ("bm25_duplicate_token_ratio@5", "dense_duplicate_token_ratio@5"):
            try:
                value = float(row.get(field, ""))
                if not 0 <= value <= 1:
                    errors.append(f"row {index}: {field} outside [0,1]")
            except ValueError:
                errors.append(f"row {index}: invalid {field}")
        status = row.get("review_status", "").strip().casefold()
        verdict = row.get("verdict", "").strip().casefold()
        codes = {value for value in row.get("taxonomy_codes", "").split("|") if value}
        if status not in REVIEW_STATUSES:
            errors.append(f"row {index}: invalid review_status {status!r}")
        if verdict and verdict not in VERDICTS:
            errors.append(f"row {index}: invalid verdict {verdict!r}")
        if not codes.issubset(TAXONOMY_CODES):
            errors.append(f"row {index}: invalid taxonomy codes {sorted(codes - TAXONOMY_CODES)}")
        if require_complete:
            if status != "approved":
                errors.append(f"row {index}: review is not approved")
            if not verdict:
                errors.append(f"row {index}: verdict is required")
            if verdict == "benchmark_issue":
                errors.append(f"row {index}: benchmark_issue invalidates finalization")
            if not row.get("reviewer", "").strip():
                errors.append(f"row {index}: reviewer is required")
    return errors


def apply_round2_review(root: str | Path, review_csv: str | Path) -> dict:
    root_path = Path(root).resolve()
    report_root, _ = _round2_paths(root_path)
    source_rows = _parse_review_csv(review_csv)
    canonical_path = report_root / "round2_manual_review.csv"
    if not canonical_path.exists():
        raise FileNotFoundError("Export the Round 2 review queue before applying a review")
    original_rows = _parse_review_csv(canonical_path)
    original = {row["review_id"]: row for row in original_rows}
    if {row.get("review_id") for row in source_rows} != set(original):
        raise ValueError("Reviewed CSV must contain the exact exported review_id set")
    immutable = {
        "strategy", "query_id", "category", "bucket", "query", "gold_node_ids", "required_node_ids",
        "bm25_top5", "dense_top5", "bm25_first_exact_rank", "dense_first_exact_rank",
        "bm25_evidence_coverage@5", "dense_evidence_coverage@5",
        "bm25_duplicate_token_ratio@5", "dense_duplicate_token_ratio@5", "suggested_taxonomy",
    }
    for row in source_rows:
        baseline = original[row["review_id"]]
        changed = [field for field in immutable if row.get(field, "") != baseline.get(field, "")]
        if changed:
            raise ValueError(f"{row['review_id']}: immutable review fields changed: {changed}")
    temporary = report_root / ".round2_review_validating.csv"
    _write_csv(temporary, source_rows)
    try:
        errors = validate_round2_review(temporary, require_complete=False)
    finally:
        temporary.unlink(missing_ok=True)
    if errors:
        raise ValueError("Round 2 review invalid: " + "; ".join(errors[:20]))
    _write_csv(canonical_path, source_rows)
    jsonl_rows = []
    for row in source_rows:
        value = dict(row)
        for key in ("gold_node_ids", "required_node_ids", "taxonomy_codes"):
            value[key] = [item for item in value[key].split("|") if item]
        for key in ("bm25_top5", "dense_top5"):
            value[key] = json.loads(value[key])
        jsonl_rows.append(value)
    write_jsonl(report_root / "round2_manual_review.jsonl", jsonl_rows)
    complete_errors = validate_round2_review(canonical_path, require_complete=True)
    selection = select_round2_candidate(_load_round2_summaries(root_path, "dev"))
    approved_count = sum(row["review_status"].strip().casefold() == "approved" for row in source_rows)
    report_errors = [] if not complete_errors else [
        f"manual review requires 100/100 approved records with verdict and reviewer; current approved={approved_count}"
    ]
    _write_decision_report(root_path, selection, report_errors)
    _write_failure_report(root_path, jsonl_rows, review_complete=not complete_errors)
    return {
        "record_count": len(source_rows),
        "approved_count": approved_count,
        "ready_to_finalize": not complete_errors,
        "finalization_errors": complete_errors,
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    fields = list(rows[0])
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_failure_report(root: Path, review_rows: list[dict], *, review_complete: bool) -> None:
    output = root / ROUND2_DIR / "round2_failure_analysis.md"
    bucket_counts = Counter(row.get("bucket") for row in review_rows)
    verdict_counts = Counter(row.get("verdict") for row in review_rows if row.get("verdict"))
    taxonomy = Counter(
        code for row in review_rows
        for code in (row.get("taxonomy_codes", []) if isinstance(row.get("taxonomy_codes"), list) else row.get("taxonomy_codes", "").split("|"))
        if code
    )
    suggested = Counter(row.get("suggested_taxonomy") for row in review_rows if row.get("suggested_taxonomy"))
    lines = [
        "# Round 2 Failure Analysis", "",
        f"Manual review complete: `{review_complete}`", "",
        f"Queue records: `{len(review_rows)}`; buckets: `{dict(bucket_counts)}`.", "",
        f"Reviewed verdicts: `{dict(verdict_counts)}`.", "",
        f"Reviewed taxonomy: `{dict(taxonomy)}`.", "",
        f"Suggested taxonomy in deterministic queue: `{dict(suggested)}`.", "",
        "Taxonomy: F1 short leaf; F2 missing parent; F3 dilution; F4 broad article; "
        "F5 repeated parent; F6 paraphrase; F7 exact reference; F8 incomplete evidence; "
        "F9 citation granularity; F10 window boundary.", "",
        "A `benchmark_issue` verdict invalidates affected runs; a legitimate retrieval failure is retained as analysis evidence.",
    ]
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _load_round2_summaries(root: Path, split: str) -> list[dict]:
    summaries = []
    for strategy in ROUND2_VARIANTS:
        for retriever in RETRIEVERS:
            path = root / "reports/chunk_ablation/runs" / f"{strategy}__{retriever.upper()}__v001__{split}" / "summary.json"
            summary = read_json(path)
            if summary.get("metrics_schema_version") != METRICS_SCHEMA_VERSION:
                raise ValueError(f"Run has not been migrated to {METRICS_SCHEMA_VERSION}: {path.parent.name}")
            summaries.append(summary)
    return summaries


def _write_decision_report(root: Path, selection: dict, review_errors: list[str]) -> None:
    review_ready = not review_errors
    means = selection["mean_metrics"]
    winner, challenger = selection["winner"], selection["challenger"]
    lines = [
        "# Round 2 Chunking Decision", "",
        f"Status: **{'REVIEW COMPLETE — AWAITING USER PRODUCTION-FREEZE APPROVAL' if review_ready else 'DEV CANDIDATE — MANUAL REVIEW REQUIRED'}**", "",
        f"Metrics schema: `{METRICS_SCHEMA_VERSION}`.", "",
        "Corrected redundancy formula: `T = all component occurrences`, `U = unique token-sequence hashes`, "
        "`duplicate_token_ratio@K = (T-U)/T`. Article/Document context contributes heading/title only.", "",
        "BM25 and Dense have equal weight. Strict selection is Recall@5 → MRR → EvidenceCoverage@5 → "
        "evidence-token cost → duplicate-token ratio → strategy ID.", "",
        "| Variant | Recall@5 | MRR | EvidenceCoverage@5 | Evidence tokens@5 | Duplicate ratio@5 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for strategy in ROUND2_VARIANTS:
        metric = means[strategy]
        lines.append(
            f"| {strategy} | {metric['recall@5']:.4f} | {metric['mrr']:.4f} | "
            f"{metric['evidence_coverage@5']:.4f} | {metric['avg_evidence_tokens@5']:.1f} | "
            f"{metric['duplicate_token_ratio@5']:.4f} |"
        )
    lines += [
        "", f"Winner: **{winner}**.", "", f"Pareto challenger: **{challenger}**.", "",
        f"Pareto frontier: `{', '.join(selection['pareto_frontier'])}`.", "",
        "## B4e / B4d paired comparison", "",
        "| Variant | Recall@5 | MRR | Coverage@5 | Cost@5 | Redundancy@5 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for strategy in ("B4e", "B4d"):
        metric = selection["paired_comparison"][strategy]
        lines.append(
            f"| {strategy} | {metric['recall@5']:.4f} | {metric['mrr']:.4f} | "
            f"{metric['evidence_coverage@5']:.4f} | {metric['avg_evidence_tokens@5']:.1f} | "
            f"{metric['duplicate_token_ratio@5']:.4f} |"
        )
    lines += [
        "", "## Retriever agreement", "",
        f"BM25 winner: **{selection['retriever_winners']['bm25']}**.  ",
        f"Dense winner: **{selection['retriever_winners']['dense']}**.  ",
        f"Disagreement: `{selection['retriever_disagreement']}`.", "",
        "## Manual review gate", "",
        f"Ready: `{review_ready}`.",
    ]
    if review_errors:
        lines += ["", *[f"- {error}" for error in review_errors[:10]]]
    lines += [
        "", "Known failure/win samples and taxonomy are recorded in `round2_failure_analysis.md`.", "",
        "Held-out test has not been opened. This report does not lock a production strategy.", "",
        "Citation policy: use canonical IDs from `citation_node_ids`; never cite a retrieval passage ID.",
    ]
    (root / ROUND2_DIR / "round2_decision.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def finalize_round2(root: str | Path, *, split: str = "dev") -> dict:
    if split != "dev":
        raise ValueError("Round 2 finalization is restricted to dev")
    root_path = Path(root).resolve()
    review_path = root_path / ROUND2_DIR / "round2_manual_review.csv"
    errors = validate_round2_review(review_path, require_complete=True)
    if errors:
        summaries = _load_round2_summaries(root_path, split)
        selection = select_round2_candidate(summaries)
        _write_decision_report(root_path, selection, errors)
        raise ValueError("Round 2 review gate failed: " + "; ".join(errors[:20]))
    summaries = _load_round2_summaries(root_path, split)
    selection = select_round2_candidate(summaries)
    _write_decision_report(root_path, selection, [])
    review_rows = _parse_review_csv(review_path)
    _write_failure_report(root_path, review_rows, review_complete=True)
    return {**selection, "review_complete": True, "production_frozen": False, "test_opened": False}
