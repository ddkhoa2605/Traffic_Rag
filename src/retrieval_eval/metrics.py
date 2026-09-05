from __future__ import annotations

import math
import statistics
from collections import Counter


TOP_K = (1, 3, 5, 10)


def aggregate_metrics(query_results: list[dict]) -> dict[str, float]:
    if not query_results:
        return {}
    metrics: dict[str, float] = {}
    count = len(query_results)
    reciprocal_ranks = []
    for item in query_results:
        first = next((row["rank"] for row in item["rows"] if row["exact_hit"]), None)
        reciprocal_ranks.append(1 / first if first else 0.0)
    metrics["mrr"] = sum(reciprocal_ranks) / count
    for k in TOP_K:
        recalls = []
        structural = []
        evidence = []
        index_tokens = []
        evidence_tokens = []
        densities = []
        duplicate_nodes = []
        duplicate_context = []
        duplicate_tokens = []
        duplicate_token_counts = []
        total_component_tokens = []
        unique_coverage = []
        for item in query_results:
            rows = item["rows"][:k]
            recalls.append(float(any(row["exact_hit"] for row in rows)))
            covered_gold = set().union(*(set(row["structurally_covered_gold_ids"]) for row in rows)) if rows else set()
            structural.append(len(covered_gold & set(item["gold_node_ids"])) / len(set(item["gold_node_ids"])))
            included = [node_id for row in rows for node_id in row["included_node_ids"]]
            included_set = set(included)
            required = set(item["required_node_ids"])
            evidence.append(len(required & included_set) / len(required))
            total_index = sum(row["index_tokens"] for row in rows)
            total_evidence = sum(row["evidence_tokens"] for row in rows)
            index_tokens.append(total_index)
            evidence_tokens.append(total_evidence)
            relevant_tokens = min(total_evidence, sum(item["required_token_counts"].get(node_id, 0) for node_id in required & included_set))
            densities.append(relevant_tokens / total_evidence if total_evidence else 0.0)
            duplicate_nodes.append(1 - len(included_set) / len(included) if included else 0.0)
            contexts = [node_id for row in rows for node_id in row["context_node_ids"]]
            duplicate_context.append(1 - len(set(contexts)) / len(contexts) if contexts else 0.0)
            components = [component for row in rows for component in row.get("evidence_components", [])]
            if components:
                component_total = sum(component["token_count"] for component in components)
                unique: dict[str, int] = {}
                for component in components:
                    unique.setdefault(component["token_sequence_hash"], component["token_count"])
                repeated_tokens = component_total - sum(unique.values())
                ratio = repeated_tokens / component_total if component_total else 0.0
                if not 0 <= ratio <= 1:
                    raise ValueError(f"Invalid component duplicate-token ratio: {ratio}")
            else:
                counts = Counter(included)
                duplicated_node_ids = {node_id for node_id, occurrences in counts.items() if occurrences > 1}
                repeated_tokens = sum(item["all_node_token_counts"].get(node_id, 0) * (counts[node_id] - 1) for node_id in duplicated_node_ids)
                component_total = total_evidence
                ratio = min(repeated_tokens / total_evidence, 1.0) if total_evidence else 0.0
            duplicate_tokens.append(ratio)
            duplicate_token_counts.append(repeated_tokens)
            total_component_tokens.append(component_total)
            unique_coverage.append(float(len(included_set)))
        metrics[f"recall@{k}"] = sum(recalls) / count
        metrics[f"structural_coverage@{k}"] = sum(structural) / count
        metrics[f"evidence_coverage@{k}"] = sum(evidence) / count
        metrics[f"avg_index_tokens@{k}"] = sum(index_tokens) / count
        metrics[f"avg_evidence_tokens@{k}"] = sum(evidence_tokens) / count
        metrics[f"relevant_token_density@{k}"] = sum(densities) / count
        metrics[f"duplicate_node_ratio@{k}"] = sum(duplicate_nodes) / count
        metrics[f"duplicate_context_ratio@{k}"] = sum(duplicate_context) / count
        metrics[f"duplicate_token_ratio@{k}"] = sum(duplicate_tokens) / count
        metrics[f"duplicate_token_ratio_median@{k}"] = statistics.median(duplicate_tokens)
        ordered_duplicate = sorted(duplicate_tokens)
        p95_index = max(0, math.ceil(.95 * len(ordered_duplicate)) - 1)
        metrics[f"duplicate_token_ratio_p95@{k}"] = ordered_duplicate[p95_index]
        metrics[f"avg_duplicate_token_count@{k}"] = sum(duplicate_token_counts) / count
        metrics[f"avg_total_component_tokens@{k}"] = sum(total_component_tokens) / count
        metrics[f"unique_source_node_coverage@{k}"] = sum(unique_coverage) / count
    return {key: round(value, 6) for key, value in metrics.items()}
