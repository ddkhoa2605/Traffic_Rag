from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import platform
from collections import defaultdict
from pathlib import Path

import yaml

from src.chunking.builder import OUTPUT_NAMES, _code_digest, load_chunk_config
from src.chunking.child_parent import expand_evidence
from src.chunking.models import EvidenceBundle, ExpansionPolicy, RetrievalPassage
from src.chunking.token_counter import BGETokenCounter, TokenCounter
from src.legal_tree.loader import load_legal_document
from src.legal_tree.resolver import LegalTreeResolver
from src.parser.io import read_jsonl, write_json, write_jsonl

from .bm25 import BM25Retriever
from .dataset import load_dataset, validate_dataset
from .dense import DenseEncoder, DenseRetriever
from .hybrid import DualGranularityRetriever, HybridConfig
from .metrics import aggregate_metrics
from .models import EvidenceComponent, PerRankResult, RunSummary
from .round2 import PassageComponent, project_b4_components


def _runtime_versions() -> dict[str, str]:
    result = {"python": platform.python_version()}
    for package in ("numpy", "pydantic", "sentence-transformers", "transformers", "torch", "huggingface-hub"):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            continue
    return result


def _ranking_digest(rows: list[PerRankResult]) -> str:
    digest = hashlib.sha256()
    for row in sorted(rows, key=lambda value: (value.query_id, value.rank)):
        value = {
            "query_id": row.query_id, "rank": row.rank,
            "retrieved_passage_id": row.retrieved_passage_id,
            "retrieved_primary_node_id": row.retrieved_primary_node_id, "score": row.score,
        }
        digest.update(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def _resolve_under_root(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _load_passages(
    root: Path,
    strategy_id: str,
    artifact_root: str | Path = "data/07_retrieval_ablation",
) -> tuple[list[RetrievalPassage], dict]:
    folder = _resolve_under_root(root, artifact_root) / OUTPUT_NAMES[strategy_id]
    passages = [RetrievalPassage.model_validate(item) for item in read_jsonl(folder / "passages.jsonl")]
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    return passages, manifest


def _hybrid_passages(
    root: Path,
    hybrid_config: dict,
    artifact_root: str | Path = "data/07_retrieval_ablation",
) -> tuple[list[RetrievalPassage], list[RetrievalPassage], dict]:
    article_id = hybrid_config["article_strategy"]
    fine_id = hybrid_config["fine_strategy"]
    if (article_id, fine_id) != ("B1", "B4e"):
        raise ValueError("B6 v0.1 is locked to B1 + B4e")
    article, article_manifest = _load_passages(root, article_id, artifact_root)
    fine, fine_manifest = _load_passages(root, fine_id, artifact_root)
    if article_manifest["documents"] != fine_manifest["documents"]:
        raise ValueError("B6 source manifests use different document sets")
    if article_manifest["document_digests"] != fine_manifest["document_digests"]:
        raise ValueError("B6 source manifests use different canonical corpora")
    if article_manifest["tokenizer_revision"] != fine_manifest["tokenizer_revision"]:
        raise ValueError("B6 source manifests use different tokenizer revisions")
    digest_payload = {
        "strategy": "B6_dual_granularity",
        "article_passage_digest": article_manifest["passage_digest"],
        "fine_passage_digest": fine_manifest["passage_digest"],
        "config": hybrid_config,
    }
    manifest = {
        "strategy": "B6_dual_granularity",
        "documents": article_manifest["documents"],
        "document_digests": article_manifest["document_digests"],
        "tokenizer_revision": article_manifest["tokenizer_revision"],
        "passage_digest": hashlib.sha256(
            json.dumps(digest_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "article_manifest_hash": article_manifest["passage_digest"],
        "fine_manifest_hash": fine_manifest["passage_digest"],
        "hybrid_config": hybrid_config,
        "git_commit": None,
    }
    return article, fine, manifest


def _component_model(component: PassageComponent) -> EvidenceComponent:
    return EvidenceComponent(
        node_id=component.node_id,
        role=component.role,
        token_sequence_hash=component.token_sequence_hash,
        token_count=component.token_count,
    )


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def build_article_scout_bundle(
    article_passage: RetrievalPassage,
    evidence_passage_ids: list[str],
    fine_by_id: dict[str, RetrievalPassage],
    resolvers: dict[str, LegalTreeResolver],
    counter: TokenCounter,
) -> tuple[EvidenceBundle, list[EvidenceComponent]]:
    if not evidence_passage_ids:
        raise ValueError(f"B6 Article scout has no B4e fallback: {article_passage.primary_node_id}")
    projected: list[PassageComponent] = []
    citations = [article_passage.primary_node_id]
    for passage_id in evidence_passage_ids:
        passage = fine_by_id.get(passage_id)
        if passage is None or passage.document_id != article_passage.document_id:
            raise ValueError(f"Invalid B6 evidence passage for {article_passage.passage_id}: {passage_id}")
        article = resolvers[passage.document_id].get_article(passage.primary_node_id)
        if article is None or article.id != article_passage.primary_node_id:
            raise ValueError(f"B6 evidence crosses Article boundary: {passage_id}")
        projected.extend(project_b4_components(passage, resolvers[passage.document_id], counter))
        citations.extend(passage.citation_node_ids)
    unique_components: list[PassageComponent] = []
    seen_hashes: set[str] = set()
    for component in projected:
        if component.token_sequence_hash not in seen_hashes:
            seen_hashes.add(component.token_sequence_hash)
            unique_components.append(component)
    evidence_text = "\n".join(component.text for component in unique_components).strip()
    token_count = counter.count(evidence_text)
    if not evidence_text or token_count <= 0:
        raise ValueError(f"Empty B6 Article evidence bundle: {article_passage.passage_id}")
    included = _unique([component.node_id for component in unique_components])
    contexts = _unique([
        component.node_id for component in unique_components
        if component.role in {"document_title", "article_heading", "clause_intro"}
    ])
    return EvidenceBundle(
        primary_node_id=article_passage.primary_node_id,
        included_node_ids=included,
        context_node_ids=contexts,
        citation_node_ids=_unique(citations),
        evidence_text=evidence_text,
        token_count=token_count,
    ), [_component_model(component) for component in unique_components]


# Backward-compatible private name used by the frozen evaluation tests.
_article_scout_bundle = build_article_scout_bundle


def _structural_gold(
    passage: RetrievalPassage,
    gold_ids: list[str],
    acceptable_parent_ids: list[str],
    resolvers: dict[str, LegalTreeResolver],
) -> set[str]:
    resolver = resolvers[passage.document_id]
    covered: set[str] = set()
    for gold_id in gold_ids:
        try:
            if passage.primary_node_id == gold_id:
                covered.add(gold_id)
            elif resolver.is_ancestor(passage.primary_node_id, gold_id) or resolver.is_ancestor(gold_id, passage.primary_node_id):
                covered.add(gold_id)
            elif passage.strategy == "B0_fixed_window" and gold_id in passage.source_node_ids:
                covered.add(gold_id)
        except KeyError:
            continue
    if passage.primary_node_id in acceptable_parent_ids:
        covered.update(gold_ids)
    return covered


def evaluate_run(
    root: str | Path,
    strategy_id: str,
    retriever_name: str,
    *,
    split: str = "dev",
    official: bool = False,
    token_counter: TokenCounter | None = None,
    dense_encoder: DenseEncoder | None = None,
    artifact_root: str | Path = "data/07_retrieval_ablation",
    report_root: str | Path = "reports/chunk_ablation/runs",
    run_version: str = "v001",
) -> RunSummary:
    root_path = Path(root).resolve()
    errors = validate_dataset(root_path, require_approved=official)
    if errors:
        raise ValueError("Benchmark dataset invalid: " + "; ".join(errors[:20]))
    queries, gold = load_dataset(root_path)
    selected_queries = [item for item in queries if item.split == split]
    gold_by_id = {item.query_id: item for item in gold}
    with (root_path / "configs/retrieval.yaml").open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    hybrid_config = config.get("hybrid_b6", {})
    article_passages: list[RetrievalPassage] = []
    fine_passages: list[RetrievalPassage] = []
    if strategy_id == "B6":
        article_passages, fine_passages, passage_manifest = _hybrid_passages(
            root_path, hybrid_config, artifact_root
        )
        passages = [*article_passages, *fine_passages]
    else:
        passages, passage_manifest = _load_passages(root_path, strategy_id, artifact_root)
    passage_by_id = {item.passage_id: item for item in passages}
    fine_by_id = {item.passage_id: item for item in fine_passages}
    chunk_config = load_chunk_config(root_path)
    expansion_config = chunk_config["strategies"].get("B5", {}).get("expansion", {})
    counter = token_counter or BGETokenCounter(passage_manifest["tokenizer_revision"])
    resolvers = {
        document_id: LegalTreeResolver(load_legal_document(root_path, document_id).nodes)
        for document_id in passage_manifest["documents"]
    }
    all_nodes = {node.id: node for resolver in resolvers.values() for node in resolver.nodes}
    all_node_token_counts = {node_id: counter.count(node.text) for node_id, node in all_nodes.items() if node.text}

    if retriever_name == "bm25":
        base_config = config["bm25"]
        if strategy_id == "B6":
            retriever = DualGranularityRetriever(
                BM25Retriever(article_passages, k1=float(base_config["k1"]), b=float(base_config["b"])),
                BM25Retriever(fine_passages, k1=float(base_config["k1"]), b=float(base_config["b"])),
                article_passages, fine_passages, resolvers, HybridConfig.from_dict(hybrid_config),
            )
            retriever_config = {**base_config, "hybrid_b6": hybrid_config}
        else:
            retriever_config = base_config
            retriever = BM25Retriever(passages, k1=float(retriever_config["k1"]), b=float(retriever_config["b"]))
    elif retriever_name == "dense":
        dense_config = config["dense"]
        encoder = dense_encoder or DenseEncoder(
            dense_config["model"], dense_config["revision"],
            max_length=int(dense_config["max_length"]),
            preferred_device=dense_config["preferred_device"],
        )
        encoder.preflight(max((item.index_text for item in passages), key=len))
        dense_batch_size = int(
            dense_config.get("cpu_passage_batch_size", dense_config["passage_batch_size"])
            if encoder.device == "cpu" else dense_config["passage_batch_size"]
        )
        if strategy_id == "B6":
            article_retriever = DenseRetriever(
                article_passages, encoder,
                cache_dir=_resolve_under_root(root_path, artifact_root) / OUTPUT_NAMES["B1"] / "embeddings",
                passage_manifest_hash=passage_manifest["article_manifest_hash"], batch_size=dense_batch_size,
            )
            fine_retriever = DenseRetriever(
                fine_passages, encoder,
                cache_dir=_resolve_under_root(root_path, artifact_root) / OUTPUT_NAMES["B4e"] / "embeddings",
                passage_manifest_hash=passage_manifest["fine_manifest_hash"], batch_size=dense_batch_size,
            )
            retriever = DualGranularityRetriever(
                article_retriever, fine_retriever, article_passages, fine_passages,
                resolvers, HybridConfig.from_dict(hybrid_config),
            )
        else:
            retriever = DenseRetriever(
                passages, encoder,
                cache_dir=_resolve_under_root(root_path, artifact_root) / OUTPUT_NAMES[strategy_id] / "embeddings",
                passage_manifest_hash=passage_manifest["passage_digest"], batch_size=dense_batch_size,
            )
        retriever_config = {**dense_config, "resolved_device": encoder.device, "resolved_dtype": encoder.dtype, "resolved_revision": encoder.revision}
        if strategy_id == "B6":
            retriever_config["hybrid_b6"] = hybrid_config
    else:
        raise ValueError("retriever must be bm25 or dense")
    retriever_config = {**retriever_config, "package_versions": _runtime_versions()}

    if not run_version.startswith("v") or not run_version[1:].isdigit():
        raise ValueError("run_version must look like v002")
    run_id = f"{strategy_id}__{retriever_name.upper()}__{run_version}__{split}"
    output = _resolve_under_root(root_path, report_root) / run_id
    if official and (output / "summary.json").exists():
        raise FileExistsError(f"Official run already exists and cannot be overwritten: {output}")
    rows: list[PerRankResult] = []
    metric_inputs: list[dict] = []
    if retriever_name == "dense":
        hit_lists = retriever.search_many(
            [item.query for item in selected_queries], top_k=10,
            batch_size=int(config["dense"]["query_batch_size"]),
        )
    else:
        hit_lists = [retriever.search(item.query, top_k=10) for item in selected_queries]
    for query, hits in zip(selected_queries, hit_lists):
        gold_item = gold_by_id[query.query_id]
        internal_rows = []
        for hit in hits:
            passage = passage_by_id[hit.passage_id]
            evidence_components: list[EvidenceComponent] = []
            if strategy_id == "B6" and hit.source_strategy == "B1":
                bundle, evidence_components = build_article_scout_bundle(
                    passage, hit.evidence_passage_ids, fine_by_id, resolvers, counter
                )
            elif passage.strategy == "B5_child_parent":
                bundle = expand_evidence(
                    passage.primary_node_id,
                    ExpansionPolicy(**expansion_config),
                    resolvers[passage.document_id], counter,
                )
            else:
                bundle = EvidenceBundle(
                    primary_node_id=passage.primary_node_id,
                    included_node_ids=passage.source_node_ids,
                    context_node_ids=passage.context_node_ids,
                    citation_node_ids=passage.citation_node_ids,
                    evidence_text=passage.evidence_text,
                    token_count=passage.token_count_evidence,
                )
                if strategy_id == "B6" or strategy_id in {"B4a", "B4b", "B4c", "B4d", "B4e"}:
                    evidence_components = [
                        _component_model(component)
                        for component in project_b4_components(
                            passage, resolvers[passage.document_id], counter
                        )
                    ]
            covered = _structural_gold(passage, gold_item.gold_node_ids, gold_item.acceptable_parent_ids, resolvers)
            exact = passage.primary_node_id in gold_item.gold_node_ids
            row = PerRankResult(
                run_id=run_id, query_id=query.query_id, split=query.split,
                category=query.category, rank=hit.rank,
                retrieved_passage_id=passage.passage_id,
                retrieved_primary_node_id=passage.primary_node_id,
                score=hit.score, exact_hit=exact, structural_hit=bool(covered),
                index_tokens=passage.token_count_index,
                evidence_tokens=bundle.token_count,
                included_node_ids=bundle.included_node_ids,
                context_node_ids=bundle.context_node_ids,
                citation_node_ids=bundle.citation_node_ids,
                structurally_covered_gold_ids=sorted(covered),
                evidence_components=evidence_components,
            )
            rows.append(row)
            internal_rows.append(row.model_dump(mode="json"))
        metric_inputs.append({
            "query_id": query.query_id,
            "category": query.category,
            "gold_node_ids": gold_item.gold_node_ids,
            "required_node_ids": gold_item.required_node_ids,
            "required_token_counts": {node_id: all_node_token_counts.get(node_id, 0) for node_id in gold_item.required_node_ids},
            "all_node_token_counts": all_node_token_counts,
            "rows": internal_rows,
        })
    metrics = aggregate_metrics(metric_inputs)
    category_metrics = {
        category: aggregate_metrics([item for item in metric_inputs if item["category"] == category])
        for category in sorted({item["category"] for item in metric_inputs})
    }
    benchmark_manifest = json.loads((root_path / "data/08_eval/benchmark_manifest.json").read_text(encoding="utf-8"))
    summary = RunSummary(
        run_id=run_id, strategy=passage_manifest["strategy"], retriever=retriever_name,
        split=split, provisional=not official, query_count=len(selected_queries),
        metrics=metrics, category_metrics=category_metrics,
        retriever_config=retriever_config,
        corpus_digests=passage_manifest["document_digests"],
        passage_manifest_hash=passage_manifest["passage_digest"],
        dataset_hash=benchmark_manifest["dataset_digest"],
        code_digest=_code_digest(root_path), git_commit=passage_manifest.get("git_commit"),
        metrics_schema_version="0.2.0"
        if strategy_id == "B6" or strategy_id in {"B4a", "B4b", "B4c", "B4d", "B4e"}
        else "0.1.0",
        ranking_digest=_ranking_digest(rows),
    )
    if strategy_id == "B6":
        write_json(output / "hybrid_manifest.json", {
            **passage_manifest,
            "dataset_digest": benchmark_manifest["dataset_digest"],
            "retriever": retriever_name,
            "retriever_config": retriever_config,
            "code_digest": summary.code_digest,
        })
    write_json(output / "summary.json", summary.model_dump(mode="json"))
    write_jsonl(output / "per_query.jsonl", (row.model_dump(mode="json") for row in rows))
    output.mkdir(parents=True, exist_ok=True)
    with (output / "per_query.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].model_dump(mode="json").keys()))
        writer.writeheader()
        for row in rows:
            value = row.model_dump(mode="json")
            for key in ("included_node_ids", "context_node_ids", "citation_node_ids", "structurally_covered_gold_ids"):
                value[key] = "|".join(value[key])
            value["evidence_components"] = json.dumps(value["evidence_components"], ensure_ascii=False, separators=(",", ":"))
            writer.writerow(value)
    return summary
