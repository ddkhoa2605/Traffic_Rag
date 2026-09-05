from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

from src.chunking.token_counter import BGETokenCounter
from src.graph.config import GraphSettings, graph_config_hash, load_graph_config
from src.graph.identity import normalized_label
from src.graph.migrations import init_database, migration_checksums
from src.graph.mirror import build_lightrag_payload
from src.graph.models import GraphEdge, GraphNode, GraphRelease, GraphSourceDocument
from src.graph.ops import backup_graph, restore_graph_check
from src.graph.pilot import (
    load_sources,
    run_full_extraction,
    run_pilot,
    run_pilot_quota_recovery,
)
from src.graph.preflight import graph_code_digest, run_preflight
from src.graph.projection import canonical_digest, export_articles
from src.graph.providers import provider_from_name
from src.graph.repository import (
    connect_graph,
    store_graph,
    store_mirror_map,
    store_sources,
    rebuild_approved_aliases,
    reset_building_release,
    transition_release,
    upsert_release,
    validate_database,
)
from src.graph.review import (
    export_review,
    load_and_validate_review,
    review_metrics,
    select_provider,
)
from src.graph.structural import build_structural_graph
from src.graph.validation import validate_article_sources, validate_graph
from src.legal_tree.loader import load_legal_document
from src.parser.io import read_json, read_jsonl, write_json, write_jsonl
from src.registry.loader import load_registry


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Directed legal graph and LightRAG mirror")
    parser.add_argument("--root", default=None)
    parser.add_argument("--release-id", default=None)
    commands = parser.add_subparsers(dest="command", required=True)

    preflight = commands.add_parser("preflight")
    preflight.add_argument("--offline", action="store_true")
    commands.add_parser("db-init")
    commands.add_parser("export-articles")
    structural = commands.add_parser("build-structural")
    structural.add_argument("--no-references", action="store_true")
    structural.add_argument(
        "--no-persist",
        action="store_true",
        help="Build and validate deterministic artifacts without connecting to PostgreSQL",
    )

    pilot = commands.add_parser("pilot-extract")
    pilot.add_argument("--provider", choices=("openai", "gemini"), required=True)
    pilot.add_argument("--allow-paid-api", action="store_true")
    pilot.add_argument(
        "--resume-quota",
        action="store_true",
        help="Retry only audited 429/RESOURCE_EXHAUSTED pilot failures",
    )
    pilot.add_argument("--recovery-id", default="quota-v001")

    review = commands.add_parser("export-review")
    review.add_argument("--provider", choices=("openai", "gemini"), required=True)
    review.add_argument("--scope", choices=("pilot", "full"), default="pilot")
    review.add_argument("--recovery-id", default=None)

    apply_review = commands.add_parser("apply-review")
    apply_review.add_argument("review_csv")
    apply_review.add_argument("--provider", choices=("openai", "gemini"), required=True)
    apply_review.add_argument("--scope", choices=("pilot", "full"), default="pilot")
    apply_review.add_argument("--recovery-id", default=None)

    commands.add_parser("select-model")
    full = commands.add_parser("extract-full")
    full.add_argument("--allow-paid-api", action="store_true")

    mirror = commands.add_parser("mirror-lightrag")
    mirror.add_argument("--execute", action="store_true")
    validate = commands.add_parser("validate")
    validate.add_argument(
        "--offline",
        action="store_true",
        help="Validate files without connecting to PostgreSQL",
    )
    commands.add_parser("activate")
    backup = commands.add_parser("backup")
    backup.add_argument("--output", default=None)
    restore = commands.add_parser("restore-check")
    restore.add_argument("backup_path")
    return parser


def _context(root: Path, release_override: str | None) -> tuple[dict, str, Path, Path]:
    config = load_graph_config(root)
    release_id = release_override or config["release_id"]
    data_dir = root / "data/12_graph" / release_id
    report_dir = root / "reports/graph" / release_id
    return config, release_id, data_dir, report_dir


def _b7_dataset_id(root: Path) -> str:
    lock = read_json(root / "reports/b7_v03/b7_policy_lock.json")
    return lock["dataset_id"]


def _source_manifest_hash(data_dir: Path) -> str:
    return hashlib.sha256((data_dir / "source_manifest.json").read_bytes()).hexdigest()


def _load_nodes(path: Path) -> list[GraphNode]:
    values: dict[str, GraphNode] = {}
    for item in read_jsonl(path):
        node = GraphNode.model_validate(item)
        if node.graph_node_id in values:
            existing = values[node.graph_node_id]
            identity = (
                existing.node_kind,
                existing.node_type,
                existing.document_id,
                existing.canonical_node_id,
            )
            incoming_identity = (
                node.node_kind,
                node.node_type,
                node.document_id,
                node.canonical_node_id,
            )
            compatible_label = (
                existing.label == node.label
                or (
                    existing.node_kind == "semantic"
                    and normalized_label(existing.label) == normalized_label(node.label)
                )
            )
            if identity != incoming_identity or not compatible_label:
                raise ValueError(f"Conflicting graph node identity: {node.graph_node_id}")
            incoming_aliases = list(node.properties.get("aliases", []))
            if node.label != existing.label:
                incoming_aliases.append(node.label)
                node.properties["aliases"] = incoming_aliases
            for key in ("aliases", "source_node_ids"):
                existing.properties[key] = list(
                    dict.fromkeys(
                        [
                            *existing.properties.get(key, []),
                            *node.properties.get(key, []),
                        ]
                    )
                )
            provenance = [
                *existing.properties.get("provenance", []),
                *node.properties.get("provenance", []),
            ]
            existing.properties["provenance"] = list(
                {
                    json.dumps(value, ensure_ascii=False, sort_keys=True): value
                    for value in provenance
                }.values()
            )
            descriptions = list(
                dict.fromkeys(
                    value
                    for value in (
                        existing.properties.get("description", ""),
                        node.properties.get("description", ""),
                    )
                    if value
                )
            )
            if descriptions:
                existing.properties["description"] = "\n".join(descriptions)
        else:
            values[node.graph_node_id] = node
    return list(values.values())


def _load_edges(path: Path) -> list[GraphEdge]:
    return list({item["edge_id"]: GraphEdge.model_validate(item) for item in read_jsonl(path)}.values())


def _write_structural(root: Path, data_dir: Path, include_references: bool) -> dict:
    nodes, edges, findings = build_structural_graph(root, include_references=include_references)
    canonical_texts = {
        node.id: node.text
        for document_id in load_registry(root).documents
        for node in load_legal_document(root, document_id).nodes
    }
    errors = validate_graph(nodes, edges, canonical_texts=canonical_texts)
    if errors:
        raise ValueError("; ".join(errors))
    write_jsonl(data_dir / "structural_nodes.jsonl", [item.model_dump(mode="json") for item in nodes])
    write_jsonl(data_dir / "structural_edges.jsonl", [item.model_dump(mode="json") for item in edges])
    write_jsonl(data_dir / "reference_findings.jsonl", [asdict(item) for item in findings])
    counts = Counter(edge.relation_type for edge in edges)
    manifest = {
        "canonical_node_count": sum(node.node_kind == "canonical" for node in nodes),
        "authority_node_count": sum(node.node_kind == "authority" for node in nodes),
        "edge_counts": dict(counts),
        "reference_status_counts": dict(Counter(item.status for item in findings)),
    }
    write_json(data_dir / "structural_manifest.json", manifest)
    return manifest


def _full_review_gate(metrics: dict, config: dict) -> list[str]:
    gates = config["selection"]
    errors = []
    checks = (
        (metrics["canonical_mapping_accuracy"] >= gates["canonical_mapping_accuracy"], "canonical mapping"),
        (metrics["source_attribution_accuracy"] >= gates["source_attribution_accuracy"], "source attribution"),
        (metrics["entity_precision"] >= gates["entity_precision_min"], "entity precision"),
        (metrics["relationship_precision"] >= gates["relationship_precision_min"], "relationship precision"),
        (metrics["relationship_direction_accuracy"] >= gates["relationship_direction_accuracy_min"], "direction accuracy"),
        (metrics["missing_important_relation_rate"] <= gates["missing_important_relation_rate_max"], "missing relation rate"),
        (metrics["benchmark_issue_count"] == 0, "benchmark issues"),
    )
    for passed, label in checks:
        if not passed:
            errors.append(f"full review gate failed: {label}")
    return errors


def _require_frozen_b7_baseline(root: Path) -> None:
    result = run_preflight(root, external=False)
    required = ("canonical", "b7_lock", "b7_frozen_files", "lightrag_pin")
    failed = [name for name in required if not result["checks"][name]["ok"]]
    if failed:
        raise ValueError(
            "B7 baseline gate failed; reconcile production before LightRAG work: "
            + ", ".join(failed)
        )


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = _parser().parse_args()
    root = Path(args.root or Path.cwd()).resolve()
    config, release_id, data_dir, report_dir = _context(root, args.release_id)
    settings = GraphSettings.load(root)
    data_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    offline_diagnostic = (
        args.command == "preflight"
        or (args.command == "build-structural" and args.no_persist)
        or (args.command == "validate" and args.offline)
    )
    if not offline_diagnostic:
        _require_frozen_b7_baseline(root)

    if args.command == "preflight":
        result = run_preflight(root, external=not args.offline)
        write_json(report_dir / "preflight.json", result)
        if result["ready"] and not args.offline:
            baseline = {
                "schema_version": "1.0.0",
                "b7_dataset_id": result["checks"]["b7_lock"]["dataset_id"],
                "canonical_digest": result["checks"]["canonical"]["digest"],
                "graph_config_hash": result["graph_config_hash"],
                "graph_code_digest": result["graph_code_digest"],
                "checks": result["checks"],
            }
            write_json(report_dir / "b7_baseline_manifest.json", baseline)
    elif args.command == "db-init":
        result = {"migrations": init_database(root, settings), "settings": settings.redacted()}
    elif args.command == "export-articles":
        revision = read_json(root / "reports/b7_v03/b7_policy_lock.json")["winner_runtime_config"]["query_encoder"]["revision"]
        result = export_articles(
            root,
            data_dir,
            BGETokenCounter(revision),
            b7_dataset_id=_b7_dataset_id(root),
        )
        sources = load_sources(data_dir / "source_articles.jsonl")
        errors = validate_article_sources(sources)
        if errors:
            raise ValueError("; ".join(errors))
    elif args.command == "build-structural":
        source_manifest = read_json(data_dir / "source_manifest.json")
        sources = load_sources(data_dir / "source_articles.jsonl")
        result = _write_structural(root, data_dir, not args.no_references)
        release = GraphRelease(
            release_id=release_id,
            workspace=config["workspace"],
            canonical_digest=source_manifest["canonical_digest"],
            b7_dataset_id=source_manifest["b7_dataset_id"],
            source_manifest_hash=_source_manifest_hash(data_dir),
            config_hash=graph_config_hash(root),
            metadata={
                "lightrag": config["lightrag"],
                "embedding": config["embedding"],
                "code_digest": graph_code_digest(root),
                "migration_checksums": migration_checksums(root),
            },
        )
        nodes = _load_nodes(data_dir / "structural_nodes.jsonl")
        edges = _load_edges(data_dir / "structural_edges.jsonl")
        if args.no_persist:
            result["persisted"] = False
        else:
            with connect_graph(settings) as connection:
                upsert_release(connection, release)
                reset_building_release(connection, release_id)
                store_sources(connection, release_id, sources)
                store_graph(connection, release_id, nodes, edges)
            result["persisted"] = True
    elif args.command == "pilot-extract":
        if not args.allow_paid_api:
            raise ValueError("Refusing paid API call without --allow-paid-api")
        if args.resume_quota and args.provider != "gemini":
            raise ValueError("Quota recovery is currently restricted to the Gemini pilot")
        provider_config = config["pilot"]["providers"][args.provider]
        provider = provider_from_name(
            args.provider,
            provider_config["model"],
            reasoning_effort=provider_config.get("reasoning_effort", "low"),
            api_key=(
                settings.openai_api_key
                if args.provider == "openai"
                else settings.gemini_api_key
            ),
        )
        provider_preflight = provider.preflight()
        if not provider_preflight.get("exists"):
            raise ValueError(
                f"Provider account did not resolve frozen model {provider_config['model']}"
            )
        provider_preflight["pricing_snapshot"] = {
            "input_price_per_million": provider_config["input_price_per_million"],
            "output_price_per_million": provider_config["output_price_per_million"],
            "checked_at": provider_config["pricing_checked_at"],
        }
        pilot_dir = data_dir / "pilot" / args.provider
        preflight_dir = (
            pilot_dir / "recovery" / args.recovery_id
            if args.resume_quota
            else pilot_dir
        )
        if not args.resume_quota:
            write_json(preflight_dir / "provider_preflight.json", provider_preflight)
        sources = load_sources(data_dir / "source_articles.jsonl")
        if args.resume_quota:
            result = run_pilot_quota_recovery(
                provider,
                sources,
                pilot_dir,
                args.recovery_id,
                retry_schema_once=config["pilot"]["retry_schema_once"],
                input_price_per_million=provider_config["input_price_per_million"],
                output_price_per_million=provider_config["output_price_per_million"],
                sample_path=data_dir / "pilot_sample.json",
            )
            write_json(preflight_dir / "provider_preflight.json", provider_preflight)
        else:
            result = run_pilot(
                provider,
                sources,
                pilot_dir,
                retry_schema_once=config["pilot"]["retry_schema_once"],
                input_price_per_million=provider_config["input_price_per_million"],
                output_price_per_million=provider_config["output_price_per_million"],
                sample_path=data_dir / "pilot_sample.json",
                protocol_path=data_dir / "pilot_protocol_lock.json",
                max_workers=config["pilot"]["max_async_llm"],
            )
    elif args.command == "export-review":
        scope_dir = data_dir / args.scope / args.provider
        if args.recovery_id:
            if args.scope != "pilot":
                raise ValueError("--recovery-id is only valid for pilot review")
            scope_dir = scope_dir / "recovery" / args.recovery_id / "reconciled"
        output = report_dir / f"{args.scope}_{args.provider}_manual_review.csv"
        result = export_review(args.provider, scope_dir, output)
        result["artifact_dir"] = str(scope_dir)
        result["output"] = str(output)
    elif args.command == "apply-review":
        baseline = report_dir / f"{args.scope}_{args.provider}_manual_review.baseline.jsonl"
        rows, errors = load_and_validate_review(args.review_csv, baseline_path=baseline)
        if errors:
            raise ValueError("; ".join(errors))
        manifest_name = "pilot_manifest.json" if args.scope == "pilot" else "full_manifest.json"
        artifact_dir = data_dir / args.scope / args.provider
        if args.recovery_id:
            if args.scope != "pilot":
                raise ValueError("--recovery-id is only valid for pilot review")
            artifact_dir = artifact_dir / "recovery" / args.recovery_id / "reconciled"
        metrics = review_metrics(
            args.review_csv,
            artifact_dir / manifest_name,
        )
        output = report_dir / f"{args.scope}_{args.provider}_review_metrics.json"
        write_json(output, metrics)
        result = {"valid": True, "review_count": len(rows), "metrics": metrics, "output": str(output)}
    elif args.command == "select-model":
        model_lock_path = report_dir / "model_lock.json"
        if model_lock_path.is_file():
            raise FileExistsError(f"Extraction model is already locked: {model_lock_path}")
        metrics = [
            read_json(report_dir / f"pilot_{provider}_review_metrics.json")
            for provider in ("openai", "gemini")
        ]
        result = select_provider(metrics, config)
        if not result["winner"]:
            write_json(report_dir / "model_selection.json", result)
            raise ValueError("No extraction model passed the frozen pilot gates")
        winner_cfg = config["pilot"]["providers"][result["winner"]]
        result["model"] = winner_cfg["model"]
        result["config_hash"] = graph_config_hash(root)
        result["code_digest"] = graph_code_digest(root)
        result["pilot_protocol_sha256"] = hashlib.sha256(
            (data_dir / "pilot_protocol_lock.json").read_bytes()
        ).hexdigest()
        write_json(model_lock_path, result)
    elif args.command == "extract-full":
        if not args.allow_paid_api:
            raise ValueError("Refusing paid API call without --allow-paid-api")
        lock = read_json(report_dir / "model_lock.json")
        if lock["config_hash"] != graph_config_hash(root):
            raise ValueError("Graph config changed after pilot model lock")
        if lock["code_digest"] != graph_code_digest(root):
            raise ValueError("Graph source/prompt changed after pilot model lock")
        protocol_hash = hashlib.sha256(
            (data_dir / "pilot_protocol_lock.json").read_bytes()
        ).hexdigest()
        if lock["pilot_protocol_sha256"] != protocol_hash:
            raise ValueError("Pilot protocol changed after model lock")
        provider_name = lock["winner"]
        provider_cfg = config["pilot"]["providers"][provider_name]
        provider = provider_from_name(
            provider_name,
            provider_cfg["model"],
            reasoning_effort=provider_cfg.get("reasoning_effort", "low"),
            api_key=(
                settings.openai_api_key
                if provider_name == "openai"
                else settings.gemini_api_key
            ),
        )
        result = run_full_extraction(
            provider,
            load_sources(data_dir / "source_articles.jsonl"),
            data_dir / "full" / provider_name,
            retry_schema_once=config["pilot"]["retry_schema_once"],
            input_price_per_million=provider_cfg["input_price_per_million"],
            output_price_per_million=provider_cfg["output_price_per_million"],
            max_workers=config["pilot"]["max_async_llm"],
        )
        if result["failure_count"]:
            raise ValueError("Full extraction contains provider/schema failures")
        semantic_nodes = _load_nodes(data_dir / "full" / provider_name / "semantic_nodes.jsonl")
        semantic_edges = _load_edges(data_dir / "full" / provider_name / "semantic_edges.jsonl")
        with connect_graph(settings) as connection:
            store_graph(connection, release_id, semantic_nodes, semantic_edges)
            connection.execute(
                "UPDATE legal_graph_release SET extraction_provider=%s, extraction_model=%s WHERE release_id=%s AND status='BUILDING'",
                (provider_name, provider_cfg["model"], release_id),
            )
            transition_release(connection, release_id, "VALIDATING")
    elif args.command == "mirror-lightrag":
        lock = read_json(report_dir / "model_lock.json")
        provider_name = lock["winner"]
        sources = load_sources(data_dir / "source_articles.jsonl")
        nodes = _load_nodes(data_dir / "structural_nodes.jsonl")
        nodes.extend(_load_nodes(data_dir / "full" / provider_name / "semantic_nodes.jsonl"))
        nodes = list({node.graph_node_id: node for node in nodes}.values())
        edges = _load_edges(data_dir / "structural_edges.jsonl")
        edges.extend(_load_edges(data_dir / "full" / provider_name / "semantic_edges.jsonl"))
        edges = list({edge.edge_id: edge for edge in edges}.values())
        payload, manifest = build_lightrag_payload(sources, nodes, edges)
        manifest_path = report_dir / "mirror_manifest.json"
        previous = read_json(manifest_path) if manifest_path.is_file() else None
        manifest.update(
            {
                "workspace": config["workspace"],
                "executed": False,
                "status": "GENERATED",
            }
        )
        if (
            previous
            and previous.get("status") == "FAILED"
            and previous.get("workspace") == config["workspace"]
        ):
            manifest["status"] = "FAILED_PREVIOUS_ATTEMPT"
            manifest["retry_blocked"] = True
        write_json(data_dir / "mirror_payload.json", payload)
        write_json(manifest_path, manifest)
        if args.execute:
            if manifest.get("retry_blocked"):
                raise ValueError(
                    "This LightRAG workspace has a failed partial mirror; use a new versioned workspace"
                )
            command = [
                "docker", "compose", "--env-file", ".env", "--env-file", ".env.lightrag",
                "-f", "docker-compose.pgvector.yml", "-f", "docker-compose.lightrag.yml",
                "exec", "-T", "-e", f"WORKSPACE={config['workspace']}",
                "lightrag", "python", "/app/traffic_rag_mirror.py",
                f"/app/data/traffic_graph/{release_id}/mirror_payload.json",
            ]
            try:
                subprocess.run(command, cwd=root, check=True)
            except Exception as exc:
                manifest["status"] = "FAILED"
                manifest["error"] = str(exc)
                write_json(manifest_path, manifest)
                raise
            rows = []
            for source in sources:
                chunk_id = "chunk-" + hashlib.md5(source.text.encode("utf-8")).hexdigest()
                rows.append(("chunk", source.article_node_id, chunk_id, {}))
            for node in nodes:
                rows.append(("entity", node.graph_node_id, node.graph_node_id, {}))
            for aggregate, edge_ids in manifest["aggregated_edge_map"].items():
                rows.append(("edge", aggregate, aggregate, {"directed_edge_ids": edge_ids}))
            with connect_graph(settings) as connection:
                store_mirror_map(connection, release_id, config["workspace"], rows)
            manifest["executed"] = True
            manifest["status"] = "READY"
            write_json(manifest_path, manifest)
        result = manifest
    elif args.command == "validate":
        sources = load_sources(data_dir / "source_articles.jsonl")
        offline_errors = validate_article_sources(sources)
        nodes = _load_nodes(data_dir / "structural_nodes.jsonl")
        edges = _load_edges(data_dir / "structural_edges.jsonl")
        lock_path = report_dir / "model_lock.json"
        if lock_path.is_file():
            provider_name = read_json(lock_path)["winner"]
            full_dir = data_dir / "full" / provider_name
            if full_dir.is_dir():
                nodes = list({item.graph_node_id: item for item in [*nodes, *_load_nodes(full_dir / "semantic_nodes.jsonl")]}.values())
                edges = list({item.edge_id: item for item in [*edges, *_load_edges(full_dir / "semantic_edges.jsonl")]}.values())
        canonical_texts = {node_id: text for source in sources for node_id, text in source.node_texts.items()}
        offline_errors.extend(validate_graph(nodes, edges, canonical_texts=canonical_texts))
        mirror_validation = None
        mirror_path = report_dir / "mirror_manifest.json"
        if mirror_path.is_file():
            mirror = read_json(mirror_path)
            _, expected_mirror = build_lightrag_payload(sources, nodes, edges)
            digest_fields = (
                "chunk_source_digest",
                "entity_digest",
                "directed_edge_digest",
                "payload_digest",
            )
            mismatches = [
                field for field in digest_fields
                if mirror.get(field) != expected_mirror.get(field)
            ]
            if mismatches:
                offline_errors.append(
                    f"LightRAG mirror manifest differs from sidecar: {mismatches}"
                )
            mirror_validation = {
                "valid": not mismatches,
                "mismatches": mismatches,
                "status": mirror.get("status"),
                "executed": mirror.get("executed", False),
            }
        database = None
        if not args.offline:
            with connect_graph(settings) as connection:
                database = validate_database(connection, release_id)
        result = {
            "valid": not offline_errors and (database is None or database["valid"]),
            "offline_errors": offline_errors,
            "database": database,
            "mirror": mirror_validation,
            "mode": "offline" if args.offline else "database",
        }
        write_json(report_dir / "validation.json", result)
        if not result["valid"]:
            raise ValueError("Graph validation failed")
    elif args.command == "backup":
        result = backup_graph(root, settings, args.output)
    elif args.command == "restore-check":
        result = restore_graph_check(root, settings, args.backup_path, release_id)
    elif args.command == "activate":
        lock = read_json(report_dir / "model_lock.json")
        metrics = read_json(report_dir / f"full_{lock['winner']}_review_metrics.json")
        errors = _full_review_gate(metrics, config)
        restore = read_json(root / "reports/graph/restore_check_report.json")
        if not restore.get("passed") or restore.get("release_id") != release_id:
            errors.append("matching backup restore-check is required")
        mirror = read_json(report_dir / "mirror_manifest.json")
        if not mirror.get("executed"):
            errors.append("LightRAG mirror has not been executed")
        if errors:
            raise ValueError("; ".join(errors))
        with connect_graph(settings) as connection:
            validation = validate_database(connection, release_id)
            if not validation["valid"]:
                raise ValueError("; ".join(validation["errors"]))
            status = connection.execute(
                "SELECT status FROM legal_graph_release WHERE release_id=%s FOR UPDATE", (release_id,)
            ).fetchone()[0]
            if status == "VALIDATING":
                transition_release(connection, release_id, "READY")
            alias_count = rebuild_approved_aliases(connection, release_id)
            transition_release(connection, release_id, "ACTIVE")
        result = {
            "release_id": release_id,
            "status": "ACTIVE",
            "approved_alias_count": alias_count,
            "validation": validation,
        }
        write_json(report_dir / "graph_decision.json", result)
    else:
        raise AssertionError(args.command)

    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
