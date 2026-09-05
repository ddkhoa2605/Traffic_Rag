from __future__ import annotations

import hashlib
import json
import re
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from src.parser.io import read_jsonl, write_json, write_jsonl

from .models import GraphSourceDocument
from .models import ArticleExtraction
from .prompt import SYSTEM_PROMPT, extraction_prompt
from .providers import ExtractionProvider, ProviderSchemaError
from .semantic import build_semantic_graph


_QUOTA_MARKERS = ("429", "RESOURCE_EXHAUSTED", "QUOTA EXCEEDED")


def is_quota_failure(failure: dict) -> bool:
    """Return true only for provider quota failures, never schema failures."""
    if failure.get("error_type") == "ProviderSchemaError":
        return False
    message = str(failure.get("error", "")).upper()
    return "429" in message and any(marker in message for marker in _QUOTA_MARKERS[1:])


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_jsonl_if_exists(path: Path) -> list[dict]:
    return read_jsonl(path) if path.is_file() else []


def load_sources(path: str | Path) -> list[GraphSourceDocument]:
    return [GraphSourceDocument.model_validate(item) for item in read_jsonl(Path(path))]


def select_pilot_articles(sources: list[GraphSourceDocument]) -> list[GraphSourceDocument]:
    by_document: dict[str, list[GraphSourceDocument]] = defaultdict(list)
    for source in sources:
        by_document[source.canonical_document_id].append(source)
    selected: list[GraphSourceDocument] = []
    for document_id in sorted(by_document):
        candidates = by_document[document_id]
        chosen: dict[str, GraphSourceDocument] = {}

        def take(values, count: int) -> None:
            for item in values:
                if len(chosen) >= 10 or count <= 0:
                    break
                if item.article_node_id not in chosen:
                    chosen[item.article_node_id] = item
                    count -= 1

        take(sorted(candidates, key=lambda item: (-item.token_count, item.article_node_id)), 2)
        take(
            sorted(
                candidates,
                key=lambda item: (-sum("TYPE=point" in line for line in item.extraction_text.splitlines()), item.article_node_id),
            ),
            2,
        )
        take(
            sorted(
                candidates,
                key=lambda item: (-item.text.casefold().count("điều "), item.article_node_id),
            ),
            2,
        )
        definitions = [
            item for item in candidates
            if "được hiểu" in item.text.casefold() or "trong luật này" in item.text.casefold()
        ]
        take(sorted(definitions, key=lambda item: item.article_node_id), 2)
        take(sorted(candidates, key=lambda item: item.article_node_id), 10)
        selected.extend(chosen.values())
    if len(selected) != 20:
        raise ValueError(f"Pilot selection requires 20 Articles, found {len(selected)}")
    return selected


def run_pilot(
    provider: ExtractionProvider,
    sources: list[GraphSourceDocument],
    output_dir: str | Path,
    *,
    extraction_version: str = "legal-kg-prompt-v1",
    retry_schema_once: bool = True,
    input_price_per_million: float = 0.0,
    output_price_per_million: float = 0.0,
    sample_path: str | Path | None = None,
    protocol_path: str | Path | None = None,
    max_workers: int = 2,
) -> dict:
    selected = select_pilot_articles(sources)
    if sample_path is not None:
        sample_file = Path(sample_path)
        frozen = {
            "schema_version": "1.0.0",
            "article_node_ids": [item.article_node_id for item in selected],
            "source_hashes": {item.article_node_id: item.content_hash for item in selected},
        }
        if sample_file.is_file():
            existing = json.loads(sample_file.read_text(encoding="utf-8"))
            if existing != frozen:
                raise ValueError("Frozen pilot sample/source hashes changed")
        else:
            write_json(sample_file, frozen)
    if protocol_path is not None:
        protocol_file = Path(protocol_path)
        protocol = {
            "schema_version": "1.0.0",
            "extraction_version": extraction_version,
            "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
            "output_schema_sha256": hashlib.sha256(
                json.dumps(
                    ArticleExtraction.model_json_schema(),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest(),
            "sample_sha256": hashlib.sha256(
                Path(sample_path).read_bytes() if sample_path is not None else b""
            ).hexdigest(),
            "max_async_llm": max_workers,
            "retry_schema_once": retry_schema_once,
        }
        if protocol_file.is_file():
            existing = json.loads(protocol_file.read_text(encoding="utf-8"))
            if existing != protocol:
                raise ValueError("Frozen pilot prompt/schema/sample protocol changed")
        else:
            write_json(protocol_file, protocol)
    return _run_extraction(
        provider,
        selected,
        output_dir,
        extraction_version=extraction_version,
        retry_schema_once=retry_schema_once,
        input_price_per_million=input_price_per_million,
        output_price_per_million=output_price_per_million,
        run_kind="pilot",
        max_workers=max_workers,
    )


def run_pilot_quota_recovery(
    provider: ExtractionProvider,
    sources: list[GraphSourceDocument],
    base_output_dir: str | Path,
    recovery_id: str,
    *,
    sample_path: str | Path,
    extraction_version: str = "legal-kg-prompt-v1",
    retry_schema_once: bool = True,
    input_price_per_million: float = 0.0,
    output_price_per_million: float = 0.0,
) -> dict:
    """Retry only audited quota failures and materialize an immutable reconciled view."""
    if not re.fullmatch(r"[a-z0-9][a-z0-9._-]{0,63}", recovery_id):
        raise ValueError("recovery_id must contain only lowercase letters, digits, '.', '_' or '-'")

    base = Path(base_output_dir)
    base_manifest_path = base / "pilot_manifest.json"
    base_failures_path = base / "failures.json"
    sample_file = Path(sample_path)
    for required in (base_manifest_path, base_failures_path, sample_file):
        if not required.is_file():
            raise FileNotFoundError(required)

    base_manifest = json.loads(base_manifest_path.read_text(encoding="utf-8"))
    if base_manifest.get("run_kind") != "pilot" or base_manifest.get("provider") != provider.name:
        raise ValueError("Base pilot manifest does not match the recovery provider")
    base_failures = json.loads(base_failures_path.read_text(encoding="utf-8"))
    quota_failures = [item for item in base_failures if is_quota_failure(item)]
    if not quota_failures:
        raise ValueError("Base pilot has no recoverable quota failures")

    frozen_sample = json.loads(sample_file.read_text(encoding="utf-8"))
    sample_ids = list(frozen_sample.get("article_node_ids", []))
    source_hashes = frozen_sample.get("source_hashes", {})
    source_map = {item.article_node_id: item for item in sources}
    retry_ids = [item["article_node_id"] for item in quota_failures]
    if len(retry_ids) != len(set(retry_ids)):
        raise ValueError("Duplicate Article in quota failure set")
    if any(article_id not in sample_ids for article_id in retry_ids):
        raise ValueError("Quota recovery attempted an Article outside the frozen pilot sample")
    selected: list[GraphSourceDocument] = []
    for article_id in sample_ids:
        if article_id not in retry_ids:
            continue
        source = source_map.get(article_id)
        if source is None or source.content_hash != source_hashes.get(article_id):
            raise ValueError(f"Frozen source missing or changed for recovery: {article_id}")
        selected.append(source)

    recovery_dir = base / "recovery" / recovery_id
    if recovery_dir.exists() and any(recovery_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite audited quota recovery: {recovery_dir}")

    retry_delays = []
    for failure in quota_failures:
        message = str(failure.get("error", ""))
        values = [
            *re.findall(r"retry in ([0-9]+(?:\.[0-9]+)?)s", message, re.IGNORECASE),
            *re.findall(r"retryDelay[^0-9]*([0-9]+(?:\.[0-9]+)?)s", message, re.IGNORECASE),
        ]
        retry_delays.extend(float(value) for value in values)
    cooldown_target = max(retry_delays, default=0.0) + (2.0 if retry_delays else 0.0)
    base_age = max(0.0, time.time() - base_manifest_path.stat().st_mtime)
    cooldown_seconds = min(45.0, max(0.0, cooldown_target - base_age))
    if cooldown_seconds:
        time.sleep(cooldown_seconds)

    recovery_manifest = _run_extraction(
        provider,
        selected,
        recovery_dir,
        extraction_version=extraction_version,
        retry_schema_once=retry_schema_once,
        input_price_per_million=input_price_per_million,
        output_price_per_million=output_price_per_million,
        run_kind="pilot_recovery",
        max_workers=1,
    )
    recovery_manifest.update(
        {
            "recovery_id": recovery_id,
            "recovery_reason": "provider_quota",
            "base_manifest_sha256": _sha256_file(base_manifest_path),
            "attempted_article_node_ids": retry_ids,
            "excluded_failure_article_node_ids": [
                item["article_node_id"] for item in base_failures if not is_quota_failure(item)
            ],
            "cooldown_seconds": cooldown_seconds,
        }
    )
    recovery_manifest_path = recovery_dir / "pilot_recovery_manifest.json"
    write_json(recovery_manifest_path, recovery_manifest)

    reconciled_dir = recovery_dir / "reconciled"
    if reconciled_dir.exists() and any(reconciled_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite reconciled pilot: {reconciled_dir}")

    base_rows = _read_jsonl_if_exists(base / "extractions.jsonl")
    recovery_rows = _read_jsonl_if_exists(recovery_dir / "extractions.jsonl")
    rows_by_article: dict[str, dict] = {}
    for row in [*base_rows, *recovery_rows]:
        article_id = row["article_node_id"]
        if article_id in rows_by_article:
            raise ValueError(f"Recovery duplicated a successful Article: {article_id}")
        rows_by_article[article_id] = row
    reconciled_rows = [rows_by_article[item] for item in sample_ids if item in rows_by_article]

    recovery_failures = json.loads((recovery_dir / "failures.json").read_text(encoding="utf-8"))
    unresolved_failures = [
        *[item for item in base_failures if not is_quota_failure(item)],
        *recovery_failures,
    ]
    if len(reconciled_rows) + len(unresolved_failures) != len(sample_ids):
        raise ValueError("Reconciled pilot does not cover the frozen 20-Article sample exactly once")

    write_jsonl(reconciled_dir / "extractions.jsonl", reconciled_rows)
    for name in ("semantic_nodes.jsonl", "semantic_edges.jsonl", "rejected.jsonl"):
        write_jsonl(
            reconciled_dir / name,
            [
                *_read_jsonl_if_exists(base / name),
                *_read_jsonl_if_exists(recovery_dir / name),
            ],
        )
    write_json(reconciled_dir / "failures.json", unresolved_failures)

    totals = {
        key: (base_manifest.get(key, 0) or 0) + (recovery_manifest.get(key, 0) or 0)
        for key in ("input_tokens", "output_tokens", "elapsed_seconds", "estimated_cost_usd")
    }
    reconciled_manifest = {
        "schema_version": "1.1.0",
        "run_kind": "pilot_reconciled",
        "provider": provider.name,
        "model": provider.model,
        "article_count": len(reconciled_rows),
        "failure_count": len(unresolved_failures),
        "schema_failure_count": sum(
            item.get("error_type") == "ProviderSchemaError" for item in unresolved_failures
        ),
        "transient_failure_count": sum(is_quota_failure(item) for item in unresolved_failures),
        **totals,
        "input_price_per_million": input_price_per_million,
        "output_price_per_million": output_price_per_million,
        "max_async_llm": {"base": base_manifest.get("max_async_llm"), "recovery": 1},
        "extraction_version": extraction_version,
        "system_prompt_sha256": base_manifest.get("system_prompt_sha256"),
        "output_schema_sha256": base_manifest.get("output_schema_sha256"),
        "base_manifest_sha256": _sha256_file(base_manifest_path),
        "recovery_manifest_sha256": _sha256_file(recovery_manifest_path),
        "frozen_sample_sha256": _sha256_file(sample_file),
        "recovery_id": recovery_id,
    }
    write_json(reconciled_dir / "pilot_manifest.json", reconciled_manifest)
    write_json(
        reconciled_dir / "lineage.json",
        {
            "schema_version": "1.0.0",
            "base_dir": str(base),
            "recovery_dir": str(recovery_dir),
            "base_success_article_node_ids": [row["article_node_id"] for row in base_rows],
            "recovered_article_node_ids": [row["article_node_id"] for row in recovery_rows],
            "unresolved_failure_article_node_ids": [
                item["article_node_id"] for item in unresolved_failures
            ],
        },
    )
    return {
        "recovery": recovery_manifest,
        "reconciled": reconciled_manifest,
        "recovery_dir": str(recovery_dir),
        "reconciled_dir": str(reconciled_dir),
    }


def run_full_extraction(
    provider: ExtractionProvider,
    sources: list[GraphSourceDocument],
    output_dir: str | Path,
    *,
    extraction_version: str = "legal-kg-prompt-v1",
    retry_schema_once: bool = True,
    input_price_per_million: float = 0.0,
    output_price_per_million: float = 0.0,
    max_workers: int = 2,
) -> dict:
    return _run_extraction(
        provider,
        sources,
        output_dir,
        extraction_version=extraction_version,
        retry_schema_once=retry_schema_once,
        input_price_per_million=input_price_per_million,
        output_price_per_million=output_price_per_million,
        run_kind="full",
        max_workers=max_workers,
    )


def _run_extraction(
    provider: ExtractionProvider,
    selected_sources: list[GraphSourceDocument],
    output_dir: str | Path,
    *,
    extraction_version: str,
    retry_schema_once: bool,
    input_price_per_million: float,
    output_price_per_million: float,
    run_kind: str,
    max_workers: int,
) -> dict:
    output = Path(output_dir)
    manifest_path = output / f"{run_kind}_manifest.json"
    if manifest_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite audited {run_kind} extraction: {manifest_path}"
        )
    raw_dir = output / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    semantic_nodes: list[dict] = []
    semantic_edges: list[dict] = []
    rejected: list[dict] = []
    failures: list[dict] = []
    totals = {"input_tokens": 0, "output_tokens": 0, "elapsed_seconds": 0.0}
    def extract_one(source: GraphSourceDocument) -> tuple[GraphSourceDocument, object | None, list[dict], Exception | None]:
        prompt = extraction_prompt(source)
        error: Exception | None = None
        result = None
        schema_failures: list[dict] = []
        for attempt in range(2 if retry_schema_once else 1):
            try:
                result = provider.extract(prompt)
                break
            except ProviderSchemaError as exc:
                error = exc
                schema_failures.append(
                    {
                        "attempt": attempt + 1,
                        "raw": exc.raw,
                        "metadata": exc.metadata,
                        "error": str(exc),
                    }
                )
                if attempt == 0 and retry_schema_once:
                    continue
            except Exception as exc:
                error = exc
                break
        return source, result, schema_failures, error

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        extracted = list(pool.map(extract_one, selected_sources))

    for source, result, schema_failures, error in extracted:
        for failure in schema_failures:
            write_json(
                raw_dir / f"{source.article_node_id}.schema_attempt_{failure['attempt']}.json",
                failure,
            )
            for key in totals:
                totals[key] += failure["metadata"].get(key, 0) or 0
        if result is None:
            failures.append(
                {
                    "article_node_id": source.article_node_id,
                    "error": str(error),
                    "error_type": type(error).__name__ if error else "UnknownError",
                    "schema_attempts": len(schema_failures),
                }
            )
            continue
        write_json(raw_dir / f"{source.article_node_id}.json", result.raw)
        built = build_semantic_graph(
            source, result.payload, extraction_version=extraction_version
        )
        semantic_nodes.extend(node.model_dump(mode="json") for node in built.nodes)
        semantic_edges.extend(edge.model_dump(mode="json") for edge in built.edges)
        rejected.extend(
            {"article_node_id": source.article_node_id, **item} for item in built.rejected
        )
        rows.append(
            {
                "article_node_id": source.article_node_id,
                "provider": provider.name,
                "model": provider.model,
                "request_sha256": hashlib.sha256(
                    extraction_prompt(source).encode("utf-8")
                ).hexdigest(),
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "entity_count": len(built.nodes),
                "relationship_count": len(built.edges),
                **result.metadata,
            }
        )
        for key in totals:
            totals[key] += result.metadata.get(key, 0) or 0
    write_jsonl(output / "extractions.jsonl", rows)
    write_jsonl(output / "semantic_nodes.jsonl", semantic_nodes)
    write_jsonl(output / "semantic_edges.jsonl", semantic_edges)
    write_jsonl(output / "rejected.jsonl", rejected)
    write_json(output / "failures.json", failures)
    cost = (
        totals["input_tokens"] * input_price_per_million
        + totals["output_tokens"] * output_price_per_million
    ) / 1_000_000
    manifest = {
        "schema_version": "1.0.0",
        "run_kind": run_kind,
        "provider": provider.name,
        "model": provider.model,
        "article_count": len(rows),
        "failure_count": len(failures),
        **totals,
        "estimated_cost_usd": cost,
        "input_price_per_million": input_price_per_million,
        "output_price_per_million": output_price_per_million,
        "max_async_llm": max_workers,
        "extraction_version": extraction_version,
        "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
        "output_schema_sha256": hashlib.sha256(
            json.dumps(
                ArticleExtraction.model_json_schema(),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
    }
    write_json(manifest_path, manifest)
    return manifest
