from __future__ import annotations

import hashlib
import json
import math
import subprocess
from pathlib import Path
from urllib.request import Request, urlopen

from src.postgres_store.config import PostgresSettings, require_postgres_packages
from src.legal_tree.loader import load_legal_document
from src.registry.loader import load_registry
from src.retrieval_runtime.service import _b7_code_digest

from .config import GraphSettings, graph_config_hash, load_graph_config
from .projection import canonical_digest


EXPECTED_B7_DATASET_ID = "49b413f0e17cba9f8f5fe9e24c972cc543505ddb3d9a172d4490b742d74542f5"
EXPECTED_CANONICAL_DIGEST = "a57cf88a0479c73d9bdad311d72db0a78428dc62a11020dd24ac69387616fa02"


def _sha256(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def _http_json(
    url: str,
    *,
    payload: dict | None = None,
    api_key: str | None = None,
    timeout: float = 5,
) -> dict:
    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key
    with urlopen(Request(url, data=body, headers=headers), timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def graph_code_digest(root: str | Path) -> str:
    root_path = Path(root).resolve()
    digest = hashlib.sha256()
    paths = [
        *sorted((root_path / "src/graph").glob("*.py")),
        *sorted((root_path / "sql/graph_migrations").glob("*.sql")),
        root_path / "docker-compose.lightrag.yml",
    ]
    for path in paths:
        if path.is_file():
            digest.update(path.relative_to(root_path).as_posix().encode("utf-8"))
            digest.update(path.read_bytes())
    return digest.hexdigest()


def run_preflight(root: str | Path, *, external: bool = True) -> dict:
    root_path = Path(root).resolve()
    graph_settings = GraphSettings.load(root_path)
    config = load_graph_config(root_path)
    lock_path = root_path / "reports/b7_v03/b7_policy_lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    registry = load_registry(root_path)
    canonical_node_count = sum(
        len(load_legal_document(root_path, document_id).nodes)
        for document_id in registry.documents
    )
    digest = canonical_digest(root_path)
    checks: dict[str, dict] = {}
    checks["canonical"] = {
        "ok": canonical_node_count == 1850 and digest == EXPECTED_CANONICAL_DIGEST,
        "digest": digest,
        "expected_digest": EXPECTED_CANONICAL_DIGEST,
        "node_count": canonical_node_count,
        "expected_node_count": 1850,
    }
    checks["b7_lock"] = {
        "ok": lock.get("dataset_id") == EXPECTED_B7_DATASET_ID,
        "dataset_id": lock.get("dataset_id"),
        "winner": lock.get("winner"),
    }
    application_hash = _sha256(root_path / "configs/application.yaml")
    strategy_lock_hash = _sha256(root_path / "reports/chunk_ablation/strategy_lock.json")
    runtime_lock_path = root_path / "reports/b7/b7_policy_lock.json"
    runtime_lock = (
        json.loads(runtime_lock_path.read_text(encoding="utf-8"))
        if runtime_lock_path.is_file()
        else None
    )
    runtime_code_digest = _b7_code_digest(root_path)
    checks["b7_frozen_files"] = {
        "ok": application_hash == lock.get("base_application_config_sha256")
        and strategy_lock_hash == lock.get("b6_strategy_lock_sha256")
        and runtime_code_digest == lock.get("code_digest")
        and runtime_lock == lock,
        "application_config_sha256": application_hash,
        "expected_application_config_sha256": lock.get("base_application_config_sha256"),
        "strategy_lock_sha256": strategy_lock_hash,
        "expected_strategy_lock_sha256": lock.get("b6_strategy_lock_sha256"),
        "runtime_code_digest": runtime_code_digest,
        "expected_runtime_code_digest": lock.get("code_digest"),
        "deployment_lock_matches_v03": runtime_lock == lock,
    }
    checks["lightrag_pin"] = {
        "ok": config["lightrag"]["version"] == "1.5.7"
        and config["lightrag"]["commit"]
        == "86e10c4aea9b65f4fcf1d769791d7a3ba4b560d7",
        **config["lightrag"],
    }
    checks["settings"] = {
        "ok": bool(graph_settings.app_dsn and graph_settings.admin_dsn and graph_settings.app_password),
        **graph_settings.redacted(),
    }

    if external:
        try:
            inspected = subprocess.run(
                [
                    "docker", "image", "inspect", config["lightrag"]["image"],
                    "--format",
                    "{{json .RepoDigests}}|{{.Id}}|{{json .Config.Labels}}",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            ).stdout.strip()
            repo_digests, image_id, labels_json = inspected.split("|", 2)
            labels = json.loads(labels_json) or {}
            revision = labels.get("org.opencontainers.image.revision")
            expected_revision = config["lightrag"]["commit"]
            checks["lightrag_image"] = {
                "ok": image_id.startswith("sha256:") and revision == expected_revision,
                "image": config["lightrag"]["image"],
                "repo_digests": json.loads(repo_digests),
                "image_id": image_id,
                "revision": revision,
                "expected_revision": expected_revision,
            }
        except Exception as exc:
            checks["lightrag_image"] = {"ok": False, "error": str(exc)}

        try:
            psycopg, _ = require_postgres_packages()
            runtime_dsn = PostgresSettings.load(root_path).require("runtime")
            with psycopg.connect(runtime_dsn, connect_timeout=5) as connection:
                rows = connection.execute(
                    "SELECT dataset_id FROM traffic_rag.retrieval_dataset WHERE status='ACTIVE'"
                ).fetchall()
            active = rows[0][0] if len(rows) == 1 else None
            checks["b7_active_database"] = {
                "ok": active == lock.get("dataset_id") and len(rows) == 1,
                "dataset_id": active,
                "active_count": len(rows),
            }
        except Exception as exc:
            checks["b7_active_database"] = {"ok": False, "error": str(exc)}

        try:
            psycopg, _ = require_postgres_packages()
            with psycopg.connect(graph_settings.require("app_dsn"), connect_timeout=5) as connection:
                vector = connection.execute(
                    "SELECT extversion FROM pg_extension WHERE extname='vector'"
                ).fetchone()
                tables = connection.execute(
                    "SELECT count(*) FROM information_schema.tables WHERE table_name='legal_graph_release'"
                ).fetchone()[0]
            checks["graph_database"] = {
                "ok": bool(vector and tables == 1),
                "vector_version": vector[0] if vector else None,
                "migration_table_ready": tables == 1,
            }
        except Exception as exc:
            checks["graph_database"] = {"ok": False, "error": str(exc)}

        try:
            tags = _http_json(f"{graph_settings.ollama_host.rstrip('/')}/api/tags")
            models = tags.get("models", [])
            names = [item.get("name") for item in models]
            matched_model = next(
                (
                    item for item in models
                    if item.get("name") in {
                        graph_settings.embedding_model,
                        f"{graph_settings.embedding_model}:latest",
                    }
                    or item.get("model") in {
                        graph_settings.embedding_model,
                        f"{graph_settings.embedding_model}:latest",
                    }
                ),
                None,
            )
            source_path = root_path / "data/12_graph/kg-v001/source_articles.jsonl"
            source_rows = (
                [
                    json.loads(line)
                    for line in source_path.read_text(encoding="utf-8").splitlines()
                    if line
                ]
                if source_path.is_file()
                else []
            )
            longest = max(source_rows, key=lambda item: item["token_count"]) if source_rows else None
            request = {
                "model": graph_settings.embedding_model,
                "input": longest["extraction_text"] if longest else "kiểm tra vector",
            }
            embedding = _http_json(
                f"{graph_settings.ollama_host.rstrip('/')}/api/embed",
                payload=request,
                timeout=120,
            )
            repeated = _http_json(
                f"{graph_settings.ollama_host.rstrip('/')}/api/embed",
                payload=request,
                timeout=120,
            )
            vectors = embedding.get("embeddings") or []
            repeated_vectors = repeated.get("embeddings") or []
            dimensions = len(vectors[0]) if vectors else 0
            repeated_dimensions = len(repeated_vectors[0]) if repeated_vectors else 0
            finite = bool(vectors) and all(
                math.isfinite(float(value)) for value in vectors[0]
            )
            checks["ollama_embedding"] = {
                "ok": bool(matched_model)
                and dimensions == 1024
                and repeated_dimensions == 1024
                and finite,
                "models": names,
                "dimensions": dimensions,
                "repeated_dimensions": repeated_dimensions,
                "finite": finite,
                "model": graph_settings.embedding_model,
                "model_digest": matched_model.get("digest") if matched_model else None,
                "preflight_input_tokens": longest.get("token_count") if longest else None,
            }
        except Exception as exc:
            checks["ollama_embedding"] = {"ok": False, "error": str(exc)}

        try:
            health = _http_json(
                f"{graph_settings.lightrag_url.rstrip('/')}/health",
                api_key=graph_settings.lightrag_api_key,
            )
            checks["lightrag_server"] = {"ok": True, "response": health}
        except Exception as exc:
            checks["lightrag_server"] = {"ok": False, "error": str(exc)}

    required = ["canonical", "b7_lock", "b7_frozen_files", "lightrag_pin", "settings"]
    if external:
        required.extend(
            (
                "lightrag_image",
                "b7_active_database",
                "graph_database",
                "ollama_embedding",
                "lightrag_server",
            )
        )
    return {
        "ready": all(checks[name]["ok"] for name in required),
        "checks": checks,
        "graph_config_hash": graph_config_hash(root_path),
        "graph_code_digest": graph_code_digest(root_path),
    }
