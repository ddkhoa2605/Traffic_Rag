from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import yaml


def _dotenv(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


@dataclass(frozen=True)
class GraphSettings:
    admin_dsn: str | None
    app_dsn: str | None
    app_password: str | None
    database: str = "traffic_rag_lightrag"
    app_role: str = "traffic_rag_lightrag_app"
    ollama_host: str = "http://127.0.0.1:11434"
    embedding_model: str = "bge-m3"
    lightrag_url: str = "http://127.0.0.1:9621"
    lightrag_api_key: str | None = None
    openai_api_key: str | None = None
    gemini_api_key: str | None = None

    @classmethod
    def load(cls, root: str | Path) -> "GraphSettings":
        root_path = Path(root).resolve()
        values = {**_dotenv(root_path / ".env"), **_dotenv(root_path / ".env.lightrag")}

        def get(name: str, default: str | None = None) -> str | None:
            return os.environ.get(name) or values.get(name) or default

        return cls(
            admin_dsn=get("LIGHTRAG_ADMIN_DSN") or get("TRAFFIC_RAG_ADMIN_DSN"),
            app_dsn=get("LIGHTRAG_DSN"),
            app_password=get("LIGHTRAG_POSTGRES_PASSWORD"),
            database=get("LIGHTRAG_POSTGRES_DATABASE", "traffic_rag_lightrag") or "traffic_rag_lightrag",
            app_role=get("LIGHTRAG_POSTGRES_USER", "traffic_rag_lightrag_app") or "traffic_rag_lightrag_app",
            ollama_host=get("OLLAMA_HOST", "http://127.0.0.1:11434") or "http://127.0.0.1:11434",
            embedding_model=get("EMBEDDING_MODEL", "bge-m3") or "bge-m3",
            lightrag_url=get("LIGHTRAG_URL", "http://127.0.0.1:9621") or "http://127.0.0.1:9621",
            lightrag_api_key=get("LIGHTRAG_API_KEY"),
            openai_api_key=get("OPENAI_API_KEY"),
            gemini_api_key=get("GEMINI_API_KEY"),
        )

    def require(self, field: str) -> str:
        value = getattr(self, field)
        if not value:
            raise ValueError(f"Missing graph setting: {field}")
        return value

    def redacted(self) -> dict:
        return {
            "admin_database": urlparse(self.admin_dsn or "").path.lstrip("/"),
            "app_database": urlparse(self.app_dsn or "").path.lstrip("/"),
            "database": self.database,
            "app_role": self.app_role,
            "ollama_host": self.ollama_host,
            "embedding_model": self.embedding_model,
            "lightrag_url": self.lightrag_url,
            "has_admin_dsn": bool(self.admin_dsn),
            "has_app_dsn": bool(self.app_dsn),
            "has_app_password": bool(self.app_password),
            "has_lightrag_api_key": bool(self.lightrag_api_key),
            "has_openai_api_key": bool(self.openai_api_key),
            "has_gemini_api_key": bool(self.gemini_api_key),
        }


def load_graph_config(root: str | Path) -> dict:
    path = Path(root).resolve() / "configs/graph/graph.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def graph_config_hash(root: str | Path) -> str:
    root_path = Path(root).resolve()
    digest = hashlib.sha256()
    for path in sorted((root_path / "configs/graph").glob("*.y*ml")):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()
