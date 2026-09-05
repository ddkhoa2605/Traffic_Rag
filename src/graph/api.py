from __future__ import annotations

import os
from collections import deque
from pathlib import Path
from typing import Any

from .config import GraphSettings
from .repository import connect_graph


def create_app(root: str | Path | None = None):
    try:
        import httpx
        from fastapi import Depends, FastAPI, Header, HTTPException, Query
        from pydantic import BaseModel, Field
    except ImportError as exc:
        raise RuntimeError('Install graph dependencies: pip install -e ".[graph]"') from exc

    root_path = Path(root or Path.cwd()).resolve()
    settings = GraphSettings.load(root_path)
    adapter_key = os.environ.get("GRAPH_API_KEY") or settings.lightrag_api_key

    def authorize(x_api_key: str | None = Header(default=None)) -> None:
        if adapter_key and x_api_key != adapter_key:
            raise HTTPException(status_code=401, detail="Invalid API key")

    def active_release(connection) -> str:
        rows = connection.execute(
            "SELECT release_id FROM legal_graph_release WHERE status='ACTIVE'"
        ).fetchall()
        if len(rows) != 1:
            raise HTTPException(status_code=503, detail=f"Expected one ACTIVE graph release, found {len(rows)}")
        return rows[0][0]

    def node_payload(connection, release_id: str, node_id: str) -> dict:
        row = connection.execute(
            """
            SELECT graph_node_id, node_kind, node_type, label, document_id,
                   canonical_node_id, properties
            FROM legal_graph_node WHERE release_id=%s AND graph_node_id=%s
            """,
            (release_id, node_id),
        ).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Graph node not found")
        source = None
        if row[5]:
            source_row = connection.execute(
                """
                SELECT article_node_id, node_texts
                FROM legal_graph_source_document
                WHERE release_id=%s AND node_texts ? %s
                ORDER BY article_node_id LIMIT 1
                """,
                (release_id, row[5]),
            ).fetchone()
            if source_row:
                source = {
                    "article_node_id": source_row[0],
                    "canonical_node_id": row[5],
                    "text": source_row[1].get(row[5]),
                }
        return {
            "graph_node_id": row[0],
            "node_kind": row[1],
            "node_type": row[2],
            "label": row[3],
            "document_id": row[4],
            "canonical_node_id": row[5],
            "properties": row[6],
            "canonical_source": source,
        }

    class GraphQueryRequest(BaseModel):
        query: str = Field(min_length=3)
        mode: str = "mix"
        only_need_context: bool = False

    app = FastAPI(
        title="Traffic RAG Experimental Legal Graph",
        version="1.0.0",
        docs_url="/experimental/graph/docs",
        openapi_url="/experimental/graph/openapi.json",
    )

    @app.get("/experimental/graph/health", dependencies=[Depends(authorize)])
    def health() -> dict:
        with connect_graph(settings) as connection:
            release_id = active_release(connection)
            counts = connection.execute(
                """
                SELECT
                  (SELECT count(*) FROM legal_graph_node WHERE release_id=%s),
                  (SELECT count(*) FROM legal_graph_edge WHERE release_id=%s)
                """,
                (release_id, release_id),
            ).fetchone()
        return {"ready": True, "release_id": release_id, "node_count": counts[0], "edge_count": counts[1]}

    @app.get("/experimental/graph/search", dependencies=[Depends(authorize)])
    def search(
        q: str = Query(min_length=1),
        node_kind: str | None = None,
        node_type: str | None = None,
        limit: int = Query(default=20, ge=1, le=100),
    ) -> dict:
        with connect_graph(settings) as connection:
            release_id = active_release(connection)
            rows = connection.execute(
                """
                SELECT n.graph_node_id, n.node_kind, n.node_type, n.label,
                       n.document_id, n.canonical_node_id
                FROM legal_graph_node n
                WHERE n.release_id=%s AND (
                    n.label ILIKE %s OR EXISTS (
                        SELECT 1 FROM legal_graph_alias a
                        WHERE a.release_id=n.release_id
                          AND a.graph_node_id=n.graph_node_id
                          AND a.review_status='approved'
                          AND a.alias ILIKE %s
                    )
                )
                  AND (%s IS NULL OR n.node_kind=%s)
                  AND (%s IS NULL OR n.node_type=%s)
                ORDER BY CASE WHEN lower(n.label)=lower(%s) THEN 0 ELSE 1 END,
                         length(n.label), n.graph_node_id
                LIMIT %s
                """,
                (
                    release_id, f"%{q}%", f"%{q}%", node_kind, node_kind,
                    node_type, node_type, q, limit,
                ),
            ).fetchall()
        return {
            "release_id": release_id,
            "results": [
                {
                    "graph_node_id": row[0], "node_kind": row[1], "node_type": row[2],
                    "label": row[3], "document_id": row[4], "canonical_node_id": row[5],
                }
                for row in rows
            ],
        }

    @app.get("/experimental/graph/nodes/{node_id:path}", dependencies=[Depends(authorize)])
    def get_node(node_id: str) -> dict:
        with connect_graph(settings) as connection:
            release_id = active_release(connection)
            result = node_payload(connection, release_id, node_id)
            edges = connection.execute(
                """
                SELECT edge_id, source_node_id, target_node_id, relation_type,
                       description, confidence
                FROM legal_graph_edge
                WHERE release_id=%s AND (source_node_id=%s OR target_node_id=%s)
                ORDER BY relation_type, edge_id
                """,
                (release_id, node_id, node_id),
            ).fetchall()
        result["release_id"] = release_id
        result["relations"] = [
            {
                "edge_id": row[0], "source_node_id": row[1], "target_node_id": row[2],
                "relation_type": row[3], "description": row[4], "confidence": row[5],
            }
            for row in edges
        ]
        return result

    @app.get("/experimental/graph/subgraph", dependencies=[Depends(authorize)])
    def subgraph(
        node_id: str,
        depth: int = Query(default=1, ge=0, le=3),
        relation_type: list[str] | None = Query(default=None),
        max_nodes: int = Query(default=200, ge=1, le=1000),
    ) -> dict:
        allowed = set(relation_type or [])
        with connect_graph(settings) as connection:
            release_id = active_release(connection)
            queue = deque([(node_id, 0)])
            visited: set[str] = set()
            edge_map: dict[str, dict] = {}
            while queue and len(visited) < max_nodes:
                current, level = queue.popleft()
                if current in visited:
                    continue
                visited.add(current)
                if level >= depth:
                    continue
                rows = connection.execute(
                    """
                    SELECT edge_id, source_node_id, target_node_id, relation_type,
                           description, confidence
                    FROM legal_graph_edge
                    WHERE release_id=%s AND (source_node_id=%s OR target_node_id=%s)
                    ORDER BY edge_id
                    """,
                    (release_id, current, current),
                ).fetchall()
                for row in rows:
                    if allowed and row[3] not in allowed:
                        continue
                    edge_map[row[0]] = {
                        "edge_id": row[0], "source_node_id": row[1], "target_node_id": row[2],
                        "relation_type": row[3], "description": row[4], "confidence": row[5],
                    }
                    neighbor = row[2] if row[1] == current else row[1]
                    if neighbor not in visited:
                        queue.append((neighbor, level + 1))
            nodes = [node_payload(connection, release_id, value) for value in sorted(visited)]
        return {"release_id": release_id, "nodes": nodes, "edges": list(edge_map.values())}

    @app.get("/experimental/graph/edges/{edge_id:path}/provenance", dependencies=[Depends(authorize)])
    def provenance(edge_id: str) -> dict:
        with connect_graph(settings) as connection:
            release_id = active_release(connection)
            rows = connection.execute(
                """
                SELECT canonical_node_id, evidence_text, start_offset, end_offset
                FROM legal_graph_provenance
                WHERE release_id=%s AND edge_id=%s ORDER BY ordinal
                """,
                (release_id, edge_id),
            ).fetchall()
        if not rows:
            raise HTTPException(status_code=404, detail="Edge provenance not found")
        return {
            "release_id": release_id,
            "edge_id": edge_id,
            "citation_node_ids": list(dict.fromkeys(row[0] for row in rows)),
            "provenance": [
                {"canonical_node_id": row[0], "evidence_text": row[1], "start_offset": row[2], "end_offset": row[3]}
                for row in rows
            ],
        }

    @app.post("/experimental/graph/query", dependencies=[Depends(authorize)])
    def graph_query(request: GraphQueryRequest) -> dict:
        headers = {"X-API-Key": settings.lightrag_api_key} if settings.lightrag_api_key else {}
        with httpx.Client(timeout=120) as client:
            response = client.post(
                f"{settings.lightrag_url.rstrip('/')}/query",
                headers=headers,
                json={
                    "query": request.query,
                    "mode": request.mode,
                    "only_need_context": request.only_need_context,
                    "include_references": True,
                },
            )
            response.raise_for_status()
            payload: Any = response.json()
        references = payload.get("references", []) if isinstance(payload, dict) else []
        citation_ids = []
        for reference in references:
            path = reference.get("file_path", "") if isinstance(reference, dict) else ""
            if path.startswith("canonical://"):
                citation_ids.append(path.removeprefix("canonical://"))
        return {
            "route": "experimental_lightrag",
            "authoritative_graph": False,
            "citation_node_ids": list(dict.fromkeys(citation_ids)),
            "lightrag_response": payload,
        }

    return app
