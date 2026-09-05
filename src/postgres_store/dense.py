from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.retrieval_eval.models import SearchHit

from .artifacts import DIMENSIONS, NORM_TOLERANCE
from .config import connect


def validate_query_vector(vector: np.ndarray) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float32)
    if value.shape != (DIMENSIONS,):
        raise ValueError(f"Query embedding must have shape ({DIMENSIONS},), found {value.shape}")
    if not np.isfinite(value).all():
        raise ValueError("Query embedding contains non-finite values")
    norm = float(np.linalg.norm(value))
    if abs(norm - 1.0) > NORM_TOLERANCE:
        raise ValueError(f"Query embedding must be normalized; norm={norm}")
    return value


@dataclass
class PostgresVectorSession:
    dsn: str
    dataset_id: str
    model_name: str
    model_revision: str
    encoder: object
    allow_building: bool = False

    def __post_init__(self) -> None:
        self.connection = connect(self.dsn)
        status = self.connection.execute(
            "SELECT status FROM traffic_rag.retrieval_dataset WHERE dataset_id = %s",
            (self.dataset_id,),
        ).fetchone()
        allowed = {"ACTIVE", "BUILDING"} if self.allow_building else {"ACTIVE"}
        if not status or status[0] not in allowed:
            self.connection.close()
            raise ValueError(f"Dataset {self.dataset_id} is not in an allowed state: {status}")
        self._query_cache: dict[str, np.ndarray] = {}

    def query_vector(self, query: str) -> np.ndarray:
        if query not in self._query_cache:
            matrix = self.encoder.encode_queries([query], batch_size=1)
            self._query_cache[query] = validate_query_vector(matrix[0])
        return self._query_cache[query]

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "PostgresVectorSession":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def global_search(self, query: str, strategy: str, top_k: int) -> list[SearchHit]:
        vector = self.query_vector(query)
        rows = self.connection.execute("""
            WITH query_vector AS (SELECT %s::vector(1024) AS value)
            SELECT p.passage_id, 1.0 - (e.embedding <=> q.value) AS score
            FROM traffic_rag.retrieval_passage p
            JOIN traffic_rag.passage_embedding e
              ON e.dataset_id = p.dataset_id AND e.passage_id = p.passage_id
            CROSS JOIN query_vector q
            WHERE p.dataset_id = %s
              AND p.strategy = %s
              AND e.model_name = %s
              AND e.model_revision = %s
              AND e.status = 'READY'
            ORDER BY e.embedding <=> q.value, p.passage_id
            LIMIT %s
        """, (
            vector, self.dataset_id, strategy, self.model_name,
            self.model_revision, top_k,
        )).fetchall()
        return [
            SearchHit(passage_id=row[0], score=float(row[1]), rank=rank)
            for rank, row in enumerate(rows, start=1)
        ]

    def guided_search(
        self,
        query: str,
        article_node_ids: list[str],
        leaves_per_article: int,
    ) -> dict[str, list[SearchHit]]:
        if not article_node_ids:
            return {}
        vector = self.query_vector(query)
        rows = self.connection.execute("""
            WITH query_vector AS (SELECT %s::vector(1024) AS value),
            requested AS (
                SELECT article_node_id, article_order
                FROM unnest(%s::text[]) WITH ORDINALITY AS item(article_node_id, article_order)
            )
            SELECT requested.article_node_id, leaf.passage_id, 1.0 - leaf.distance AS score
            FROM requested
            CROSS JOIN query_vector q
            CROSS JOIN LATERAL (
                SELECT p.passage_id, e.embedding <=> q.value AS distance
                FROM traffic_rag.retrieval_passage p
                JOIN traffic_rag.passage_embedding e
                  ON e.dataset_id = p.dataset_id AND e.passage_id = p.passage_id
                WHERE p.dataset_id = %s
                  AND p.strategy = 'B4e_document_article_clause_point'
                  AND p.article_node_id = requested.article_node_id
                  AND e.model_name = %s
                  AND e.model_revision = %s
                  AND e.status = 'READY'
                ORDER BY e.embedding <=> q.value, p.passage_id
                LIMIT %s
            ) AS leaf
            ORDER BY requested.article_order, leaf.distance, leaf.passage_id
        """, (
            vector, article_node_ids, self.dataset_id, self.model_name,
            self.model_revision, leaves_per_article,
        )).fetchall()
        grouped: dict[str, list[SearchHit]] = {article_id: [] for article_id in article_node_ids}
        for article_id, passage_id, score in rows:
            values = grouped[article_id]
            values.append(SearchHit(passage_id=passage_id, score=float(score), rank=len(values) + 1))
        missing = [article_id for article_id, hits in grouped.items() if not hits]
        if missing:
            raise ValueError(f"Article scouts have no B4e descendants: {missing}")
        return grouped


class PostgresArticleRetriever:
    def __init__(self, session: PostgresVectorSession):
        self.session = session

    def search(self, query: str, top_k: int = 10) -> list[SearchHit]:
        return self.session.global_search(query, "B1_article", top_k)


class PostgresFineRetriever:
    def __init__(self, session: PostgresVectorSession):
        self.session = session

    def search(self, query: str, top_k: int = 10) -> list[SearchHit]:
        return self.session.global_search(query, "B4e_document_article_clause_point", top_k)

    def search_guided(
        self,
        query: str,
        article_node_ids: list[str],
        *,
        global_top_n: int,
        leaves_per_article: int,
    ) -> tuple[list[SearchHit], dict[str, list[SearchHit]]]:
        global_hits = self.search(query, global_top_n)
        guided = self.session.guided_search(query, article_node_ids, leaves_per_article)
        return global_hits, guided
