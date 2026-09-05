from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from src.chunking.token_counter import RegexTokenCounter
from src.graph.config import GraphSettings
from src.graph.identity import graph_edge_id, semantic_node_id
from src.graph.mirror import build_lightrag_payload
from src.graph.models import (
    ArticleExtraction,
    ExtractedEntity,
    ExtractedRelationship,
    ExtractionEvidence,
    GraphEdge,
    GraphNode,
    GraphProvenance,
)
from src.graph.pilot import (
    is_quota_failure,
    run_full_extraction,
    run_pilot_quota_recovery,
    select_pilot_articles,
)
from src.graph.projection import project_articles
from src.graph.providers import ProviderResult, ProviderSchemaError, provider_from_name
from src.graph.review import select_provider
from src.graph.repository import _executemany, validate_release_transition
from src.graph.semantic import build_semantic_graph
from src.graph.structural import build_structural_graph
from src.graph.validation import validate_article_sources, validate_graph
from src.parser.io import read_json, write_json, write_jsonl


ROOT = Path(__file__).resolve().parents[2]


def test_psycopg_batch_execution_uses_cursor_api():
    calls = []

    class FakeCursor:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def executemany(self, query, params):
            calls.append((query, params))

    class FakeConnection:
        def cursor(self):
            return FakeCursor()

    params = [("kg-v001", "node-1")]
    _executemany(FakeConnection(), "INSERT INTO example VALUES (%s, %s)", params)
    assert calls == [("INSERT INTO example VALUES (%s, %s)", params)]

    _executemany(FakeConnection(), "INSERT INTO example VALUES (%s, %s)", [])
    assert len(calls) == 1


def test_provider_keys_load_from_lightrag_env_without_leaking(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    (tmp_path / ".env.lightrag").write_text(
        "OPENAI_API_KEY=openai-test-secret\nGEMINI_API_KEY=gemini-test-secret\n",
        encoding="utf-8",
    )

    settings = GraphSettings.load(tmp_path)
    assert settings.openai_api_key == "openai-test-secret"
    assert settings.gemini_api_key == "gemini-test-secret"
    assert "test-secret" not in str(settings.redacted())

    provider = provider_from_name("gemini", api_key=settings.gemini_api_key)
    assert getattr(provider, "api_key") == "gemini-test-secret"


def test_article_projection_is_complete_deterministic_and_clean():
    first = project_articles(ROOT, RegexTokenCounter())
    second = project_articles(ROOT, RegexTokenCounter())
    assert len(first) == 175
    assert Counter(item.canonical_document_id for item in first) == {
        "LAW_35_2024": 86,
        "LAW_36_2024": 89,
    }
    assert [item.content_hash for item in first] == [item.content_hash for item in second]
    assert validate_article_sources(first) == []
    assert all(f"[NODE_ID={item.article_node_id} TYPE=article]" in item.extraction_text for item in first)
    assert all("thongtinchinhphu@chinhphu.vn" not in item.text.casefold() for item in first)


def test_structural_graph_has_frozen_counts_and_direction():
    nodes, edges, _ = build_structural_graph(ROOT, include_references=False)
    counts = Counter(edge.relation_type for edge in edges)
    assert sum(node.node_kind == "canonical" for node in nodes) == 1850
    assert sum(node.node_kind == "authority" for node in nodes) == 1
    assert counts == {"CONTAINS": 1848, "NEXT": 1474, "ISSUED_BY": 2}
    assert validate_graph(nodes, edges) == []
    contains = next(
        edge for edge in edges
        if edge.relation_type == "CONTAINS" and edge.target_node_id == "LAW_35_2024__A1"
    )
    assert contains.source_node_id != contains.target_node_id
    assert contains.target_node_id == "LAW_35_2024__A1"


def test_reference_extraction_never_creates_dangling_edges():
    nodes, edges, findings = build_structural_graph(ROOT, include_references=True)
    assert findings
    assert any(item.status == "RESOLVED" for item in findings)
    assert any(edge.relation_type == "REFERENCES" for edge in edges)
    assert validate_graph(nodes, edges) == []


def test_semantic_identity_is_document_scoped():
    left = semantic_node_id("LAW_35_2024", "Actor", "Người lái xe")
    right = semantic_node_id("LAW_36_2024", "Actor", "Người lái xe")
    assert left != right
    assert left == semantic_node_id("LAW_35_2024", "Actor", "  NGƯỜI   LÁI XE ")


def test_semantic_build_requires_exact_canonical_provenance():
    source = project_articles(ROOT, RegexTokenCounter())[0]
    canonical_node_id = source.source_node_ids[0]
    evidence = source.node_texts[canonical_node_id][:20]
    extraction = ArticleExtraction(
        entities=[
            ExtractedEntity(
                local_key="actor_1",
                label="Người tham gia giao thông",
                entity_type="Actor",
                description="Chủ thể được luật điều chỉnh",
                aliases=[],
                provenance=[
                    ExtractionEvidence(
                        canonical_node_id=canonical_node_id,
                        evidence_text=evidence,
                    )
                ],
            )
        ],
        relationships=[
            ExtractedRelationship(
                source_ref=canonical_node_id,
                target_ref="actor_1",
                relation_type="APPLIES_TO",
                description="Phạm vi áp dụng",
                confidence=0.9,
                provenance=[
                    ExtractionEvidence(
                        canonical_node_id=canonical_node_id,
                        evidence_text=evidence,
                    )
                ],
            )
        ],
    )
    built = build_semantic_graph(source, extraction, extraction_version="test-v1")
    assert len(built.nodes) == 1
    assert len(built.edges) == 1
    assert built.rejected == ()
    assert built.edges[0].provenance[0].evidence_text == evidence
    assert built.nodes[0].properties["provenance"][0]["evidence_text"] == evidence

    bad = extraction.model_copy(deep=True)
    bad.relationships[0].provenance[0].evidence_text = "không tồn tại trong văn bản"
    rejected = build_semantic_graph(source, bad, extraction_version="test-v1")
    assert not rejected.edges
    assert any(item["reason"] == "EVIDENCE_NOT_IN_CANONICAL_TEXT" for item in rejected.rejected)


def test_other_relationship_is_audited_but_not_released():
    source = next(
        item for item in project_articles(ROOT, RegexTokenCounter())
        if len(item.source_node_ids) > 1
    )
    canonical_node_id = source.source_node_ids[0]
    evidence = source.node_texts[canonical_node_id][:20]
    extraction = ArticleExtraction(
        entities=[],
        relationships=[
            ExtractedRelationship(
                source_ref=canonical_node_id,
                target_ref=source.source_node_ids[1],
                relation_type="OTHER",
                description="Quan hệ ngoài ontology",
                confidence=0.5,
                provenance=[
                    ExtractionEvidence(
                        canonical_node_id=canonical_node_id,
                        evidence_text=evidence,
                    )
                ],
            )
        ]
    )
    built = build_semantic_graph(source, extraction, extraction_version="test-v1")
    assert built.edges == ()
    assert built.rejected[0]["reason"] == "OTHER_RELATION"


def test_paid_extraction_retries_schema_only_and_preserves_raw(tmp_path):
    source = project_articles(ROOT, RegexTokenCounter())[0]

    class FakeProvider:
        name = "fake"
        model = "fake-v1"

        def __init__(self):
            self.calls = 0

        def extract(self, prompt):
            self.calls += 1
            if self.calls == 1:
                raise ProviderSchemaError(
                    "invalid schema",
                    raw={"output": "bad"},
                    metadata={"input_tokens": 2, "output_tokens": 1},
                )
            return ProviderResult(
                payload=ArticleExtraction(entities=[], relationships=[]),
                raw={"output": "ok"},
                metadata={
                    "input_tokens": 2,
                    "output_tokens": 1,
                    "elapsed_seconds": 0.1,
                },
            )

    provider = FakeProvider()
    manifest = run_full_extraction(provider, [source], tmp_path, max_workers=1)
    assert provider.calls == 2
    assert manifest["failure_count"] == 0
    assert (tmp_path / "raw" / f"{source.article_node_id}.schema_attempt_1.json").is_file()

    class AuthFailureProvider(FakeProvider):
        def extract(self, prompt):
            self.calls += 1
            raise PermissionError("invalid API key")

    auth = AuthFailureProvider()
    failed = run_full_extraction(auth, [source], tmp_path / "auth", max_workers=1)
    assert auth.calls == 1
    assert failed["failure_count"] == 1


def test_gemini_quota_recovery_retries_only_transient_failures(tmp_path):
    selected = select_pilot_articles(project_articles(ROOT, RegexTokenCounter()))
    base = tmp_path / "pilot" / "gemini"
    sample = tmp_path / "pilot_sample.json"
    write_json(
        sample,
        {
            "schema_version": "1.0.0",
            "article_node_ids": [item.article_node_id for item in selected],
            "source_hashes": {
                item.article_node_id: item.content_hash for item in selected
            },
        },
    )
    write_json(
        base / "pilot_manifest.json",
        {
            "schema_version": "1.0.0",
            "run_kind": "pilot",
            "provider": "gemini",
            "model": "gemini-test",
            "article_count": 15,
            "failure_count": 5,
            "input_tokens": 15,
            "output_tokens": 15,
            "elapsed_seconds": 1.5,
            "estimated_cost_usd": 0.01,
            "max_async_llm": 2,
            "system_prompt_sha256": "prompt",
            "output_schema_sha256": "schema",
        },
    )
    write_jsonl(
        base / "extractions.jsonl",
        [
            {
                "article_node_id": item.article_node_id,
                "provider": "gemini",
                "model": "gemini-test",
            }
            for item in selected[:15]
        ],
    )
    for name in ("semantic_nodes.jsonl", "semantic_edges.jsonl", "rejected.jsonl"):
        write_jsonl(base / name, [])
    schema_failure = {
        "article_node_id": selected[15].article_node_id,
        "error": "invalid relation IS_A",
        "error_type": "ProviderSchemaError",
        "schema_attempts": 2,
    }
    quota_failures = [
        {
            "article_node_id": item.article_node_id,
            "error": "429 RESOURCE_EXHAUSTED: quota exceeded",
            "error_type": "ClientError",
            "schema_attempts": 0,
        }
        for item in selected[16:]
    ]
    write_json(base / "failures.json", [schema_failure, *quota_failures])
    assert not is_quota_failure(schema_failure)
    assert all(is_quota_failure(item) for item in quota_failures)

    class RecoveryProvider:
        name = "gemini"
        model = "gemini-test"

        def __init__(self):
            self.calls = []

        def extract(self, prompt):
            article_id = next(
                item.article_node_id
                for item in selected[16:]
                if f"[NODE_ID={item.article_node_id} TYPE=" in prompt
            )
            self.calls.append(article_id)
            return ProviderResult(
                payload=ArticleExtraction(entities=[], relationships=[]),
                raw={"article_node_id": article_id},
                metadata={
                    "input_tokens": 1,
                    "output_tokens": 1,
                    "elapsed_seconds": 0.1,
                },
            )

    provider = RecoveryProvider()
    result = run_pilot_quota_recovery(
        provider,
        selected,
        base,
        "quota-v001",
        sample_path=sample,
    )
    assert provider.calls == [item.article_node_id for item in selected[16:]]
    assert result["recovery"]["article_count"] == 4
    assert result["reconciled"]["article_count"] == 19
    assert result["reconciled"]["failure_count"] == 1
    assert result["reconciled"]["schema_failure_count"] == 1
    reconciled_failures = read_json(
        base / "recovery" / "quota-v001" / "reconciled" / "failures.json"
    )
    assert reconciled_failures == [schema_failure]


def test_lightrag_mirror_aggregates_directed_multi_edges():
    source = project_articles(ROOT, RegexTokenCounter())[0]
    left = GraphNode(
        graph_node_id=source.article_node_id,
        node_kind="canonical",
        node_type="article",
        label="Điều",
        document_id=source.canonical_document_id,
        canonical_node_id=source.article_node_id,
    )
    right = GraphNode(
        graph_node_id="SEM::X",
        node_kind="semantic",
        node_type="Action",
        label="Hành động",
        document_id=source.canonical_document_id,
        properties={"source_node_ids": [source.article_node_id]},
    )
    provenance = GraphProvenance(
        canonical_node_id=source.article_node_id,
        evidence_text=source.node_texts[source.article_node_id][:5],
        start_offset=0,
        end_offset=5,
    )
    edges = [
        GraphEdge(
            edge_id=graph_edge_id(left.graph_node_id, relation, right.graph_node_id, [provenance]),
            source_node_id=left.graph_node_id,
            target_node_id=right.graph_node_id,
            relation_type=relation,
            provenance=[provenance],
        )
        for relation in ("REQUIRES", "PROHIBITS")
    ]
    payload, manifest = build_lightrag_payload([source], [left, right], edges)
    assert len(payload["relationships"]) == 1
    assert manifest["directed_edge_count"] == 2
    assert manifest["mirrored_undirected_edge_count"] == 1
    assert all("\0" not in key for key in manifest["aggregated_edge_map"])
    assert "REQUIRES" in payload["relationships"][0]["description"]
    assert "PROHIBITS" in payload["relationships"][0]["description"]


def test_pilot_selection_is_deterministic_and_balanced():
    sources = project_articles(ROOT, RegexTokenCounter())
    first = select_pilot_articles(sources)
    second = select_pilot_articles(sources)
    assert [item.article_node_id for item in first] == [item.article_node_id for item in second]
    assert Counter(item.canonical_document_id for item in first) == {
        "LAW_35_2024": 10,
        "LAW_36_2024": 10,
    }


def test_model_selection_uses_frozen_gates_cost_and_openai_tie_break():
    config = {
        "selection": {
            "canonical_mapping_accuracy": 1.0,
            "source_attribution_accuracy": 1.0,
            "entity_precision_min": 0.95,
            "relationship_precision_min": 0.95,
            "relationship_direction_accuracy_min": 0.95,
            "missing_important_relation_rate_max": 0.10,
            "meaningful_relationship_precision_delta": 0.02,
        }
    }
    base = {
        "schema_failure_count": 0,
        "benchmark_issue_count": 0,
        "canonical_mapping_accuracy": 1.0,
        "source_attribution_accuracy": 1.0,
        "entity_precision": 0.97,
        "relationship_precision": 0.97,
        "relationship_direction_accuracy": 0.98,
        "missing_important_relation_rate": 0.05,
        "mean_latency_seconds": 1.0,
    }
    openai = {**base, "provider": "openai", "cost_per_accepted_edge": 0.01}
    gemini = {**base, "provider": "gemini", "cost_per_accepted_edge": 0.02}
    assert select_provider([gemini, openai], config)["winner"] == "openai"

    gemini["relationship_precision"] = 1.0
    assert select_provider([openai, gemini], config)["winner"] == "gemini"


def test_release_state_machine_rejects_unsafe_activation():
    validate_release_transition("BUILDING", "VALIDATING")
    validate_release_transition("VALIDATING", "READY")
    validate_release_transition("READY", "ACTIVE")
    validate_release_transition("ACTIVE", "RETIRED")
    validate_release_transition("RETIRED", "ACTIVE")
    with pytest.raises(ValueError):
        validate_release_transition("BUILDING", "ACTIVE")
    with pytest.raises(ValueError):
        validate_release_transition("ACTIVE", "BUILDING")


def test_graph_database_and_lightrag_are_isolated_and_pinned():
    migration = (ROOT / "sql/graph_migrations/0001_legal_graph.sql").read_text(encoding="utf-8")
    compose = (ROOT / "docker-compose.lightrag.yml").read_text(encoding="utf-8")
    env_example = (ROOT / ".env.lightrag.example").read_text(encoding="utf-8")
    assert "CREATE TABLE IF NOT EXISTS legal_graph_release" in migration
    assert "traffic_rag.retrieval_dataset" not in migration
    assert "86e10c4aea9b65f4fcf1d769791d7a3ba4b560d7" in compose
    assert "traffic-rag-lightrag:1.5.7-86e10c4" in compose
    assert "POSTGRES_VECTOR_INDEX_TYPE=HNSW" in env_example
    assert "POSTGRES_DATABASE=traffic_rag_lightrag" in env_example
