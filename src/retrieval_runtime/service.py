from __future__ import annotations

import hashlib
import json
from pathlib import Path

from src.chunking.models import EvidenceBundle
from src.chunking.token_counter import BGETokenCounter, TokenCounter
from src.legal_tree.resolver import LegalTreeResolver
from src.postgres_store.config import PostgresSettings, connect
from src.postgres_store.dense import (
    PostgresArticleRetriever,
    PostgresFineRetriever,
    PostgresVectorSession,
)
from src.retrieval_eval.dense import DenseEncoder
from src.retrieval_eval.evaluator import build_article_scout_bundle
from src.retrieval_eval.models import EvidenceComponent
from src.retrieval_eval.round2 import project_b4_components
from src.retrieval_eval.hybrid import DualGranularityRetriever, HybridConfig
from src.registry.loader import load_registry

from .config import ApplicationConfig, load_application_config
from .models import RuntimeSearchResponse, RuntimeSearchResult
from .repository import ActiveRuntimeCorpus, load_active_runtime_corpus
from .evidence import (
    build_b7_article_scout_bundle,
    build_direct_evidence,
    project_b7_passage_bundle,
)
from .reference import parse_legal_reference, resolve_legal_reference


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _b7_code_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for base in (root / "src/retrieval_runtime", root / "src/retrieval_eval"):
        for path in sorted(base.glob("*.py")):
            digest.update(path.relative_to(root).as_posix().encode("utf-8"))
            digest.update(path.read_bytes())
    return digest.hexdigest()


def _b7_lock_matches(root: Path, config: ApplicationConfig, dataset_id: str, b6_lock_sha: str) -> bool:
    if config.routing_policy == "B6":
        return True
    lock_path = root / "reports/b7/b7_policy_lock.json"
    config_path = root / "configs/b7.yaml"
    if not lock_path.is_file() or not config_path.is_file():
        return False
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    return (
        lock.get("winner") == config.routing_policy
        and lock.get("dataset_id") == dataset_id
        and lock.get("b6_strategy_lock_sha256") == b6_lock_sha
        and lock.get("b7_config_sha256") == _sha256(config_path)
        and lock.get("code_digest") == _b7_code_digest(root)
        and lock.get("winner_runtime_config") == config.model_dump(mode="json")
    )


def application_health(root: str | Path) -> dict:
    root_path = Path(root).resolve()
    config = load_application_config(root_path)
    dsn = PostgresSettings.load(root_path).require("runtime")
    with connect(dsn) as connection:
        row = connection.execute("""
            SELECT dataset_id, status, embedding_model, embedding_revision,
                   strategy_lock_sha256,
                   (SELECT count(*) FROM traffic_rag.retrieval_passage p WHERE p.dataset_id = d.dataset_id),
                   (SELECT count(*) FROM traffic_rag.passage_embedding e
                    WHERE e.dataset_id = d.dataset_id AND e.status = 'READY')
            FROM traffic_rag.retrieval_dataset d
            WHERE status = 'ACTIVE'
        """).fetchall()
    if len(row) != 1:
        return {"ready": False, "error": f"expected one ACTIVE dataset, found {len(row)}"}
    dataset_id, status, model, revision, lock_sha, passage_count, embedding_count = row[0]
    lock_path = root_path / "reports/chunk_ablation/strategy_lock.json"
    lock_match = lock_path.is_file() and _sha256(lock_path) == lock_sha
    b7_lock_match = _b7_lock_matches(root_path, config, dataset_id, lock_sha)
    ready = (
        status == "ACTIVE"
        and model == config.query_encoder.model
        and revision == config.query_encoder.revision
        and passage_count == 1635
        and embedding_count == 1635
        and (lock_match or not config.require_strategy_lock_match)
        and (b7_lock_match or not config.require_b7_policy_lock_match)
    )
    return {
        "ready": ready,
        "backend": config.retrieval_backend,
        "strategy": config.strategy,
        "routing_policy": config.routing_policy,
        "dataset_id": dataset_id,
        "status": status,
        "model": model,
        "model_revision": revision,
        "passage_count": passage_count,
        "embedding_count": embedding_count,
        "strategy_lock_match": lock_match,
        "b7_policy_lock_match": b7_lock_match,
    }


class PostgresB6Application:
    """Default runtime retriever backed by the ACTIVE PostgreSQL release."""

    def __init__(
        self,
        root: str | Path,
        *,
        config: ApplicationConfig | None = None,
        encoder: DenseEncoder | None = None,
        token_counter: TokenCounter | None = None,
        allow_unlocked_policy: bool = False,
        dataset_id: str | None = None,
        allow_building_dataset: bool = False,
        strategy_lock_path: str | Path = "reports/chunk_ablation/strategy_lock.json",
    ):
        self.root = Path(root).resolve()
        self.config = config or load_application_config(self.root)
        settings = PostgresSettings.load(self.root)
        self.dsn = settings.require("runtime")
        self.corpus: ActiveRuntimeCorpus = load_active_runtime_corpus(
            self.root,
            self.dsn,
            require_strategy_lock_match=self.config.require_strategy_lock_match,
            dataset_id=dataset_id,
            allow_building=allow_building_dataset,
            strategy_lock_path=strategy_lock_path,
        )
        if (
            self.config.routing_policy != "B6"
            and self.config.require_b7_policy_lock_match
            and not allow_unlocked_policy
            and not _b7_lock_matches(
                self.root,
                self.config,
                self.corpus.dataset_id,
                _sha256(self.root / "reports/chunk_ablation/strategy_lock.json"),
            )
        ):
            raise ValueError("B7 runtime policy does not match b7_policy_lock.json")
        if (
            self.corpus.model_name != self.config.query_encoder.model
            or self.corpus.model_revision != self.config.query_encoder.revision
        ):
            raise ValueError("Application encoder does not match the ACTIVE PostgreSQL release")
        self.resolvers = {
            document.document_id: LegalTreeResolver(document.nodes)
            for document in self.corpus.documents
        }
        self.registry = load_registry(self.root)
        self.document_titles = {
            document_id: record.title
            for document_id, record in self.registry.documents.items()
        }
        self.article_passages = list(self.corpus.article_passages)
        self.fine_passages = list(self.corpus.fine_passages)
        self.passage_by_id = {
            passage.passage_id: passage
            for passage in (*self.corpus.article_passages, *self.corpus.fine_passages)
        }
        self.fine_by_id = {passage.passage_id: passage for passage in self.corpus.fine_passages}
        self.fine_by_primary_id = {
            passage.primary_node_id: passage for passage in self.corpus.fine_passages
        }
        encoder_config = self.config.query_encoder
        self.encoder = encoder or DenseEncoder(
            encoder_config.model,
            encoder_config.revision,
            max_length=encoder_config.max_length,
            preferred_device=encoder_config.preferred_device,
        )
        if self.encoder.revision != self.corpus.model_revision:
            raise ValueError("Resolved query encoder revision differs from the ACTIVE release")
        self.counter = token_counter or BGETokenCounter(self.corpus.model_revision)
        self.session = PostgresVectorSession(
            dsn=self.dsn,
            dataset_id=self.corpus.dataset_id,
            model_name=self.corpus.model_name,
            model_revision=self.corpus.model_revision,
            encoder=self.encoder,
            allow_building=allow_building_dataset,
        )
        hybrid_config = self.corpus.release_manifest.get("hybrid_b6_config")
        if not hybrid_config:
            self.close()
            raise ValueError("ACTIVE release has no frozen B6 configuration")
        self.retriever = DualGranularityRetriever(
            PostgresArticleRetriever(self.session),
            PostgresFineRetriever(self.session),
            self.article_passages,
            self.fine_passages,
            self.resolvers,
            HybridConfig.from_dict(hybrid_config),
        )
        if self.config.routing_policy == "B7d":
            self.close()
            raise ValueError("B7d is gated until B6 Recall@30 and ordering-failure criteria pass")

    def close(self) -> None:
        session = getattr(self, "session", None)
        if session is not None:
            session.close()
            self.session = None

    def __enter__(self) -> "PostgresB6Application":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def _search_b6(
        self,
        query: str,
        *,
        top_k: int,
        document_ids: list[str] | None = None,
        result_type: str = "passage",
        b7_evidence_projection: bool = False,
    ) -> list[RuntimeSearchResult]:
        hits = self.retriever.search(query, top_k=10)
        results: list[RuntimeSearchResult] = []
        allowed_documents = set(document_ids or [])
        for hit in hits:
            passage = self.passage_by_id[hit.passage_id]
            if allowed_documents and passage.document_id not in allowed_documents:
                continue
            if hit.source_strategy == "B1":
                if b7_evidence_projection:
                    bundle, components = build_b7_article_scout_bundle(
                        passage,
                        hit.evidence_passage_ids,
                        self.fine_by_id,
                        self.resolvers,
                        self.document_titles,
                        self.counter,
                    )
                else:
                    bundle, components = build_article_scout_bundle(
                        passage,
                        hit.evidence_passage_ids,
                        self.fine_by_id,
                        self.resolvers,
                        self.counter,
                    )
            else:
                if b7_evidence_projection:
                    bundle, components = project_b7_passage_bundle(
                        passage,
                        self.resolvers[passage.document_id],
                        self.document_titles[passage.document_id],
                        self.counter,
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
                    components = [EvidenceComponent(
                        node_id=item.node_id,
                        role=item.role,
                        token_sequence_hash=item.token_sequence_hash,
                        token_count=item.token_count,
                    ) for item in project_b4_components(
                        passage, self.resolvers[passage.document_id], self.counter,
                    )]
            results.append(RuntimeSearchResult(
                rank=len(results) + 1,
                score=hit.score,
                source_strategy=hit.source_strategy or passage.strategy,
                passage_id=passage.passage_id,
                primary_node_id=passage.primary_node_id,
                document_id=passage.document_id,
                hierarchy=passage.hierarchy,
                evidence_text=bundle.evidence_text,
                included_node_ids=bundle.included_node_ids,
                context_node_ids=bundle.context_node_ids,
                citation_node_ids=bundle.citation_node_ids,
                evidence_tokens=bundle.token_count,
                result_type=result_type,
                member_node_ids=bundle.citation_node_ids,
                evidence_components=components,
            ))
            if len(results) == top_k:
                break
        return results

    def _response(
        self,
        query: str,
        *,
        route: str,
        resolution_status: str,
        results: list[RuntimeSearchResult],
        parsed_reference=None,
        ambiguity_candidates: list[str] | None = None,
    ) -> RuntimeSearchResponse:
        return RuntimeSearchResponse(
            query=query,
            dataset_id=self.corpus.dataset_id,
            model=self.corpus.model_name,
            model_revision=self.corpus.model_revision,
            routing_policy=self.config.routing_policy,
            route=route,
            resolution_status=resolution_status,
            parsed_reference=parsed_reference,
            ambiguity_candidates=ambiguity_candidates or [],
            results=results,
        )

    def search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        document_ids: list[str] | None = None,
    ) -> RuntimeSearchResponse:
        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("Query must not be empty")
        requested_top_k = top_k or self.config.default_top_k
        if not 1 <= requested_top_k <= self.config.max_top_k:
            raise ValueError(f"top_k must be between 1 and {self.config.max_top_k}")
        policy = self.config.routing_policy
        if policy == "B6":
            return self._response(
                normalized_query,
                route="B6",
                resolution_status="NO_REFERENCE",
                results=self._search_b6(normalized_query, top_k=requested_top_k),
            )

        intent = parse_legal_reference(normalized_query, self.registry)
        resolution = resolve_legal_reference(intent, self.resolvers, document_ids)
        direct_single = policy in {"B7a", "B7c"} and len(resolution.resolved_node_ids) == 1
        direct_multi = policy in {"B7b", "B7c"} and len(resolution.resolved_node_ids) > 1
        if resolution.status == "RESOLVED" and (direct_single or direct_multi):
            document_id = resolution.candidate_document_ids[0]
            direct = build_direct_evidence(
                resolution.resolved_node_ids,
                self.resolvers[document_id],
                self.fine_by_primary_id,
                self.counter,
                document_title=self.document_titles[document_id],
            )
            bundle = direct.bundle
            result = RuntimeSearchResult(
                rank=1,
                score=1.0,
                source_strategy=policy,
                passage_id=direct.passage_id,
                primary_node_id=direct.primary_node.id,
                document_id=direct.document_id,
                hierarchy=direct.primary_node.hierarchy,
                evidence_text=bundle.evidence_text,
                included_node_ids=bundle.included_node_ids,
                context_node_ids=bundle.context_node_ids,
                citation_node_ids=bundle.citation_node_ids,
                evidence_tokens=bundle.token_count,
                result_type="evidence_bundle" if len(direct.member_node_ids) > 1 else "canonical_node",
                member_node_ids=direct.member_node_ids,
                evidence_components=direct.components,
            )
            return self._response(
                normalized_query,
                route="DIRECT_REFERENCE" if direct_single else "DIRECT_MULTI_EVIDENCE",
                resolution_status=resolution.status,
                parsed_reference=intent,
                results=[result],
            )

        if resolution.status == "CONFLICT":
            return self._response(
                normalized_query,
                route="REFERENCE_CONFLICT",
                resolution_status=resolution.status,
                parsed_reference=intent,
                ambiguity_candidates=resolution.candidate_document_ids,
                results=[],
            )
        candidate_scope = resolution.candidate_document_ids or list(document_ids or [])
        suggestion = resolution.status == "AMBIGUOUS"
        results = self._search_b6(
            normalized_query,
            top_k=requested_top_k,
            document_ids=None if suggestion else candidate_scope,
            result_type="suggestion" if suggestion else "passage",
            b7_evidence_projection=True,
        )
        return self._response(
            normalized_query,
            route="AMBIGUOUS_REFERENCE" if suggestion else "B6_FALLBACK",
            resolution_status=resolution.status,
            parsed_reference=intent,
            ambiguity_candidates=resolution.candidate_document_ids if suggestion else [],
            results=results,
        )
