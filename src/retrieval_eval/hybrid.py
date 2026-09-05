from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Protocol

from src.chunking.models import RetrievalPassage
from src.legal_tree.resolver import LegalTreeResolver

from .models import SearchHit


class RankedRetriever(Protocol):
    def search(self, query: str, top_k: int = 10) -> list[SearchHit]: ...


class GuidedFineRetriever(Protocol):
    """Fine retriever that can fetch only the rows B6 actually consumes."""

    def search_guided(
        self,
        query: str,
        article_node_ids: list[str],
        *,
        global_top_n: int,
        leaves_per_article: int,
    ) -> tuple[list[SearchHit], dict[str, list[SearchHit]]]: ...


@dataclass(frozen=True)
class HybridConfig:
    article_top_n: int = 10
    fine_global_top_n: int = 50
    guided_leaves_per_article: int = 3
    rrf_k: int = 60
    article_weight: float = 1.0
    fine_weight: float = 1.0
    max_article_results_at_5: int = 1
    max_article_results_at_10: int = 2
    max_fine_results_per_article: int = 3

    @classmethod
    def from_dict(cls, value: dict) -> "HybridConfig":
        fields = cls.__dataclass_fields__
        return cls(**{key: value[key] for key in fields if key in value})

    def validate(self) -> None:
        if min(self.article_top_n, self.fine_global_top_n, self.guided_leaves_per_article, self.rrf_k) <= 0:
            raise ValueError("B6 rank depths and rrf_k must be positive")
        if self.article_weight <= 0 or self.fine_weight <= 0:
            raise ValueError("B6 RRF weights must be positive")
        if not 0 <= self.max_article_results_at_5 <= 5:
            raise ValueError("Invalid B6 top-5 Article cap")
        if not self.max_article_results_at_5 <= self.max_article_results_at_10 <= 10:
            raise ValueError("Invalid B6 top-10 Article cap")
        if self.max_fine_results_per_article <= 0:
            raise ValueError("Invalid B6 fine-per-Article cap")


@dataclass(frozen=True)
class _Candidate:
    passage: RetrievalPassage
    source_strategy: str
    score: float
    evidence_passage_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class HybridSearchTrace:
    hits: tuple[SearchHit, ...]
    article_hits: tuple[SearchHit, ...]
    fine_global_hits: tuple[SearchHit, ...]
    guided_hits: dict[str, tuple[SearchHit, ...]]


class DualGranularityRetriever:
    """One-shot B1 scout + B4e fine retrieval with deterministic weighted RRF."""

    def __init__(
        self,
        article_retriever: RankedRetriever,
        fine_retriever: RankedRetriever,
        article_passages: list[RetrievalPassage],
        fine_passages: list[RetrievalPassage],
        resolvers: dict[str, LegalTreeResolver],
        config: HybridConfig,
    ):
        config.validate()
        self.article_retriever = article_retriever
        self.fine_retriever = fine_retriever
        self.article_passages = {item.passage_id: item for item in article_passages}
        self.fine_passages = {item.passage_id: item for item in fine_passages}
        self.config = config
        self._fine_article: dict[str, str] = {}
        self._fine_by_article: dict[str, list[str]] = defaultdict(list)
        for passage in fine_passages:
            resolver = resolvers[passage.document_id]
            article = resolver.get_article(passage.primary_node_id)
            if article is None:
                raise ValueError(f"B4e passage has no Article ancestor: {passage.passage_id}")
            self._fine_article[passage.passage_id] = article.id
            self._fine_by_article[article.id].append(passage.passage_id)

    def _rrf(self, rank: int, weight: float) -> float:
        return weight / (self.config.rrf_k + rank)

    def search(self, query: str, top_k: int = 10) -> list[SearchHit]:
        return list(self.search_with_trace(query, top_k=top_k).hits)

    def search_with_trace(self, query: str, top_k: int = 10) -> HybridSearchTrace:
        if top_k != 10:
            raise ValueError("B6 persists exactly ranks 1..10")
        article_hits = self.article_retriever.search(query, self.config.article_top_n)
        article_node_ids = [self.article_passages[hit.passage_id].primary_node_id for hit in article_hits]
        search_guided = getattr(self.fine_retriever, "search_guided", None)
        if search_guided is None:
            fine_hits = self.fine_retriever.search(query, len(self.fine_passages))
            fine_global_hits = fine_hits[: self.config.fine_global_top_n]
            fine_rank = {hit.passage_id: hit.rank for hit in fine_hits}
            guided_hits = {
                article_id: [
                    next(hit for hit in fine_hits if hit.passage_id == passage_id)
                    for passage_id in sorted(
                        self._fine_by_article.get(article_id, ()),
                        key=lambda value: (fine_rank[value], value),
                    )[: self.config.guided_leaves_per_article]
                ]
                for article_id in article_node_ids
            }
        else:
            fine_global_hits, guided_hits = search_guided(
                query,
                article_node_ids,
                global_top_n=self.config.fine_global_top_n,
                leaves_per_article=self.config.guided_leaves_per_article,
            )
        hits = self._fuse_preselected(article_hits, fine_global_hits, guided_hits)
        return HybridSearchTrace(
            hits=tuple(hits),
            article_hits=tuple(article_hits),
            fine_global_hits=tuple(fine_global_hits),
            guided_hits={key: tuple(value) for key, value in guided_hits.items()},
        )

    def search_candidate_pool(self, query: str) -> list[SearchHit]:
        """Return uncapped frozen-B6 candidates for reranker eligibility.

        The persisted B6 top-10 path is unchanged. This diagnostic reuses the
        same RRF equations and canonical-primary dedupe without top-k caps.
        """
        trace = self.search_with_trace(query, top_k=10)
        fine_global_hits = list(trace.fine_global_hits)
        fine_rank = {hit.passage_id: hit.rank for hit in fine_global_hits}
        guided_rank: dict[str, int] = {}
        guided_for_article: dict[str, tuple[str, ...]] = {}
        for article_hit in trace.article_hits:
            article = self.article_passages[article_hit.passage_id]
            descendants = tuple(
                hit.passage_id for hit in trace.guided_hits.get(article.primary_node_id, ())
            )
            guided_for_article[article.primary_node_id] = descendants
            for local_rank, passage_id in enumerate(descendants, start=1):
                rank = self.config.guided_leaves_per_article * (article_hit.rank - 1) + local_rank
                guided_rank.setdefault(passage_id, rank)
        candidates: dict[str, _Candidate] = {}
        for passage_id in set(fine_rank) | set(guided_rank):
            score = 0.0
            if passage_id in fine_rank:
                score += self._rrf(fine_rank[passage_id], self.config.fine_weight)
            if passage_id in guided_rank:
                score += self._rrf(guided_rank[passage_id], self.config.article_weight)
            passage = self.fine_passages[passage_id]
            candidates[passage.primary_node_id] = _Candidate(passage, "B4e", score)
        for hit in trace.article_hits:
            passage = self.article_passages[hit.passage_id]
            evidence_ids = guided_for_article.get(passage.primary_node_id, ())
            score = self._rrf(hit.rank, self.config.article_weight)
            descendant_ranks = [fine_rank[value] for value in evidence_ids if value in fine_rank]
            if descendant_ranks:
                score += self._rrf(min(descendant_ranks), self.config.fine_weight)
            previous = candidates.get(passage.primary_node_id)
            if previous is None:
                candidates[passage.primary_node_id] = _Candidate(passage, "B1", score, evidence_ids)
            elif score > previous.score:
                candidates[passage.primary_node_id] = _Candidate(
                    previous.passage, previous.source_strategy, score, previous.evidence_passage_ids,
                )
        ordered = sorted(
            candidates.values(),
            key=lambda item: (-item.score, 0 if item.source_strategy == "B4e" else 1, item.passage.passage_id),
        )
        return [
            SearchHit(
                passage_id=item.passage.passage_id,
                score=item.score,
                rank=rank,
                source_strategy=item.source_strategy,
                evidence_passage_ids=list(item.evidence_passage_ids),
            )
            for rank, item in enumerate(ordered, start=1)
        ]

    def _fuse(self, article_hits: list[SearchHit], fine_hits: list[SearchHit]) -> list[SearchHit]:
        fine_rank = {hit.passage_id: hit.rank for hit in fine_hits}
        guided_hits = {}
        for article_hit in article_hits:
            article_id = self.article_passages[article_hit.passage_id].primary_node_id
            guided_hits[article_id] = [
                next(hit for hit in fine_hits if hit.passage_id == passage_id)
                for passage_id in sorted(
                    self._fine_by_article.get(article_id, ()),
                    key=lambda value: (fine_rank[value], value),
                )[: self.config.guided_leaves_per_article]
            ]
        return self._fuse_preselected(
            article_hits,
            fine_hits[: self.config.fine_global_top_n],
            guided_hits,
        )

    def _fuse_preselected(
        self,
        article_hits: list[SearchHit],
        fine_global_hits: list[SearchHit],
        guided_hits: dict[str, list[SearchHit]],
    ) -> list[SearchHit]:
        fine_rank = {hit.passage_id: hit.rank for hit in fine_global_hits}
        fine_global = set(fine_rank)
        guided_rank: dict[str, int] = {}
        guided_for_article: dict[str, tuple[str, ...]] = {}
        for article_hit in article_hits:
            article = self.article_passages[article_hit.passage_id]
            descendants = tuple(hit.passage_id for hit in guided_hits.get(article.primary_node_id, ()))
            guided_for_article[article.primary_node_id] = descendants
            for local_rank, passage_id in enumerate(descendants, start=1):
                rank = self.config.guided_leaves_per_article * (article_hit.rank - 1) + local_rank
                guided_rank.setdefault(passage_id, rank)

        candidates: dict[str, _Candidate] = {}
        for passage_id in fine_global | set(guided_rank):
            score = 0.0
            if passage_id in fine_global:
                score += self._rrf(fine_rank[passage_id], self.config.fine_weight)
            if passage_id in guided_rank:
                score += self._rrf(guided_rank[passage_id], self.config.article_weight)
            passage = self.fine_passages[passage_id]
            candidates[passage.primary_node_id] = _Candidate(passage, "B4e", score)

        for hit in article_hits:
            passage = self.article_passages[hit.passage_id]
            evidence_ids = guided_for_article.get(passage.primary_node_id, ())
            score = self._rrf(hit.rank, self.config.article_weight)
            global_descendant_ranks = [fine_rank[value] for value in evidence_ids if value in fine_global]
            if global_descendant_ranks:
                score += self._rrf(min(global_descendant_ranks), self.config.fine_weight)
            previous = candidates.get(passage.primary_node_id)
            if previous is None:
                candidates[passage.primary_node_id] = _Candidate(passage, "B1", score, evidence_ids)
            elif score > previous.score:
                # Canonical-primary dedupe keeps the B4e representation but the stronger score.
                candidates[passage.primary_node_id] = _Candidate(
                    previous.passage, previous.source_strategy, score, previous.evidence_passage_ids
                )

        ordered = sorted(
            candidates.values(),
            key=lambda item: (-item.score, 0 if item.source_strategy == "B4e" else 1, item.passage.passage_id),
        )
        selected: list[_Candidate] = []
        selected_ids: set[str] = set()
        fine_counts: dict[str, int] = defaultdict(int)

        def take(limit: int, article_cap: int) -> None:
            article_count = sum(item.source_strategy == "B1" for item in selected)
            for candidate in ordered:
                if len(selected) >= limit:
                    break
                if candidate.passage.primary_node_id in selected_ids:
                    continue
                if candidate.source_strategy == "B1":
                    if article_count >= article_cap:
                        continue
                else:
                    article_id = self._fine_article[candidate.passage.passage_id]
                    if fine_counts[article_id] >= self.config.max_fine_results_per_article:
                        continue
                selected.append(candidate)
                selected_ids.add(candidate.passage.primary_node_id)
                if candidate.source_strategy == "B1":
                    article_count += 1
                else:
                    fine_counts[self._fine_article[candidate.passage.passage_id]] += 1

        take(5, self.config.max_article_results_at_5)
        take(10, self.config.max_article_results_at_10)
        if len(selected) != 10:
            raise ValueError(f"B6 produced only {len(selected)}/10 candidates")
        return [
            SearchHit(
                passage_id=item.passage.passage_id,
                score=item.score,
                rank=rank,
                source_strategy=item.source_strategy,
                evidence_passage_ids=list(item.evidence_passage_ids),
            )
            for rank, item in enumerate(selected, start=1)
        ]

    def search_many(self, queries: list[str], *, top_k: int = 10, batch_size: int = 8) -> list[list[SearchHit]]:
        if top_k != 10:
            raise ValueError("B6 persists exactly ranks 1..10")
        if getattr(self.fine_retriever, "search_guided", None) is not None:
            return [self.search(query, top_k=top_k) for query in queries]
        article_many = getattr(self.article_retriever, "search_many", None)
        fine_many = getattr(self.fine_retriever, "search_many", None)
        if article_many is None or fine_many is None:
            return [self.search(query, top_k=top_k) for query in queries]
        article_lists = article_many(queries, top_k=self.config.article_top_n, batch_size=batch_size)
        # DenseEncoder query cache makes the second call reuse the exact same query matrix.
        fine_lists = fine_many(queries, top_k=len(self.fine_passages), batch_size=batch_size)
        return [self._fuse(article, fine) for article, fine in zip(article_lists, fine_lists)]
