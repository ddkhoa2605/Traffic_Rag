from __future__ import annotations

from dataclasses import dataclass

from src.legal_tree.resolver import LegalTreeResolver

from .common import passage_content_hash
from .models import RetrievalPassage
from .token_counter import TokenCounter


@dataclass(frozen=True)
class PassageIssue:
    passage_id: str
    code: str
    message: str


def validate_passages(
    passages: list[RetrievalPassage],
    resolver: LegalTreeResolver,
    counter: TokenCounter,
) -> list[PassageIssue]:
    issues: list[PassageIssue] = []
    seen: set[str] = set()
    for passage in passages:
        if passage.passage_id in seen:
            issues.append(PassageIssue(passage.passage_id, "DUPLICATE_PASSAGE_ID", "Passage ID is not unique"))
        seen.add(passage.passage_id)
        node_ids = passage.source_node_ids + passage.index_node_ids + passage.context_node_ids + passage.citation_node_ids
        resolved = []
        for node_id in dict.fromkeys(node_ids):
            try:
                node = resolver.get_node(node_id)
                resolved.append(node)
                if node.document_id != passage.document_id:
                    issues.append(PassageIssue(passage.passage_id, "CROSS_DOCUMENT_MAPPING", node_id))
                if node.source is None or not node.source.blocks:
                    issues.append(PassageIssue(passage.passage_id, "MISSING_PROVENANCE", node_id))
            except KeyError:
                issues.append(PassageIssue(passage.passage_id, "UNKNOWN_NODE_ID", node_id))
        index_count = counter.count(passage.index_text)
        evidence_count = counter.count(passage.evidence_text)
        if index_count != passage.token_count_index:
            issues.append(PassageIssue(passage.passage_id, "INDEX_TOKEN_COUNT_MISMATCH", f"{passage.token_count_index} != {index_count}"))
        if evidence_count != passage.token_count_evidence:
            issues.append(PassageIssue(passage.passage_id, "EVIDENCE_TOKEN_COUNT_MISMATCH", f"{passage.token_count_evidence} != {evidence_count}"))
        if index_count > counter.max_tokens:
            issues.append(PassageIssue(passage.passage_id, "INDEX_TOO_LONG", str(index_count)))
        source_nodes = []
        for node_id in passage.source_node_ids:
            try:
                source_nodes.append(resolver.get_node(node_id))
            except KeyError:
                pass
        expected_hash = passage_content_hash(passage.strategy, passage.index_text, passage.evidence_text, source_nodes)
        if expected_hash != passage.content_hash:
            issues.append(PassageIssue(passage.passage_id, "CONTENT_HASH_MISMATCH", "Passage content hash is stale"))
    return issues


def require_valid_passages(passages, resolver, counter) -> None:
    issues = validate_passages(passages, resolver, counter)
    if issues:
        summary = "; ".join(f"{issue.passage_id}:{issue.code}" for issue in issues[:20])
        raise ValueError(f"Passage integrity failed ({len(issues)} issues): {summary}")
