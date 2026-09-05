from __future__ import annotations

import hashlib
import json
import re
import unicodedata

from .models import GraphProvenance


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def normalized_label(value: str) -> str:
    value = unicodedata.normalize("NFC", value).casefold().strip()
    return re.sub(r"\s+", " ", value)


def semantic_node_id(document_id: str, entity_type: str, label: str) -> str:
    identity = "\0".join((document_id, entity_type.casefold(), normalized_label(label)))
    return f"SEM::{document_id}::{entity_type.upper()}::{sha256_text(identity)}"


def graph_edge_id(
    source_node_id: str,
    relation_type: str,
    target_node_id: str,
    provenance: list[GraphProvenance] | None = None,
) -> str:
    evidence = [
        {
            "canonical_node_id": item.canonical_node_id,
            "evidence_text": item.evidence_text,
            "start_offset": item.start_offset,
            "end_offset": item.end_offset,
        }
        for item in (provenance or [])
    ]
    payload = json.dumps(
        [source_node_id, relation_type, target_node_id, evidence],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return f"EDGE::{sha256_text(payload)}"
