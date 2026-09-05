from __future__ import annotations

from .models import GraphSourceDocument


SYSTEM_PROMPT = """Bạn trích xuất knowledge graph từ văn bản luật Việt Nam.
Chỉ dùng thông tin có trong văn bản. Không suy diễn nghĩa vụ, cấm đoán hoặc ngoại lệ.
Giữ đúng hướng quan hệ. Mọi entity và relationship phải có ít nhất một provenance
gồm canonical_node_id và evidence_text trích nguyên văn. canonical_node_id phải là
NODE_ID xuất hiện trong input. Relationship chỉ được dùng enum trong JSON schema.
Canonical provision có thể được dùng trực tiếp làm source_ref/target_ref; semantic
entity phải được tham chiếu bằng local_key. Không đưa marker kỹ thuật thành entity.
"""


def extraction_prompt(source: GraphSourceDocument) -> str:
    return (
        f"Document: {source.canonical_document_id}\n"
        f"Article: {source.article_node_id}\n\n"
        f"{source.extraction_text}"
    )
