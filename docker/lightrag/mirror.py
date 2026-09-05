from __future__ import annotations

import asyncio
import json
import os
import sys
from functools import partial
from pathlib import Path

from lightrag import LightRAG
from lightrag.llm.ollama import ollama_embed
from lightrag.utils import EmbeddingFunc


async def _unused_llm(*args, **kwargs):
    raise RuntimeError("The mirror importer never calls an LLM")


async def main(path: str) -> None:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    model = os.environ.get("EMBEDDING_MODEL", "bge-m3")
    host = os.environ.get("EMBEDDING_BINDING_HOST", "http://host.docker.internal:11434")
    embedding = EmbeddingFunc(
        embedding_dim=int(os.environ.get("EMBEDDING_DIM", "1024")),
        max_token_size=int(os.environ.get("EMBEDDING_TOKEN_LIMIT", "8192")),
        model_name=model,
        supports_asymmetric=True,
        func=partial(ollama_embed.func, embed_model=model, host=host),
    )
    rag = LightRAG(
        working_dir=os.environ.get("WORKING_DIR", "/app/data/rag_storage"),
        workspace=os.environ["WORKSPACE"],
        llm_model_func=_unused_llm,
        llm_model_name="mirror-no-llm",
        embedding_func=embedding,
        kv_storage=os.environ.get("LIGHTRAG_KV_STORAGE", "PGKVStorage"),
        vector_storage=os.environ.get("LIGHTRAG_VECTOR_STORAGE", "PGVectorStorage"),
        graph_storage=os.environ.get("LIGHTRAG_GRAPH_STORAGE", "PGTableGraphStorage"),
        doc_status_storage=os.environ.get("LIGHTRAG_DOC_STATUS_STORAGE", "PGDocStatusStorage"),
    )
    await rag.initialize_storages()
    try:
        await rag.ainsert_custom_kg(payload)
    finally:
        await rag.finalize_storages()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: mirror.py /data/mirror_payload.json")
    asyncio.run(main(sys.argv[1]))
