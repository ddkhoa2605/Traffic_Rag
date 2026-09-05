from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from src.chunking.models import RetrievalPassage

from .models import SearchHit


class DenseEncoder:
    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        revision: str = "main",
        *,
        max_length: int = 8192,
        preferred_device: str = "cuda",
    ):
        import torch
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.requested_revision = revision
        self.max_length = max_length
        self.device = preferred_device if preferred_device == "cuda" and torch.cuda.is_available() else "cpu"
        self.dtype = "float16" if self.device == "cuda" else "float32"
        self.preflight_done = False
        self._query_cache: dict[tuple[tuple[str, ...], int], np.ndarray] = {}
        self.model = SentenceTransformer(model_name, revision=revision, device=self.device)
        self.model.max_seq_length = max_length
        if self.device == "cuda":
            self.model.half()
        modules = getattr(self.model, "_modules", {})
        first = next(iter(modules.values()), None)
        tokenizer = getattr(first, "tokenizer", None)
        self.revision = getattr(tokenizer, "init_kwargs", {}).get("_commit_hash") or revision

    def preflight(self, longest_text: str) -> None:
        if self.preflight_done:
            return
        if self.device == "cpu":
            self.preflight_done = True
            return
        try:
            self.encode([longest_text], batch_size=1)
        except RuntimeError as exc:
            if self.device != "cuda" or "out of memory" not in str(exc).casefold():
                raise
            import torch
            from sentence_transformers import SentenceTransformer

            del self.model
            torch.cuda.empty_cache()
            self.device = "cpu"
            self.dtype = "float32"
            self.model = SentenceTransformer(self.model_name, revision=self.requested_revision, device="cpu")
            self.model.max_seq_length = self.max_length
        self.preflight_done = True

    def encode(self, texts: list[str], *, batch_size: int) -> np.ndarray:
        return np.asarray(self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        ), dtype=np.float32)

    def encode_queries(self, texts: list[str], *, batch_size: int) -> np.ndarray:
        key = (tuple(texts), batch_size)
        if key not in self._query_cache:
            self._query_cache[key] = self.encode(texts, batch_size=batch_size)
        return self._query_cache[key]


class DenseRetriever:
    def __init__(
        self,
        passages: list[RetrievalPassage],
        encoder: DenseEncoder,
        *,
        cache_dir: Path,
        passage_manifest_hash: str,
        batch_size: int = 1,
    ):
        self.passages = passages
        self.encoder = encoder
        key = hashlib.sha256(f"{encoder.model_name}:{encoder.revision}:{passage_manifest_hash}:{encoder.device}:{encoder.dtype}".encode()).hexdigest()[:20]
        cache_dir.mkdir(parents=True, exist_ok=True)
        matrix_path = cache_dir / f"dense_{key}.npy"
        meta_path = cache_dir / f"dense_{key}.json"
        if matrix_path.is_file() and meta_path.is_file():
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
            expected_ids = [item.passage_id for item in passages]
            if metadata.get("passage_ids") != expected_ids:
                raise ValueError("Dense cache passage IDs do not match manifest")
            self.matrix = np.load(matrix_path)
        else:
            self.matrix = encoder.encode([item.index_text for item in passages], batch_size=batch_size)
            np.save(matrix_path, self.matrix)
            meta_path.write_text(json.dumps({
                "model": encoder.model_name,
                "revision": encoder.revision,
                "device": encoder.device,
                "dtype": encoder.dtype,
                "passage_manifest_hash": passage_manifest_hash,
                "passage_ids": [item.passage_id for item in passages],
            }, ensure_ascii=False, indent=2), encoding="utf-8")

    def search(self, query: str, top_k: int = 10) -> list[SearchHit]:
        return self.search_many([query], top_k=top_k, batch_size=1)[0]

    def search_many(self, queries: list[str], *, top_k: int = 10, batch_size: int = 8) -> list[list[SearchHit]]:
        vectors = self.encoder.encode_queries(queries, batch_size=batch_size)
        score_matrix = vectors @ self.matrix.T
        output = []
        for scores in score_matrix:
            ranked = sorted(
                ((float(score), self.passages[index].passage_id) for index, score in enumerate(scores)),
                key=lambda item: (-item[0], item[1]),
            )[:top_k]
            output.append([
                SearchHit(passage_id=passage_id, score=score, rank=rank)
                for rank, (score, passage_id) in enumerate(ranked, start=1)
            ])
        return output
