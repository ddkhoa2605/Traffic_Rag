from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter

from src.chunking.models import RetrievalPassage

from .models import SearchHit


def bm25_tokenize(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFC", text).casefold()
    return re.findall(r"[^\W_]+", normalized, flags=re.UNICODE)


class BM25Retriever:
    def __init__(self, passages: list[RetrievalPassage], *, k1: float = 1.5, b: float = 0.75):
        self.passages = passages
        self.k1 = k1
        self.b = b
        self.documents = [bm25_tokenize(item.index_text) for item in passages]
        self.lengths = [len(tokens) for tokens in self.documents]
        self.avgdl = sum(self.lengths) / len(self.lengths) if self.lengths else 0.0
        document_frequency: Counter[str] = Counter()
        self.term_frequencies: list[Counter[str]] = []
        for tokens in self.documents:
            frequencies = Counter(tokens)
            self.term_frequencies.append(frequencies)
            document_frequency.update(frequencies.keys())
        count = len(self.documents)
        self.idf = {
            term: math.log(1.0 + (count - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequency.items()
        }

    def search(self, query: str, top_k: int = 10) -> list[SearchHit]:
        query_terms = bm25_tokenize(query)
        scores = []
        for index, (frequencies, length) in enumerate(zip(self.term_frequencies, self.lengths)):
            score = 0.0
            for term in query_terms:
                frequency = frequencies.get(term, 0)
                if not frequency:
                    continue
                denominator = frequency + self.k1 * (1 - self.b + self.b * length / (self.avgdl or 1.0))
                score += self.idf.get(term, 0.0) * (frequency * (self.k1 + 1)) / denominator
            scores.append((score, self.passages[index].passage_id))
        ranked = sorted(scores, key=lambda item: (-item[0], item[1]))[:top_k]
        return [SearchHit(passage_id=passage_id, score=float(score), rank=rank) for rank, (score, passage_id) in enumerate(ranked, start=1)]

