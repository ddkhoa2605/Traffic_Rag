from __future__ import annotations

from abc import ABC, abstractmethod

from src.legal_tree.models import LegalDocument

from .models import RetrievalPassage, StrategyName
from .token_counter import TokenCounter


class ChunkStrategy(ABC):
    name: StrategyName

    def __init__(self, token_counter: TokenCounter, config: dict | None = None):
        self.token_counter = token_counter
        self.config = config or {}

    @abstractmethod
    def build(self, document: LegalDocument) -> list[RetrievalPassage]:
        raise NotImplementedError
