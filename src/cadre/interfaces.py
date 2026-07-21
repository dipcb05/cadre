from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any

from .models import Document, EvidenceState, Head, TrustedContext


class LLMProvider(ABC):
    @abstractmethod
    def generate(self, prompt: str, *, max_tokens: int, metadata: Mapping[str, Any]) -> str:
        raise NotImplementedError


class Retriever(ABC):
    @abstractmethod
    def retrieve(self, query: str, *, k: int, metadata: Mapping[str, Any]) -> Sequence[Document]:
        raise NotImplementedError


class QueryRewriter(ABC):
    @abstractmethod
    def rewrite(self, query: str, *, metadata: Mapping[str, Any]) -> str:
        raise NotImplementedError


class Reranker(ABC):
    @abstractmethod
    def rerank(
        self,
        query: str,
        documents: Sequence[Document],
        *,
        k: int,
        metadata: Mapping[str, Any],
    ) -> Sequence[Document]:
        raise NotImplementedError


class EvidenceAnalyzer(ABC):
    @abstractmethod
    def analyze(
        self,
        query: str,
        documents: Sequence[Document],
        *,
        metadata: Mapping[str, Any],
    ) -> tuple[EvidenceState, dict[str, float]]:
        raise NotImplementedError


class ClaimVerifier(ABC):
    @abstractmethod
    def verify(
        self,
        response: str,
        documents: Sequence[Document],
        *,
        metadata: Mapping[str, Any],
    ) -> tuple[EvidenceState, dict[str, float]]:
        raise NotImplementedError


class FeatureProducer(ABC):
    @abstractmethod
    def produce(
        self,
        head: Head,
        context: TrustedContext,
        documents: Sequence[Document],
        response: str | None,
        *,
        metadata: Mapping[str, Any],
    ) -> Mapping[str, float | None]:
        raise NotImplementedError
