from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from .interfaces import (
    ClaimVerifier,
    EvidenceAnalyzer,
    LLMProvider,
    QueryRewriter,
    Reranker,
    Retriever,
)
from .models import Document, EvidenceState


class SimpleLLM(LLMProvider):
    def __init__(self, fn: Callable[[str], str]) -> None:
        self.fn = fn

    def generate(self, prompt: str, *, max_tokens: int, metadata: Mapping[str, Any]) -> str:
        result = self.fn(prompt)
        if not isinstance(result, str) or not result.strip():
            raise ValueError("LLM provider returned an empty or non-string response")
        return result.strip()


class InMemoryRetriever(Retriever):
    def __init__(self, documents: Sequence[Document]) -> None:
        self.documents = tuple(documents)

    def retrieve(self, query: str, *, k: int, metadata: Mapping[str, Any]) -> Sequence[Document]:
        terms = set(query.lower().split())
        ranked = sorted(
            self.documents,
            key=lambda doc: len(terms & set(doc.text.lower().split())),
            reverse=True,
        )
        return ranked[:k]


class IdentityRewriter(QueryRewriter):
    def rewrite(self, query: str, *, metadata: Mapping[str, Any]) -> str:
        return query


class ScoreReranker(Reranker):
    def rerank(
        self,
        query: str,
        documents: Sequence[Document],
        *,
        k: int,
        metadata: Mapping[str, Any],
    ) -> Sequence[Document]:
        return sorted(
            documents,
            key=lambda d: d.score if d.score is not None else 0.0,
            reverse=True,
        )[:k]


class BasicEvidenceAnalyzer(EvidenceAnalyzer):
    def analyze(
        self,
        query: str,
        documents: Sequence[Document],
        *,
        metadata: Mapping[str, Any],
    ) -> tuple[EvidenceState, dict[str, float]]:
        if not documents:
            return EvidenceState.INSUFFICIENT, {
                "support_gap": 1.0,
                "contradiction_density": 0.0,
                "staleness": 0.0,
                "fragmentation": 1.0,
            }
        query_terms = set(query.lower().split())
        overlap = sum(
            len(query_terms & set(doc.text.lower().split()))
            for doc in documents
        ) / max(1, len(query_terms) * len(documents))
        support_gap = max(0.0, min(1.0, 1.0 - overlap))
        state = EvidenceState.SUPPORTED if support_gap < 0.8 else EvidenceState.INSUFFICIENT
        return state, {
            "support_gap": support_gap,
            "contradiction_density": 0.0,
            "staleness": 0.0,
            "fragmentation": 0.0,
        }


class BasicClaimVerifier(ClaimVerifier):
    def verify(
        self,
        response: str,
        documents: Sequence[Document],
        *,
        metadata: Mapping[str, Any],
    ) -> tuple[EvidenceState, dict[str, float]]:
        if not response.strip():
            return EvidenceState.INSUFFICIENT, {"unsupported_insufficient": 1.0}
        if not documents:
            return EvidenceState.INSUFFICIENT, {"unsupported_insufficient": 1.0}
        response_terms = set(response.lower().split())
        evidence_terms = set(" ".join(d.text for d in documents).lower().split())
        support = len(response_terms & evidence_terms) / max(1, len(response_terms))
        if support >= 0.25:
            return EvidenceState.SUPPORTED, {
                "unsupported_insufficient": 1.0 - support,
                "dependency_unsupported": 1.0 - support,
            }
        return EvidenceState.INSUFFICIENT, {
            "unsupported_insufficient": 1.0 - support,
            "dependency_unsupported": 1.0 - support,
        }
