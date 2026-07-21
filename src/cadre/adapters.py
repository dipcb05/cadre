from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from .interfaces import (
    ClaimExtractor,
    ClaimVerifier,
    EvidenceAnalyzer,
    LLMProvider,
    QueryRewriter,
    Reranker,
    Retriever,
    StructuredClaimVerifier,
)
from .models import Document, EvidenceState
from .graphs import ClaimNode, ClaimEvidenceEdge, ClaimEvidenceRelation, EvidenceGraph


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


class BasicClaimExtractor(ClaimExtractor):
    def extract(self, response: str, *, metadata: Mapping[str, Any]) -> list[ClaimNode]:
        if not response.strip():
            return []
        # Basic heuristic sentence-based claim extractor
        import re
        sentences = re.split(r"(?<=[.!?])\s+", response)
        claims = []
        for idx, sentence in enumerate(sentences):
            clean = sentence.strip()
            if len(clean) > 5:
                claims.append(ClaimNode(
                    id=f"c_{idx}",
                    text=clean,
                    claim_type="factual",
                    confidence=1.0,
                ))
        return claims


class BasicStructuredClaimVerifier(StructuredClaimVerifier):
    def verify_claims(
        self,
        claims: list[ClaimNode],
        evidence_graph: EvidenceGraph,
        *,
        metadata: Mapping[str, Any],
    ) -> tuple[EvidenceState, dict[str, float], list[ClaimEvidenceEdge]]:
        evidence_docs = evidence_graph.documents()
        if not claims or not evidence_docs:
            return EvidenceState.INSUFFICIENT, {"unsupported_insufficient": 1.0}, []

        evidence_text = " ".join(d.text for d in evidence_docs).lower()
        edges = []
        entailed_count = 0
        contradicted_count = 0
        
        for claim in claims:
            claim_terms = set(claim.text.lower().split())
            if not claim_terms:
                continue
            
            # Simple check for contradiction words + match
            contradicts_evidence = any(
                ("not" in claim.text.lower() and term in evidence_text) 
                for term in claim_terms if len(term) > 3
            )
            
            overlap = sum(1 for term in claim_terms if term in evidence_text) / len(claim_terms)
            
            if contradicts_evidence:
                relation = ClaimEvidenceRelation.CONTRADICTED
                contradicted_count += 1
            elif overlap >= 0.3:
                relation = ClaimEvidenceRelation.ENTAILED
                entailed_count += 1
            elif overlap >= 0.1:
                relation = ClaimEvidenceRelation.PARTIALLY_SUPPORTED
            else:
                relation = ClaimEvidenceRelation.UNSUPPORTED

            # Find matching document source for edge
            for doc in evidence_docs:
                doc_terms = set(doc.text.lower().split())
                doc_overlap = len(claim_terms & doc_terms) / len(claim_terms) if claim_terms else 0.0
                if doc_overlap > 0:
                    edges.append(ClaimEvidenceEdge(
                        claim_id=claim.id,
                        evidence_id=doc.id,
                        relation=relation,
                        score=float(doc_overlap),
                    ))

        if contradicted_count > 0:
            state = EvidenceState.CONTRADICTED
        elif entailed_count / len(claims) >= 0.5:
            state = EvidenceState.SUPPORTED
        else:
            state = EvidenceState.INSUFFICIENT

        unsupported_ratio = sum(1 for e in edges if e.relation == ClaimEvidenceRelation.UNSUPPORTED) / max(1, len(edges))
        contradicted_ratio = contradicted_count / len(claims)

        features = {
            "unsupported_insufficient": float(unsupported_ratio),
            "unsupported_contradicted": float(contradicted_ratio),
            "unsupported_conflicting": 0.0,
            "dependency_unsupported": 0.0,
        }

        return state, features, edges

