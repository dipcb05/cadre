from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import networkx as nx
from pydantic import BaseModel, ConfigDict

from .models import Document


class EvidenceRelation(str, Enum):
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    DUPLICATES = "duplicates"
    RELATED_TO = "related_to"
    DERIVED_FROM = "derived_from"


@dataclass(frozen=True)
class EvidenceEdge:
    source_id: str
    target_id: str
    relation: EvidenceRelation
    weight: float = 1.0


class EvidenceGraph:
    def __init__(
        self, documents: list[Document], edges: list[EvidenceEdge] | None = None
    ) -> None:
        self.graph = nx.MultiDiGraph()
        for doc in documents:
            self.graph.add_node(doc.id, document=doc)
        for edge in edges or []:
            if edge.source_id in self.graph and edge.target_id in self.graph:
                self.graph.add_edge(
                    edge.source_id,
                    edge.target_id,
                    relation=edge.relation.value,
                    weight=edge.weight,
                )

    def expand(
        self, documents: list[Document], edges: list[EvidenceEdge] | None = None
    ) -> None:
        for doc in documents:
            if doc.id not in self.graph:
                self.graph.add_node(doc.id, document=doc)
        for edge in edges or []:
            if edge.source_id in self.graph and edge.target_id in self.graph:
                self.graph.add_edge(
                    edge.source_id,
                    edge.target_id,
                    relation=edge.relation.value,
                    weight=edge.weight,
                )

    def documents(self) -> list[Document]:
        return [data["document"] for _, data in self.graph.nodes(data=True)]

    def contradiction_density(self) -> float:
        edges = list(self.graph.edges(data=True))
        if not edges:
            return 0.0
        contradictions = sum(
            data.get("relation") == EvidenceRelation.CONTRADICTS.value
            for _, _, data in edges
        )
        return contradictions / len(edges)

    def provenance_diversity(self) -> float:
        docs = self.documents()
        if not docs:
            return 0.0
        return len({doc.source for doc in docs}) / len(docs)

    def fragmentation(self) -> float:
        if self.graph.number_of_nodes() <= 1:
            return 0.0
        undirected = self.graph.to_undirected()
        components = nx.number_connected_components(undirected)
        return (components - 1) / max(1, self.graph.number_of_nodes() - 1)

    def source_coverage(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for doc in self.documents():
            counts[doc.source] = counts.get(doc.source, 0) + 1
        return counts


class ClaimEvidenceRelation(str, Enum):
    ENTAILED = "entailed"
    CONTRADICTED = "contradicted"
    PARTIALLY_SUPPORTED = "partially_supported"
    UNSUPPORTED = "unsupported"


class ClaimNode(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    text: str
    claim_type: str = "factual"
    confidence: float | None = None


class ClaimEvidenceEdge(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    claim_id: str
    evidence_id: str
    relation: ClaimEvidenceRelation
    score: float = 0.0


class ClaimGraph:
    def __init__(
        self,
        claims: list[ClaimNode],
        dependencies: list[tuple[int, int]] | None = None,
        evidence_edges: list[ClaimEvidenceEdge] | None = None,
    ) -> None:
        self.graph = nx.DiGraph()
        self.claims = list(claims)
        self.evidence_edges = list(evidence_edges or [])
        for idx, claim in enumerate(claims):
            self.graph.add_node(idx, claim=claim)
        for source, target in dependencies or []:
            if source != target and source in self.graph and target in self.graph:
                self.graph.add_edge(source, target)

    def importance_weights(self) -> list[float]:
        n = self.graph.number_of_nodes()
        if n == 0:
            return []
        condensed = nx.condensation(self.graph)
        node_to_component = condensed.graph["mapping"]
        raw = [1.0 + condensed.out_degree(node_to_component[i]) for i in range(n)]
        total = sum(raw)
        return [value / total for value in raw]

    def unsupported_claims(self) -> list[ClaimNode]:
        supported_ids = {
            e.claim_id
            for e in self.evidence_edges
            if e.relation
            in {ClaimEvidenceRelation.ENTAILED, ClaimEvidenceRelation.PARTIALLY_SUPPORTED}
        }
        return [c for c in self.claims if c.id not in supported_ids]

    def contradicted_claims(self) -> list[ClaimNode]:
        ids = {
            e.claim_id
            for e in self.evidence_edges
            if e.relation is ClaimEvidenceRelation.CONTRADICTED
        }
        return [c for c in self.claims if c.id in ids]

    def coverage(self) -> float:
        if not self.claims:
            return 1.0
        supported = {
            e.claim_id
            for e in self.evidence_edges
            if e.relation
            in {ClaimEvidenceRelation.ENTAILED, ClaimEvidenceRelation.PARTIALLY_SUPPORTED}
        }
        return len(supported & {c.id for c in self.claims}) / len(self.claims)
