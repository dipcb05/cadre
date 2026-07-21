from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import networkx as nx

from .models import Document


@dataclass(frozen=True)
class EvidenceEdge:
    source_id: str
    target_id: str
    relation: Literal["entails", "contradicts", "duplicates", "depends", "temporal_conflict"]
    weight: float = 1.0


class EvidenceGraph:
    def __init__(self, documents: list[Document], edges: list[EvidenceEdge] | None = None) -> None:
        self.graph = nx.MultiDiGraph()
        for doc in documents:
            self.graph.add_node(doc.id, document=doc)
        for edge in edges or []:
            if edge.source_id in self.graph and edge.target_id in self.graph:
                self.graph.add_edge(
                    edge.source_id,
                    edge.target_id,
                    relation=edge.relation,
                    weight=edge.weight,
                )

    def contradiction_density(self) -> float:
        edges = list(self.graph.edges(data=True))
        if not edges:
            return 0.0
        contradictions = sum(data.get("relation") == "contradicts" for _, _, data in edges)
        return contradictions / len(edges)

    def provenance_diversity(self) -> float:
        documents = [data["document"] for _, data in self.graph.nodes(data=True)]
        if not documents:
            return 0.0
        return len({doc.source for doc in documents}) / len(documents)

    def fragmentation(self) -> float:
        if self.graph.number_of_nodes() <= 1:
            return 0.0
        undirected = self.graph.to_undirected()
        components = nx.number_connected_components(undirected)
        return (components - 1) / max(1, self.graph.number_of_nodes() - 1)


class ClaimGraph:
    def __init__(self, claims: list[str], dependencies: list[tuple[int, int]]) -> None:
        self.graph = nx.DiGraph()
        for idx, claim in enumerate(claims):
            self.graph.add_node(idx, claim=claim)
        for source, target in dependencies:
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
