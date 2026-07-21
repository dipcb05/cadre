from cadre.graphs import ClaimGraph, EvidenceEdge, EvidenceGraph
from cadre.models import Document


def test_claim_weights_sum_to_one():
    graph = ClaimGraph(["a", "b", "c"], [(0, 1), (0, 2)])
    weights = graph.importance_weights()
    assert abs(sum(weights) - 1.0) < 1e-9


def test_evidence_graph_metrics():
    docs = [
        Document(id="a", text="x", source="s1"),
        Document(id="b", text="y", source="s2"),
    ]
    graph = EvidenceGraph(docs, [EvidenceEdge("a", "b", "contradicts")])
    assert graph.contradiction_density() == 1.0
    assert graph.provenance_diversity() == 1.0
