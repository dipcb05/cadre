from cadre.graphs import ClaimGraph, EvidenceEdge, EvidenceGraph, ClaimNode, EvidenceRelation
from cadre.models import Document


def test_claim_weights_sum_to_one():
    claims = [
        ClaimNode(id="0", text="a"),
        ClaimNode(id="1", text="b"),
        ClaimNode(id="2", text="c"),
    ]
    graph = ClaimGraph(claims, [(0, 1), (0, 2)])
    weights = graph.importance_weights()
    assert abs(sum(weights) - 1.0) < 1e-9


def test_evidence_graph_metrics():
    docs = [
        Document(id="a", text="x", source="s1"),
        Document(id="b", text="y", source="s2"),
    ]
    graph = EvidenceGraph(docs, [EvidenceEdge("a", "b", EvidenceRelation.CONTRADICTS)])
    assert graph.contradiction_density() == 1.0
    assert graph.provenance_diversity() == 1.0
