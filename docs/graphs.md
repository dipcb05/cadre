# Graphs

CADRE provides graph structures in `cadre.graphs` based on NetworkX to analyze relationships between documents and generated claims.

---

## `EvidenceGraph`

`EvidenceGraph` constructs a directed multi-graph over `Document` nodes connected by relational `EvidenceEdge` instances.

```python
from cadre import Document
from cadre.graphs import EvidenceGraph, EvidenceEdge

docs = [
    Document(id="doc-1", source="wiki", text="Version 2.0 released in 2025."),
    Document(id="doc-2", source="blog", text="Version 2.0 released in 2026."),
    Document(id="doc-3", source="forum", text="Unrelated discussion topic."),
]

edges = [
    EvidenceEdge(source_id="doc-1", target_id="doc-2", relation="contradicts", weight=1.0),
    EvidenceEdge(source_id="doc-1", target_id="doc-3", relation="depends", weight=0.5),
]

graph = EvidenceGraph(docs, edges)

print(f"Contradiction Density: {graph.contradiction_density():.2f}")
print(f"Provenance Diversity: {graph.provenance_diversity():.2f}")
print(f"Fragmentation: {graph.fragmentation():.2f}")
```

### Methods & Metrics

| Method | Return Type | Description |
| :--- | :--- | :--- |
| `contradiction_density()` | `float` | Proportion of edges having `relation="contradicts"`. Range `[0.0, 1.0]`. |
| `provenance_diversity()` | `float` | Unique document `source` count divided by total document count. |
| `fragmentation()` | `float` | Normalized count of disconnected components across the evidence graph. |

---

## `ClaimGraph`

`ClaimGraph` constructs a directed acyclic graph (DAG) representing dependency structures between individual claim sentences extracted from candidate responses.

```python
from cadre.graphs import ClaimGraph

claims = [
    "CADRE is a Python library.",
    "CADRE uses Pydantic for validation.",
    "Therefore, CadreConfig validates parameters strictly.",
]

# Directed edge (source -> target) indicates dependency: claim 2 depends on claims 0 and 1
dependencies = [
    (0, 2),
    (1, 2),
]

claim_graph = ClaimGraph(claims, dependencies)
weights = claim_graph.importance_weights()

for claim, weight in zip(claims, weights):
    print(f"[{weight:.3f}] {claim}")
```

### Importance Weighting

`claim_graph.importance_weights()` computes structural importance weights based on DAG condensation degrees:
- Fundamental premise claims (nodes with high out-degree dependency edges) receive higher relative weights.
- Derived downstream claims receive lower relative weights.
- Weights are normalized so $\sum w_i = 1.0$.
