# CADRE LLM Security

Reference implementation of a calibration-aware, four-head risk estimation and bounded adaptive routing architecture for secure LLM and RAG systems.

The package separates:

- instruction risk
- retrieval risk
- evidence risk
- generation risk

It also provides:

- typed trust boundaries
- monotone constrained risk heads
- probability calibration
- complete-lineage threshold calibration
- evidence and claim graph utilities
- finite-state adaptive policy routing
- bounded corrective RAG execution
- provider-neutral LLM, retriever, verifier, and feature interfaces

## Install

```bash
pip install cadre-llm-security
```

## Minimal use

```python
from cadre import (
    CadreEngine,
    CadreConfig,
    RuntimeBudget,
    TrustedContext,
)
from cadre.adapters import SimpleLLM, InMemoryRetriever

engine = CadreEngine(
    config=CadreConfig.safe_default(),
    llm=SimpleLLM(lambda prompt: "A grounded answer."),
    retriever=InMemoryRetriever([]),
)

result = engine.run(
    TrustedContext(
        system="Answer only from trusted evidence.",
        developer="Cite evidence IDs.",
        user="What is the policy?",
    ),
    budget=RuntimeBudget(retrieval=2, generation=2, verification=1, tokens=4096),
)
print(result.status, result.response)
```

The package is intentionally provider-neutral. Production deployments should replace built-in heuristic feature producers and adapters with frozen, versioned models calibrated on lineage-disjoint data.
