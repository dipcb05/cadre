# CADRE Documentation

**CADRE** is a Python security and reliability engine designed for Retrieval-Augmented Generation (RAG) and LLM-based application workflows. It provides deterministic control flow, multi-stage risk evaluation, probabilistic calibration, and safety-oriented action routing to safeguard LLM applications against untrusted inputs, context contamination, missing evidence, and ungrounded generation.

---

## Key Features

- **Multi-Stage Risk Evaluation**: Evaluates risk across four distinct risk heads: Instruction, Retrieval, Evidence, and Generation.
- **Deterministic Policy Engine**: `SafeRoutingPolicy` routes execution based on stage risk estimates, evidence states, and runtime budgets.
- **Probabilistic Calibration**: Built-in Platt scaling, Isotonic regression, and lineaged distribution-free risk control utilities.
- **Evidence & Claim Graph Analysis**: `EvidenceGraph` and `ClaimGraph` tools for detecting structural contradictions, fragmentation, and claim dependency weights.
- **Pluggable Architecture**: Modular interfaces (`LLMProvider`, `Retriever`, `QueryRewriter`, `Reranker`, `EvidenceAnalyzer`, `ClaimVerifier`, `FeatureProducer`) for seamless integration into custom stacks.

---

## Documentation Map

- [Installation](installation.md): Installation guide and optional dependencies.
- [Quickstart](quickstart.md): Get up and running in minutes with basic examples.
- [Architecture](architecture.md): Core concepts, engine execution lifecycle, and policy routing mechanics.
- [Configuration](configuration.md): Complete guide to `CadreConfig`, `HeadConfig`, `PolicyConfig`, and `RuntimeBudget`.
- [Components](components.md): Deep dive into the four risk heads, models, and decision statuses.
- [Adapters](adapters.md): Built-in adapter implementations for LLM providers, retrievers, analyzers, and verifiers.
- [Calibration](calibration.md): Probability calibration and statistical threshold calibration utilities.
- [Graphs](graphs.md): Working with `EvidenceGraph` and `ClaimGraph` structures.
- [Examples](examples.md): Practical usage patterns, integration recipes, and production workflows.
- [Development](development.md): Extending CADRE with custom components, adapters, and feature producers.
