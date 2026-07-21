# CADRE

A modular Python framework for building secure, risk-aware LLM and Retrieval-Augmented Generation (RAG) applications.

CADRE provides a configurable execution pipeline with independent risk estimation, adaptive routing, retrieval control, evidence validation, and provider-neutral interfaces.

---

## Features

- Adaptive routing engine
- Typed trust boundaries
- Monotonic risk estimation
- Probability calibration
- Threshold calibration
- Evidence graph utilities
- Claim graph utilities
- Configurable security policies
- Bounded execution budgets
- Provider-neutral architecture
- Pluggable feature extraction
- Pluggable retrievers
- Pluggable rerankers
- Pluggable verifiers
- Pluggable LLM providers

---

## Installation

```bash
pip install cadre
```

or

```bash
pip install git+https://github.com/dipcb05/cadre.git
```

---

## Quick Start

```python
from cadre import (
    CadreEngine,
    CadreConfig,
    RuntimeBudget,
    TrustedContext,
)

from cadre.adapters import (
    SimpleLLM,
    InMemoryRetriever,
)

engine = CadreEngine(
    config=CadreConfig.safe_default(),
    llm=SimpleLLM(lambda _: "Grounded response."),
    retriever=InMemoryRetriever([]),
)

result = engine.run(
    TrustedContext(
        system="Answer only from trusted evidence.",
        developer="Cite retrieved documents.",
        user="What is the refund policy?",
    ),
    budget=RuntimeBudget(),
)

print(result.status)
print(result.response)
```

---

# Components

```
cadre/
├── adapters/
├── calibration/
├── config/
├── engine/
├── features/
├── graphs/
├── interfaces/
├── models/
├── policy/
├── risk/
└── serialization/
```

---

# Supported Integrations

- OpenAI
- Anthropic
- Google Gemini
- Ollama
- vLLM
- HuggingFace Transformers
- LangChain
- LlamaIndex
- Haystack
- Qdrant
- Pinecone
- Weaviate
- Chroma
- pgvector

---

# Development

```bash
git clone https://github.com/dipcb05/cadre.git

cd cadre

pip install -e ".[dev]"

pytest

ruff check

mypy src
```

---

# License

Apache-2.0
