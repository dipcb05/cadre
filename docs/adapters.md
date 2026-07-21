# Adapters

CADRE defines abstract interfaces in `cadre.interfaces` and ships lightweight reference implementations in `cadre.adapters`.

---

## Abstract Interfaces (`cadre.interfaces`)

Custom integrations subclass these abstract base classes:

```python
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any
from cadre.models import Document, EvidenceState, Head, TrustedContext

class LLMProvider(ABC):
    @abstractmethod
    def generate(self, prompt: str, *, max_tokens: int, metadata: Mapping[str, Any]) -> str:
        pass

class Retriever(ABC):
    @abstractmethod
    def retrieve(self, query: str, *, k: int, metadata: Mapping[str, Any]) -> Sequence[Document]:
        pass

class QueryRewriter(ABC):
    @abstractmethod
    def rewrite(self, query: str, *, metadata: Mapping[str, Any]) -> str:
        pass

class Reranker(ABC):
    @abstractmethod
    def rerank(
        self,
        query: str,
        documents: Sequence[Document],
        *,
        k: int,
        metadata: Mapping[str, Any],
    ) -> Sequence[Document]:
        pass

class EvidenceAnalyzer(ABC):
    @abstractmethod
    def analyze(
        self,
        query: str,
        documents: Sequence[Document],
        *,
        metadata: Mapping[str, Any],
    ) -> tuple[EvidenceState, dict[str, float]]:
        pass

class ClaimVerifier(ABC):
    @abstractmethod
    def verify(
        self,
        response: str,
        documents: Sequence[Document],
        *,
        metadata: Mapping[str, Any],
    ) -> tuple[EvidenceState, dict[str, float]]:
        pass

class FeatureProducer(ABC):
    @abstractmethod
    def produce(
        self,
        head: Head,
        context: TrustedContext,
        documents: Sequence[Document],
        response: str | None,
        *,
        metadata: Mapping[str, Any],
    ) -> Mapping[str, float | None]:
        pass
```

---

## Built-In Reference Adapters (`cadre.adapters`)

CADRE provides built-in adapter classes for prototyping and testing:

### 1. `SimpleLLM`
Wraps any Python callable `(str) -> str`:
```python
from cadre.adapters import SimpleLLM

llm = SimpleLLM(lambda prompt: f"Response to: {prompt[:20]}")
```

### 2. `InMemoryRetriever`
Performs Jaccard term-overlap matching over an in-memory sequence of `Document` objects:
```python
from cadre.adapters import InMemoryRetriever
from cadre import Document

docs = [Document(id="1", text="Python security library")]
retriever = InMemoryRetriever(docs)
```

### 3. `IdentityRewriter`
Pass-through query rewriter returning `query` unchanged.

### 4. `ScoreReranker`
Sorts documents in descending order according to `doc.score`.

### 5. `BasicEvidenceAnalyzer`
Calculates term-overlap ratio between query and document texts to assign `EvidenceState` (`SUPPORTED` vs `INSUFFICIENT`) and compute `support_gap`.

### 6. `BasicClaimVerifier`
Verifies generated candidate response terms against retrieved document terms to assess claim grounding.

---

## Default Fallback Feature Producer

`HeuristicFeatureProducer` (`cadre.features`) is used by default if no custom `feature_producer` is supplied to `CadreEngine`. It computes heuristic script anomalies, prompt-injection pattern counts, role conflicts, and term Jaccard distances.
