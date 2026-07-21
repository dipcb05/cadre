# Examples

This section provides usage patterns for deploying CADRE in typical application environments.

---

## 1. Custom LLM Integration (e.g. OpenAI / Anthropic)

Wrap external LLM API client calls by implementing `LLMProvider`:

```python
from typing import Mapping, Any
from cadre import CadreEngine, CadreConfig, TrustedContext
from cadre.interfaces import LLMProvider

class CustomOpenAIProvider(LLMProvider):
    def __init__(self, client: Any, model_name: str = "gpt-4o"):
        self.client = client
        self.model_name = model_name

    def generate(self, prompt: str, *, max_tokens: int, metadata: Mapping[str, Any]) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.0,
        )
        return response.choices[0].message.content
```

---

## 2. Custom Vector Store Retriever (e.g. Qdrant / Pinecone)

Wrap vector store index queries by implementing `Retriever`:

```python
from typing import Mapping, Sequence, Any
from cadre import Document
from cadre.interfaces import Retriever

class VectorStoreRetriever(Retriever):
    def __init__(self, vector_client: Any, collection_name: str):
        self.client = vector_client
        self.collection_name = collection_name

    def retrieve(self, query: str, *, k: int, metadata: Mapping[str, Any]) -> Sequence[Document]:
        hits = self.client.search(
            collection_name=self.collection_name,
            query_filter=query,
            limit=k,
        )
        return [
            Document(
                id=str(hit.id),
                text=hit.payload["text"],
                source=hit.payload.get("source", "vector_db"),
                score=float(hit.score),
            )
            for hit in hits
        ]
```

---

## 3. Strict Safety Guardrail Pipeline

Configure low thresholds on `instruction` and `evidence` heads to enforce conservative release behavior:

```python
from cadre import CadreConfig, HeadConfig, Head, CadreEngine, DecisionStatus

# Custom configuration with tight thresholds
config = CadreConfig(
    heads={
        Head.INSTRUCTION: HeadConfig(
            feature_names=("uncertainty", "task_inconsistency", "role_conflict", "boundary_violation", "script_anomaly", "missing_indicator"),
            threshold=0.20,  # Flag instruction risk aggressively
        ),
        Head.RETRIEVAL: HeadConfig(
            feature_names=("facet_gap", "rank_instability", "source_concentration", "query_evidence_mismatch", "poison_likelihood", "language_mismatch", "missing_indicator"),
            threshold=0.40,
        ),
        Head.EVIDENCE: HeadConfig(
            feature_names=("support_gap", "contradiction_density", "provenance_gap", "staleness", "fragmentation", "missing_indicator"),
            threshold=0.30,
        ),
        Head.GENERATION: HeadConfig(
            feature_names=("unsupported_insufficient", "unsupported_contradicted", "unsupported_conflicting", "dependency_unsupported", "no_claim", "segmentation_failure", "verifier_failure", "missing_indicator"),
            threshold=0.30,
        ),
    },
    strict_provider_failures=False,
)
```

---

## 4. Extracting Trace Audits

Inspect detailed audit logs from `CadreResult.trace`:

```python
result = engine.run(context)

print(f"Final Status: {result.status.value}")
print(f"Final Action: {result.action.value}")

for step in result.trace:
    print(f"Step {step['step']}: Action={step['action']}")
    print(f"  Stage Risks: {step['risks']}")
    print(f"  Evidence State: {step['evidence_state']}")
```
