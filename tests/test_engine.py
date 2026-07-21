from cadre import CadreConfig, CadreEngine, DecisionStatus, Document, RuntimeBudget, TrustedContext
from cadre.adapters import InMemoryRetriever, SimpleLLM


def test_engine_terminates():
    docs = [
        Document(
            id="d1",
            text="The refund period is thirty days with a receipt.",
            source="policy",
            score=1.0,
        )
    ]
    engine = CadreEngine(
        config=CadreConfig.safe_default(),
        llm=SimpleLLM(lambda _: "The refund period is thirty days with a receipt."),
        retriever=InMemoryRetriever(docs),
    )
    result = engine.run(
        TrustedContext(
            system="Use evidence.",
            user="What is the refund period?",
        ),
        budget=RuntimeBudget(retrieval=2, generation=2, verification=1, tokens=1024),
    )
    assert result.status in {
        DecisionStatus.ACCEPTED,
        DecisionStatus.CLARIFICATION,
        DecisionStatus.ABSTAINED,
        DecisionStatus.REFUSED,
    }
    assert len(result.trace) <= 6
