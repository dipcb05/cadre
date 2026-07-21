from __future__ import annotations

import logging
from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from .adapters import (
    BasicClaimVerifier,
    BasicEvidenceAnalyzer,
    IdentityRewriter,
    ScoreReranker,
)
from .config import CadreConfig
from .errors import ProviderError
from .features import HeuristicFeatureProducer
from .interfaces import (
    ClaimVerifier,
    EvidenceAnalyzer,
    FeatureProducer,
    LLMProvider,
    QueryRewriter,
    Reranker,
    Retriever,
)
from .models import (
    Action,
    CadreResult,
    DecisionStatus,
    Document,
    EvidenceState,
    Head,
    RiskBundle,
    RuntimeBudget,
    TrustedContext,
)
from .policy import PolicyState, SafeRoutingPolicy
from .risk import MonotoneRiskHead, RiskModelBundle
from .serialization import serialize_context

logger = logging.getLogger(__name__)


class CadreEngine:
    def __init__(
        self,
        *,
        config: CadreConfig,
        llm: LLMProvider,
        retriever: Retriever,
        risk_models: RiskModelBundle | None = None,
        feature_producer: FeatureProducer | None = None,
        rewriter: QueryRewriter | None = None,
        reranker: Reranker | None = None,
        evidence_analyzer: EvidenceAnalyzer | None = None,
        claim_verifier: ClaimVerifier | None = None,
    ) -> None:
        self.config = config
        self.llm = llm
        self.retriever = retriever
        self.feature_producer = feature_producer or HeuristicFeatureProducer()
        self.rewriter = rewriter or IdentityRewriter()
        self.reranker = reranker or ScoreReranker()
        self.evidence_analyzer = evidence_analyzer or BasicEvidenceAnalyzer()
        self.claim_verifier = claim_verifier or BasicClaimVerifier()
        self.policy = SafeRoutingPolicy(config.policy)
        self.risk_models = risk_models or RiskModelBundle(
            {
                head: MonotoneRiskHead(
                    cfg.feature_names,
                    threshold=cfg.threshold,
                )
                for head, cfg in config.heads.items()
            }
        )

    def _extract_risks(
        self,
        context: TrustedContext,
        documents: list[Document],
        response: str | None,
        metadata: dict[str, Any],
    ) -> RiskBundle:
        availability = {
            Head.INSTRUCTION: True,
            Head.RETRIEVAL: bool(documents),
            Head.EVIDENCE: bool(documents),
            Head.GENERATION: response is not None,
        }
        feature_map = {}
        for head in Head:
            try:
                features = dict(
                    self.feature_producer.produce(
                        head,
                        context,
                        documents,
                        response,
                        metadata=metadata,
                    )
                )
            except Exception as exc:
                logger.exception("feature extraction failed for %s", head.value)
                features = {
                    name: None
                    for name in self.config.heads[head].feature_names
                }
                metadata.setdefault("feature_failures", {})[head.value] = str(exc)
            feature_map[head] = features
        return self.risk_models.estimate(feature_map, availability)

    def run(
        self,
        context: TrustedContext,
        *,
        budget: RuntimeBudget | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> CadreResult:
        live_budget = deepcopy(budget or RuntimeBudget())
        runtime_meta: dict[str, Any] = dict(metadata or {})
        documents = list(context.evidence)
        query = context.user
        response: str | None = None
        evidence_state = EvidenceState.UNKNOWN
        verification_state = EvidenceState.UNKNOWN
        trace: list[dict[str, Any]] = []

        max_steps = live_budget.total_nonterminal()
        for step in range(max_steps + 1):
            if documents:
                try:
                    evidence_state, evidence_features = self.evidence_analyzer.analyze(
                        query, documents, metadata=runtime_meta
                    )
                    runtime_meta["evidence_features"] = evidence_features
                except Exception as exc:
                    logger.exception("evidence analysis failed")
                    evidence_state = EvidenceState.UNKNOWN
                    runtime_meta["evidence_features"] = {"support_gap": 1.0}
                    runtime_meta["evidence_analysis_error"] = str(exc)

            risks = self._extract_risks(context, documents, response, runtime_meta)
            state = PolicyState(
                risks=risks,
                has_evidence=bool(documents),
                has_response=response is not None,
                evidence_state=evidence_state,
                verification_state=verification_state,
                budget=live_budget,
            )
            action = self.policy.choose(state)
            trace.append(
                {
                    "step": step,
                    "action": action.value,
                    "budget": live_budget.model_dump(),
                    "risks": {
                        h.value: risks.for_head(h).probability
                        for h in Head
                    },
                    "evidence_state": evidence_state.value,
                    "verification_state": verification_state.value,
                }
            )

            if action is Action.ACCEPT:
                return CadreResult(
                    status=DecisionStatus.ACCEPTED,
                    response=response,
                    action=action,
                    reason="All available stage risks are below their configured thresholds.",
                    risks=risks,
                    evidence=tuple(documents),
                    trace=tuple(trace),
                )
            if action is Action.CLARIFY:
                return CadreResult(
                    status=DecisionStatus.CLARIFICATION,
                    response=None,
                    action=action,
                    reason="The request or evidence is insufficient or unresolved.",
                    risks=risks,
                    evidence=tuple(documents),
                    trace=tuple(trace),
                )
            if action is Action.ABSTAIN:
                return CadreResult(
                    status=DecisionStatus.ABSTAINED,
                    response=None,
                    action=action,
                    reason="Residual risk could not be reduced within the available budget.",
                    risks=risks,
                    evidence=tuple(documents),
                    trace=tuple(trace),
                )
            if action is Action.REFUSE:
                return CadreResult(
                    status=DecisionStatus.REFUSED,
                    response=None,
                    action=action,
                    reason="Instruction-control risk exceeded the configured threshold.",
                    risks=risks,
                    evidence=tuple(documents),
                    trace=tuple(trace),
                )

            try:
                if action is Action.RETRIEVE:
                    live_budget.consume(action)
                    documents = list(
                        self.retriever.retrieve(
                            query,
                            k=self.config.policy.max_retrieval_depth,
                            metadata=runtime_meta,
                        )
                    )
                elif action is Action.REWRITE:
                    live_budget.consume(action)
                    query = self.rewriter.rewrite(query, metadata=runtime_meta)
                    documents = list(
                        self.retriever.retrieve(
                            query,
                            k=self.config.policy.max_retrieval_depth,
                            metadata=runtime_meta,
                        )
                    )
                elif action is Action.RERANK:
                    live_budget.consume(action)
                    documents = list(
                        self.reranker.rerank(
                            query,
                            documents,
                            k=self.config.policy.max_retrieval_depth,
                            metadata=runtime_meta,
                        )
                    )
                elif action is Action.GRAPH:
                    live_budget.consume(action)
                    # Graph construction is delegated to the evidence analyzer/feature producer.
                    runtime_meta["graph_requested"] = True
                elif action in {Action.GENERATE, Action.REGENERATE}:
                    live_budget.consume(action)
                    prompt = serialize_context(context.model_copy(update={"user": query}), documents)
                    response = self.llm.generate(
                        prompt,
                        max_tokens=max(1, live_budget.tokens),
                        metadata=runtime_meta,
                    )
                    verification_state = EvidenceState.UNKNOWN
                elif action is Action.VERIFY:
                    live_budget.consume(action)
                    if response is None:
                        verification_state = EvidenceState.INSUFFICIENT
                        runtime_meta["verification_features"] = {
                            "unsupported_insufficient": 1.0
                        }
                    else:
                        verification_state, verification_features = self.claim_verifier.verify(
                            response,
                            documents,
                            metadata=runtime_meta,
                        )
                        runtime_meta["verification_features"] = verification_features
                else:
                    raise RuntimeError(f"unsupported nonterminal action: {action}")
            except Exception as exc:
                logger.exception("CADRE action failed: %s", action.value)
                if self.config.strict_provider_failures:
                    raise ProviderError(f"{action.value} failed") from exc
                runtime_meta.setdefault("action_failures", []).append(
                    {"action": action.value, "error": str(exc)}
                )
                # Fail closed.
                final_risks = self._extract_risks(context, documents, response, runtime_meta)
                return CadreResult(
                    status=DecisionStatus.ABSTAINED,
                    response=None,
                    action=Action.ABSTAIN,
                    reason=f"{action.value} failed; CADRE failed closed.",
                    risks=final_risks,
                    evidence=tuple(documents),
                    trace=tuple(trace),
                )

        final_risks = self._extract_risks(context, documents, response, runtime_meta)
        return CadreResult(
            status=DecisionStatus.ABSTAINED,
            response=None,
            action=Action.ABSTAIN,
            reason="Maximum bounded execution depth reached.",
            risks=final_risks,
            evidence=tuple(documents),
            trace=tuple(trace),
        )
