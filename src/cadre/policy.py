from __future__ import annotations

from dataclasses import dataclass

from .config import PolicyConfig
from .models import Action, EvidenceState, Head, RiskBundle, RuntimeBudget


@dataclass(frozen=True)
class PolicyState:
    risks: RiskBundle
    has_evidence: bool
    has_response: bool
    evidence_state: EvidenceState
    verification_state: EvidenceState
    budget: RuntimeBudget


class SafeRoutingPolicy:
    """Deterministic bounded policy with conservative release behavior."""

    def __init__(self, config: PolicyConfig) -> None:
        self.config = config

    def choose(self, state: PolicyState) -> Action:
        i = state.risks.for_head(Head.INSTRUCTION)
        r = state.risks.for_head(Head.RETRIEVAL)
        e = state.risks.for_head(Head.EVIDENCE)
        g = state.risks.for_head(Head.GENERATION)

        if i.available and i.flagged:
            return Action.REFUSE if self.config.refuse_on_instruction_risk else Action.ABSTAIN

        if not state.has_evidence:
            if state.budget.retrieval > 0:
                return Action.RETRIEVE
            return Action.CLARIFY if self.config.clarify_on_missing_evidence else Action.ABSTAIN

        if r.available and r.flagged:
            if state.budget.retrieval > 0:
                return Action.REWRITE
            return Action.ABSTAIN

        if state.evidence_state is EvidenceState.CONTRADICTED:
            if state.budget.retrieval > 0:
                return Action.RETRIEVE
            return Action.ABSTAIN

        if state.evidence_state is EvidenceState.CONFLICTING:
            if state.budget.verification > 0:
                return Action.VERIFY
            return Action.CLARIFY

        if e.available and e.flagged:
            if state.budget.retrieval > 0:
                return Action.RERANK
            return Action.CLARIFY if self.config.clarify_on_missing_evidence else Action.ABSTAIN

        if not state.has_response:
            return Action.GENERATE if state.budget.generation > 0 else Action.ABSTAIN

        if state.verification_state is EvidenceState.CONTRADICTED:
            return Action.REGENERATE if state.budget.generation > 0 else Action.ABSTAIN

        if g.available and g.flagged:
            if state.budget.verification > 0:
                return Action.VERIFY
            if state.budget.generation > 0:
                return Action.REGENERATE
            return Action.ABSTAIN

        if (
            state.verification_state in {EvidenceState.SUPPORTED, EvidenceState.UNKNOWN}
            and state.has_response
        ):
            return Action.ACCEPT

        if state.verification_state is EvidenceState.INSUFFICIENT:
            return Action.CLARIFY

        return Action.ABSTAIN
