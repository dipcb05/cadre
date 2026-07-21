from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Head(str, Enum):
    INSTRUCTION = "instruction"
    RETRIEVAL = "retrieval"
    EVIDENCE = "evidence"
    GENERATION = "generation"


class Action(str, Enum):
    RETRIEVE = "retrieve"
    REWRITE = "rewrite"
    RERANK = "rerank"
    GRAPH = "graph"
    GENERATE = "generate"
    REGENERATE = "regenerate"
    VERIFY = "verify"
    ACCEPT = "accept"
    CLARIFY = "clarify"
    ABSTAIN = "abstain"
    REFUSE = "refuse"


class DecisionStatus(str, Enum):
    ACCEPTED = "accepted"
    CLARIFICATION = "clarification"
    ABSTAINED = "abstained"
    REFUSED = "refused"
    FAILED = "failed"


class EvidenceState(str, Enum):
    SUPPORTED = "supported"
    INSUFFICIENT = "insufficient"
    CONTRADICTED = "contradicted"
    CONFLICTING = "conflicting"
    UNKNOWN = "unknown"


class Document(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    text: str
    source: str = "unknown"
    score: float | None = None
    language: str | None = None
    timestamp: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("id", "text")
    @classmethod
    def not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value


class TrustedContext(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    system: str
    developer: str = ""
    user: str
    evidence: tuple[Document, ...] = ()

    @field_validator("system", "user")
    @classmethod
    def required_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value


class RuntimeBudget(BaseModel):
    model_config = ConfigDict(extra="forbid")

    retrieval: int = Field(default=2, ge=0)
    generation: int = Field(default=2, ge=0)
    verification: int = Field(default=1, ge=0)
    tokens: int = Field(default=4096, ge=0)

    def total_nonterminal(self) -> int:
        return self.retrieval + self.generation + self.verification

    def consume(self, action: Action, tokens: int = 0) -> None:
        if tokens < 0:
            raise ValueError("tokens must be nonnegative")
        if action in {Action.RETRIEVE, Action.REWRITE, Action.RERANK, Action.GRAPH}:
            if self.retrieval <= 0:
                raise ValueError("retrieval budget exhausted")
            self.retrieval -= 1
        elif action in {Action.GENERATE, Action.REGENERATE}:
            if self.generation <= 0:
                raise ValueError("generation budget exhausted")
            self.generation -= 1
        elif action is Action.VERIFY:
            if self.verification <= 0:
                raise ValueError("verification budget exhausted")
            self.verification -= 1
        if tokens > self.tokens:
            raise ValueError("token budget exhausted")
        self.tokens -= tokens


class RiskEstimate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    probability: float = Field(ge=0.0, le=1.0)
    threshold: float = Field(ge=0.0, le=1.0)
    flagged: bool
    available: bool = True
    features: dict[str, float] = Field(default_factory=dict)
    missing: tuple[str, ...] = ()


class RiskBundle(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    instruction: RiskEstimate
    retrieval: RiskEstimate
    evidence: RiskEstimate
    generation: RiskEstimate

    def for_head(self, head: Head) -> RiskEstimate:
        return getattr(self, head.value)

    def any_flagged(self) -> bool:
        return any(self.for_head(h).flagged for h in Head if self.for_head(h).available)


class CadreResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: DecisionStatus
    response: str | None
    action: Action
    reason: str
    risks: RiskBundle
    evidence: tuple[Document, ...] = ()
    trace: tuple[dict[str, Any], ...] = ()
