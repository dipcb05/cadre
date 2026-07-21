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
    EXPAND_GRAPH = "expand_graph"
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


class TrustLevel(str, Enum):
    TRUSTED = "trusted"
    UNTRUSTED = "untrusted"
    DERIVED = "derived"


class ContentOrigin(str, Enum):
    SYSTEM = "system"
    DEVELOPER = "developer"
    USER = "user"
    RETRIEVAL = "retrieval"
    TOOL = "tool"
    MODEL = "model"


class Provenance(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    source_id: str
    retrieval_timestamp: str | None = None
    retrieval_score: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class MessageSegment(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    content: str
    trust: TrustLevel
    origin: ContentOrigin
    provenance: Provenance | None = None


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
    rewrite: int = Field(default=1, ge=0)
    rerank: int = Field(default=1, ge=0)
    graph_expansion: int = Field(default=1, ge=0)
    generation: int = Field(default=2, ge=0)
    verification: int = Field(default=1, ge=0)
    clarification: int = Field(default=1, ge=0)
    tokens: int = Field(default=4096, ge=0)
    wall_time_ms: int | None = Field(default=None, ge=0)

    @staticmethod
    def _budget_field(action: Action) -> str | None:
        return {
            Action.RETRIEVE: "retrieval",
            Action.REWRITE: "rewrite",
            Action.RERANK: "rerank",
            Action.EXPAND_GRAPH: "graph_expansion",
            Action.GENERATE: "generation",
            Action.REGENERATE: "generation",
            Action.VERIFY: "verification",
            Action.CLARIFY: "clarification",
        }.get(action)

    def total_nonterminal(self) -> int:
        return (
            self.retrieval + self.rewrite + self.rerank
            + self.graph_expansion + self.generation + self.verification
        )

    def allows(self, action: Action) -> bool:
        field = self._budget_field(action)
        if field is None:
            return True
        return getattr(self, field) > 0

    def consume(self, action: Action, tokens: int = 0) -> None:
        if tokens < 0:
            raise ValueError("tokens must be nonnegative")
        field = self._budget_field(action)
        if field is not None:
            current = getattr(self, field)
            if current <= 0:
                raise ValueError(f"{field} budget exhausted")
            setattr(self, field, current - 1)
        if tokens > self.tokens:
            raise ValueError("token budget exhausted")
        self.tokens -= tokens


class RiskEstimate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    raw_score: float = Field(default=0.0, ge=0.0, le=1.0)
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
