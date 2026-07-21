from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .models import Action, Head


class HeadConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    feature_names: tuple[str, ...]
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    missing_feature_value: float = 0.0
    always_flag_if_unavailable: bool = False

    @model_validator(mode="after")
    def validate_names(self) -> "HeadConfig":
        if not self.feature_names:
            raise ValueError("feature_names must not be empty")
        if len(set(self.feature_names)) != len(self.feature_names):
            raise ValueError("feature_names must be unique")
        return self


class PolicyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    max_retrieval_depth: int = Field(default=5, ge=1, le=100)
    accept_requires_generation: bool = True
    contamination_upper_bound: float = Field(default=0.05, ge=0.0, le=1.0)
    safe_actions: tuple[Action, ...] = (
        Action.CLARIFY,
        Action.ABSTAIN,
        Action.REFUSE,
    )
    refuse_on_instruction_risk: bool = True
    clarify_on_missing_evidence: bool = True


class CadreConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    heads: dict[Head, HeadConfig]
    policy: PolicyConfig = PolicyConfig()
    strict_provider_failures: bool = False
    delimiter_escape: bool = True
    log_sensitive_text: bool = False

    @model_validator(mode="after")
    def all_heads_present(self) -> "CadreConfig":
        missing = set(Head) - set(self.heads)
        if missing:
            raise ValueError(f"missing head configs: {sorted(x.value for x in missing)}")
        return self

    @classmethod
    def safe_default(cls) -> "CadreConfig":
        return cls(
            heads={
                Head.INSTRUCTION: HeadConfig(
                    feature_names=(
                        "uncertainty",
                        "task_inconsistency",
                        "role_conflict",
                        "boundary_violation",
                        "script_anomaly",
                        "missing_indicator",
                    ),
                    threshold=0.45,
                ),
                Head.RETRIEVAL: HeadConfig(
                    feature_names=(
                        "facet_gap",
                        "rank_instability",
                        "source_concentration",
                        "query_evidence_mismatch",
                        "poison_likelihood",
                        "language_mismatch",
                        "missing_indicator",
                    ),
                    threshold=0.50,
                ),
                Head.EVIDENCE: HeadConfig(
                    feature_names=(
                        "support_gap",
                        "contradiction_density",
                        "provenance_gap",
                        "staleness",
                        "fragmentation",
                        "missing_indicator",
                    ),
                    threshold=0.50,
                ),
                Head.GENERATION: HeadConfig(
                    feature_names=(
                        "unsupported_insufficient",
                        "unsupported_contradicted",
                        "unsupported_conflicting",
                        "dependency_unsupported",
                        "no_claim",
                        "segmentation_failure",
                        "verifier_failure",
                        "missing_indicator",
                    ),
                    threshold=0.45,
                    always_flag_if_unavailable=False,
                ),
            }
        )
