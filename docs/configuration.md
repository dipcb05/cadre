# Configuration

CADRE uses Pydantic models to define immutable, validated configuration objects.

---

## `CadreConfig`

`CadreConfig` is the top-level configuration object passed to `CadreEngine`.

```python
from cadre import CadreConfig, HeadConfig, PolicyConfig, Head

config = CadreConfig(
    heads={...},  # Dict mapping all Head enums to HeadConfig
    policy=PolicyConfig(),
    strict_provider_failures=False,
    delimiter_escape=True,
    log_sensitive_text=False,
)
```

### Fields

| Field | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `heads` | `dict[Head, HeadConfig]` | Required | Map containing configuration for all four risk heads (`instruction`, `retrieval`, `evidence`, `generation`). |
| `policy` | `PolicyConfig` | `PolicyConfig()` | Policy routing settings. |
| `strict_provider_failures` | `bool` | `False` | When `True`, raises `ProviderError` on adapter exceptions instead of failing closed with `ABSTAIN`. |
| `delimiter_escape` | `bool` | `True` | Escapes prompt delimiter tags during context serialization. |
| `log_sensitive_text` | `bool` | `False` | Controls whether sensitive user/document text is emitted to logs. |

### Factory Method

Use `CadreConfig.safe_default()` to instantiate standard default thresholds and feature requirements for all heads.

---

## `HeadConfig`

Configures feature names and risk decision thresholds for an individual stage head.

```python
from cadre import HeadConfig

head_cfg = HeadConfig(
    feature_names=("uncertainty", "role_conflict", "boundary_violation", "missing_indicator"),
    threshold=0.45,
    missing_feature_value=0.0,
    always_flag_if_unavailable=False,
)
```

| Field | Type | Default | Constraints / Description |
| :--- | :--- | :--- | :--- |
| `feature_names` | `tuple[str, ...]` | Required | Unique, non-empty tuple of feature names expected by this risk head. |
| `threshold` | `float` | `0.5` | Risk threshold in range `[0.0, 1.0]`. If predicted risk probability exceeds this threshold, the head flags risk. |
| `missing_feature_value` | `float` | `0.0` | Fallback value used when feature extraction omits a key. |
| `always_flag_if_unavailable` | `bool` | `False` | Forces `flagged=True` if stage inputs are missing/unavailable. |

---

## `PolicyConfig`

Controls routing threshold decisions inside `SafeRoutingPolicy`.

```python
from cadre import PolicyConfig, Action

policy_cfg = PolicyConfig(
    max_retrieval_depth=5,
    accept_requires_generation=True,
    contamination_upper_bound=0.05,
    safe_actions=(Action.CLARIFY, Action.ABSTAIN, Action.REFUSE),
    refuse_on_instruction_risk=True,
    clarify_on_missing_evidence=True,
)
```

| Field | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `max_retrieval_depth` | `int` | `5` | Maximum number of documents `k` requested during `RETRIEVE` or `RERANK`. |
| `accept_requires_generation` | `bool` | `True` | Ensures generation step completed prior to `ACCEPT`. |
| `contamination_upper_bound` | `float` | `0.05` | Target statistical bound for risk control. |
| `safe_actions` | `tuple[Action, ...]` | `(CLARIFY, ABSTAIN, REFUSE)` | Non-accept terminal actions. |
| `refuse_on_instruction_risk` | `bool` | `True` | Selects `Action.REFUSE` over `Action.ABSTAIN` when instruction risk is flagged. |
| `clarify_on_missing_evidence` | `bool` | `True` | Selects `Action.CLARIFY` over `Action.ABSTAIN` when evidence is missing. |

---

## `RuntimeBudget`

Tracks execution resource bounds. Passed to `engine.run(context, budget=...)`.

```python
from cadre import RuntimeBudget

budget = RuntimeBudget(
    retrieval=2,     # Maximum retrieval actions (RETRIEVE, REWRITE, RERANK, GRAPH)
    generation=2,    # Maximum generation actions (GENERATE, REGENERATE)
    verification=1,  # Maximum verification actions (VERIFY)
    tokens=4096,     # Maximum LLM generation token budget
)
```

When an action executes, `budget.consume(action, tokens=...)` decrements the respective counter. If a budget counter hits 0, CADRE prevents further nonterminal actions of that type and falls back to terminal actions (`CLARIFY` or `ABSTAIN`).
