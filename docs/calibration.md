# Calibration

CADRE provides risk-head fitting and statistical calibration tools in `cadre.risk` and `cadre.calibration`.

---

## Risk Head Fitting (`MonotoneRiskHead`)

`MonotoneRiskHead` fits a monotonic logistic model over extracted feature vectors to produce calibrated stage risk probabilities.

```python
from cadre.risk import MonotoneRiskHead
from cadre.calibration import CalibrationMethod

feature_names = ["role_conflict", "boundary_violation", "script_anomaly"]
fit_rows = [
    {"role_conflict": 0.0, "boundary_violation": 0.0, "script_anomaly": 0.0},
    {"role_conflict": 0.8, "boundary_violation": 1.0, "script_anomaly": 0.3},
    {"role_conflict": 0.1, "boundary_violation": 0.0, "script_anomaly": 0.1},
    {"role_conflict": 1.0, "boundary_violation": 1.0, "script_anomaly": 0.8},
]
fit_labels = [0, 1, 0, 1]

head = MonotoneRiskHead(
    feature_names,
    threshold=0.45,
    calibration=CalibrationMethod.MONOTONE_PLATT,
)

# 1. Fit monotonic weight vector
head.fit(fit_rows, fit_labels)

# 2. Fit post-hoc calibrator
head.fit_calibrator(fit_rows, fit_labels)

# 3. Estimate risk for new feature dictionary
estimate = head.estimate({"role_conflict": 0.9, "boundary_violation": 1.0, "script_anomaly": 0.2})
print(f"Probability: {estimate.probability:.4f}, Flagged: {estimate.flagged}")
```

---

## Probability Calibrators (`ProbabilityCalibrator`)

Supported methods (`CalibrationMethod`):
1. **`IDENTITY`**: Pass-through raw scores.
2. **`ISOTONIC`**: Fits non-decreasing non-parametric isotonic regression (`sklearn.isotonic.IsotonicRegression`).
3. **`MONOTONE_PLATT`**: Fits monotonically constrained logistic regression (`slope > 0`) over logit-transformed probabilities using SciPy's L-BFGS-B optimizer.

---

## Distribution-Free Lineage Threshold Calibration

`calibrate_complete_lineage_threshold` calibrates risk thresholds with statistical coverage guarantees over grouped prompt lineages (e.g. multi-turn user conversations or document lineages) using the Clopper-Pearson exact binomial confidence bound.

```python
from cadre.calibration import calibrate_complete_lineage_threshold

probabilities = [0.1, 0.45, 0.82, 0.15, 0.91]
labels = [0, 0, 1, 0, 1]
lineage_ids = ["convo_1", "convo_1", "convo_2", "convo_3", "convo_3"]
grid = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

result = calibrate_complete_lineage_threshold(
    probabilities=probabilities,
    labels=labels,
    lineage_ids=lineage_ids,
    grid=grid,
    alpha=0.05,  # Target risk control bound (5%)
    delta=0.05,  # Confidence level parameter
)

print(f"Calibrated Threshold: {result.threshold}")
print(f"Upper Bound Error: {result.upper_bound:.4f}")
print(f"Always Flag Fallback: {result.always_flag}")
```

### `ThresholdCalibrationResult` Fields

- **`threshold`**: Maximum threshold grid value that satisfies upper bound constraint $\le \alpha$.
- **`upper_bound`**: Clopper-Pearson upper confidence bound for complete lineage misses.
- **`complete_misses`**: Number of positive lineage groups where all positive samples fell below threshold.
- **`positive_groups`**: Total count of positive lineage groups evaluated.
- **`always_flag`**: Boolean flag indicating no grid threshold satisfied $\le \alpha$; engine falls back to flagging all samples conservatively.
