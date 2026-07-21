from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit
from scipy.stats import beta
from sklearn.isotonic import IsotonicRegression

from .errors import CalibrationError


class CalibrationMethod(str, Enum):
    IDENTITY = "identity"
    ISOTONIC = "isotonic"
    MONOTONE_PLATT = "monotone_platt"


class ProbabilityCalibrator:
    def __init__(self, method: CalibrationMethod = CalibrationMethod.ISOTONIC) -> None:
        self.method = method
        self._model = None

    def fit(self, scores: Iterable[float], labels: Iterable[int]) -> "ProbabilityCalibrator":
        x = np.asarray(list(scores), dtype=float)
        y = np.asarray(list(labels), dtype=int)
        if x.ndim != 1 or y.ndim != 1 or len(x) != len(y) or len(x) == 0:
            raise CalibrationError("scores and labels must be nonempty one-dimensional arrays")
        if not np.all(np.isfinite(x)):
            raise CalibrationError("scores contain non-finite values")
        if not set(np.unique(y)).issubset({0, 1}):
            raise CalibrationError("labels must be binary")
        x = np.clip(x, 1e-8, 1 - 1e-8)

        if self.method is CalibrationMethod.IDENTITY:
            self._model = "identity"
        elif self.method is CalibrationMethod.ISOTONIC:
            model = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip")
            model.fit(x, y)
            self._model = model
        elif self.method is CalibrationMethod.MONOTONE_PLATT:
            z = logit(x)

            def objective(params: np.ndarray) -> float:
                slope = np.exp(params[0])
                intercept = params[1]
                p = np.clip(expit(slope * z + intercept), 1e-12, 1 - 1e-12)
                return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

            result = minimize(objective, np.array([0.0, 0.0]), method="L-BFGS-B")
            if not result.success:
                raise CalibrationError(f"monotone Platt fitting failed: {result.message}")
            self._model = (float(np.exp(result.x[0])), float(result.x[1]))
        else:
            raise CalibrationError(f"unsupported calibration method: {self.method}")
        return self

    def predict(self, scores: Iterable[float]) -> np.ndarray:
        if self._model is None:
            raise CalibrationError("calibrator is not fitted")
        x = np.clip(np.asarray(list(scores), dtype=float), 1e-8, 1 - 1e-8)
        if self._model == "identity":
            return x
        if isinstance(self._model, IsotonicRegression):
            return np.asarray(self._model.predict(x), dtype=float)
        slope, intercept = self._model
        return expit(slope * logit(x) + intercept)


def clopper_pearson_upper(errors: int, groups: int, gamma: float) -> float:
    if groups <= 0:
        return 1.0
    if not 0 < gamma < 1:
        raise ValueError("gamma must be in (0, 1)")
    if not 0 <= errors <= groups:
        raise ValueError("errors must satisfy 0 <= errors <= groups")
    if errors == groups:
        return 1.0
    return float(beta.ppf(1 - gamma, errors + 1, groups - errors))


@dataclass(frozen=True)
class ThresholdCalibrationResult:
    threshold: float
    upper_bound: float
    complete_misses: int
    positive_groups: int
    always_flag: bool


def calibrate_complete_lineage_threshold(
    probabilities: Iterable[float],
    labels: Iterable[int],
    lineage_ids: Iterable[str],
    grid: Iterable[float],
    *,
    alpha: float,
    delta: float,
) -> ThresholdCalibrationResult:
    p = np.asarray(list(probabilities), dtype=float)
    y = np.asarray(list(labels), dtype=int)
    groups = np.asarray(list(lineage_ids), dtype=object)
    thresholds = sorted(set(float(t) for t in grid))
    if len(p) != len(y) or len(p) != len(groups) or len(p) == 0:
        raise CalibrationError("probabilities, labels, and lineage_ids must have equal nonzero length")
    if not thresholds:
        raise CalibrationError("threshold grid must not be empty")
    if not 0 < alpha < 1 or not 0 < delta < 1:
        raise CalibrationError("alpha and delta must be in (0, 1)")
    if not np.all((0 <= p) & (p <= 1)):
        raise CalibrationError("probabilities must be in [0, 1]")

    positive_groups = []
    for group in np.unique(groups):
        mask = (groups == group) & (y == 1)
        if np.any(mask):
            positive_groups.append((group, p[mask]))
    g_count = len(positive_groups)
    if g_count == 0:
        raise CalibrationError("no positive lineage groups available")

    gamma = delta / len(thresholds)
    feasible: list[tuple[float, float, int]] = []
    for threshold in thresholds:
        misses = sum(int(np.max(values) < threshold) for _, values in positive_groups)
        upper = clopper_pearson_upper(misses, g_count, gamma)
        if upper <= alpha:
            feasible.append((threshold, upper, misses))

    if not feasible:
        return ThresholdCalibrationResult(
            threshold=0.0,
            upper_bound=0.0,
            complete_misses=0,
            positive_groups=g_count,
            always_flag=True,
        )

    threshold, upper, misses = max(feasible, key=lambda item: item[0])
    return ThresholdCalibrationResult(
        threshold=threshold,
        upper_bound=upper,
        complete_misses=misses,
        positive_groups=g_count,
        always_flag=False,
    )
