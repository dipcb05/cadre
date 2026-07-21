from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from .calibration import CalibrationMethod, ProbabilityCalibrator
from .errors import CalibrationError
from .models import Head, RiskBundle, RiskEstimate


@dataclass
class Standardizer:
    mean_: np.ndarray | None = None
    scale_: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> "Standardizer":
        self.mean_ = np.nanmean(x, axis=0)
        scale = np.nanstd(x, axis=0)
        self.scale_ = np.where(scale > 1e-8, scale, 1.0)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise CalibrationError("standardizer is not fitted")
        return (x - self.mean_) / self.scale_


class MonotoneRiskHead:
    def __init__(
        self,
        feature_names: Iterable[str],
        *,
        threshold: float = 0.5,
        l2: float = 1e-3,
        calibration: CalibrationMethod = CalibrationMethod.ISOTONIC,
    ) -> None:
        self.feature_names = tuple(feature_names)
        self.threshold = float(threshold)
        self.l2 = float(l2)
        self.standardizer = Standardizer()
        self.calibrator = ProbabilityCalibrator(calibration)
        self.bias_: float | None = None
        self.weights_: np.ndarray | None = None

    def _matrix(self, rows: Iterable[Mapping[str, float | None]]) -> tuple[np.ndarray, np.ndarray]:
        values, missing = [], []
        for row in rows:
            row_values, row_missing = [], []
            for name in self.feature_names:
                value = row.get(name)
                is_missing = value is None or not np.isfinite(float(value))
                row_values.append(0.0 if is_missing else float(value))
                row_missing.append(float(is_missing))
            values.append(row_values)
            missing.append(row_missing)
        if not values:
            raise CalibrationError("feature rows must not be empty")
        return np.asarray(values, dtype=float), np.asarray(missing, dtype=float)

    def fit(
        self,
        feature_rows: Iterable[Mapping[str, float | None]],
        labels: Iterable[int],
        *,
        class_weight: tuple[float, float] = (1.0, 1.0),
    ) -> "MonotoneRiskHead":
        rows = list(feature_rows)
        y = np.asarray(list(labels), dtype=float)
        x, _ = self._matrix(rows)
        if len(x) != len(y) or len(x) == 0:
            raise CalibrationError("features and labels must have equal nonzero length")
        if len(np.unique(y)) < 2:
            raise CalibrationError("fit data must contain both classes")
        z = self.standardizer.fit(x).transform(x)
        n_features = z.shape[1]

        def unpack(params: np.ndarray) -> tuple[float, np.ndarray]:
            bias = float(params[0])
            logits = params[1:]
            logits -= np.max(logits)
            weights = np.exp(logits)
            weights /= np.sum(weights)
            return bias, weights

        sample_weight = np.where(y == 1, class_weight[1], class_weight[0])

        def objective(params: np.ndarray) -> float:
            bias, weights = unpack(params)
            p = np.clip(expit(bias + z @ weights), 1e-12, 1 - 1e-12)
            loss = -np.average(y * np.log(p) + (1 - y) * np.log(1 - p), weights=sample_weight)
            return float(loss + self.l2 * np.dot(weights, weights))

        result = minimize(objective, np.zeros(n_features + 1), method="L-BFGS-B")
        if not result.success:
            raise CalibrationError(f"risk-head fitting failed: {result.message}")
        self.bias_, self.weights_ = unpack(result.x)
        return self

    def fit_calibrator(
        self,
        feature_rows: Iterable[Mapping[str, float | None]],
        labels: Iterable[int],
    ) -> "MonotoneRiskHead":
        raw = self.predict_raw(feature_rows)
        self.calibrator.fit(raw, labels)
        return self

    def predict_raw(self, feature_rows: Iterable[Mapping[str, float | None]]) -> np.ndarray:
        if self.bias_ is None or self.weights_ is None:
            raise CalibrationError("risk head is not fitted")
        x, _ = self._matrix(feature_rows)
        z = self.standardizer.transform(x)
        return expit(self.bias_ + z @ self.weights_)

    def predict_proba(self, feature_rows: Iterable[Mapping[str, float | None]]) -> np.ndarray:
        return self.calibrator.predict(self.predict_raw(feature_rows))

    def estimate(self, features: Mapping[str, float | None], *, available: bool = True) -> RiskEstimate:
        missing = tuple(
            name
            for name in self.feature_names
            if features.get(name) is None or not np.isfinite(float(features.get(name, 0.0)))
        )
        full_features = dict(features)
        full_features["missing_indicator"] = float(bool(missing))
        if self.bias_ is None or self.weights_ is None:
            # Conservative deterministic fallback before fitting.
            values = [
                float(full_features.get(name, 0.0) or 0.0)
                for name in self.feature_names
            ]
            raw_score = float(np.clip(np.mean(values), 0.0, 1.0))
            probability = raw_score
        else:
            raw_score = float(self.predict_raw([full_features])[0])
            probability = float(self.calibrator.predict([raw_score])[0])
        return RiskEstimate(
            raw_score=raw_score,
            probability=probability,
            threshold=self.threshold,
            flagged=available and probability >= self.threshold,
            available=available,
            features={k: float(v or 0.0) for k, v in full_features.items()},
            missing=missing,
        )


class RiskModelBundle:
    def __init__(self, heads: Mapping[Head, MonotoneRiskHead]) -> None:
        missing = set(Head) - set(heads)
        if missing:
            raise ValueError(f"missing heads: {sorted(h.value for h in missing)}")
        self.heads = dict(heads)

    def estimate(
        self,
        feature_map: Mapping[Head, Mapping[str, float | None]],
        availability: Mapping[Head, bool],
    ) -> RiskBundle:
        estimates = {
            h: self.heads[h].estimate(feature_map[h], available=availability[h])
            for h in Head
        }
        return RiskBundle(
            instruction=estimates[Head.INSTRUCTION],
            retrieval=estimates[Head.RETRIEVAL],
            evidence=estimates[Head.EVIDENCE],
            generation=estimates[Head.GENERATION],
        )
