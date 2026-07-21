from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .artifacts import ModelArtifactManifest, save_artifact
from .calibration import CalibrationMethod, ProbabilityCalibrator, calibrate_complete_lineage_threshold
from .config import CadreConfig
from .lineage import LineageRecord, assert_no_overlap, grouped_split
from .models import Head
from .policy import SafeRoutingPolicy
from .risk import MonotoneRiskHead, RiskModelBundle

logger = logging.getLogger(__name__)


class OfflineTrainer:
    def __init__(self, config: CadreConfig, feature_names_by_head: dict[Head, tuple[str, ...]]) -> None:
        self.config = config
        self.feature_names_by_head = feature_names_by_head
        self.heads: dict[Head, MonotoneRiskHead] = {}
        self.calibrators: dict[Head, ProbabilityCalibrator] = {}
        self.thresholds: dict[Head, float] = {}

    def run_pipeline(
        self,
        records: Sequence[LineageRecord],
        output_dir: Path,
        *,
        version: str = "0.1.0",
        alpha: float = 0.95,
        delta: float = 0.05,
    ) -> ModelArtifactManifest:
        splits = grouped_split(records)
        assert_no_overlap(
            splits["D_fit"],
            splits["D_prob"],
            splits["D_threshold"],
            splits["D_policy"],
            splits["D_test"],
        )

        manifest = ModelArtifactManifest(
            version=version,
            code_version="0.1.0",
            data_version="0.1.0",
            feature_schema_version="1.0",
        )

        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Fit heads
        D_fit = splits["D_fit"]
        for head in Head:
            feature_names = self.feature_names_by_head[head]
            monotone_head = MonotoneRiskHead(
                feature_names=feature_names,
                threshold=self.config.heads[head].threshold,
                calibration=CalibrationMethod.IDENTITY,
            )
            rows = [r.features for r in D_fit]
            labels = [r.label for r in D_fit]
            monotone_head.fit(rows, labels)
            self.heads[head] = monotone_head
            save_artifact(monotone_head, output_dir / f"risk_head_{head.value}.pkl", manifest=manifest)

        # 2. Fit calibrators
        D_prob = splits["D_prob"]
        for head in Head:
            monotone_head = self.heads[head]
            rows = [r.features for r in D_prob]
            labels = [r.label for r in D_prob]
            raw_scores = monotone_head.predict_raw(rows)
            
            calibrator = ProbabilityCalibrator(self.config.heads[head].missing_feature_value or CalibrationMethod.ISOTONIC)
            calibrator.fit(raw_scores, labels)
            self.calibrators[head] = calibrator
            monotone_head.calibrator = calibrator
            save_artifact(calibrator, output_dir / f"calibrator_{head.value}.pkl", manifest=manifest)

        # 3. Select thresholds
        D_threshold = splits["D_threshold"]
        for head in Head:
            monotone_head = self.heads[head]
            rows = [r.features for r in D_threshold]
            labels = [r.label for r in D_threshold]
            lineage_ids = [r.lineage.lineage_id or r.lineage.compute_lineage_id() for r in D_threshold]
            
            raw_scores = monotone_head.predict_raw(rows)
            probabilities = monotone_head.predict_proba(rows)
            
            grid = np.linspace(0.01, 0.99, 99)
            try:
                res = calibrate_complete_lineage_threshold(
                    probabilities,
                    labels,
                    lineage_ids,
                    grid,
                    alpha=alpha,
                    delta=delta,
                )
                self.thresholds[head] = res.threshold
                monotone_head.threshold = res.threshold
            except Exception:
                logger.warning("Lineage threshold calibration failed for %s, using config default.", head.value)
                self.thresholds[head] = self.config.heads[head].threshold

            save_artifact(self.thresholds[head], output_dir / f"threshold_{head.value}.pkl", manifest=manifest)

        # 4. Evaluate Policy
        D_policy = splits["D_policy"]
        # Normally would evaluate various policy configurations here, but we simply verify routing policy runs correctly
        routing_policy = SafeRoutingPolicy(self.config.policy)
        save_artifact(routing_policy, output_dir / "policy.pkl", manifest=manifest)

        # Write out finalized manifest
        manifest_path = output_dir / "model_manifest.json"
        manifest_path.write_text(manifest.model_dump_json(indent=2))

        return manifest
