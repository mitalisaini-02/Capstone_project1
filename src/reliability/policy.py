"""
Threshold policy: turns a survival probability into an eligibility decision.

Separated from the predictor so the same model can be evaluated under different
thresholds (primary theta = 0.62, sensitivity theta = 0.65) without touching the model.
Values are read from the recorded Phase 1 / Phase 2 files, never hard-coded.
"""
import json
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ThresholdPolicy:
    theta: float
    source: str = "explicit"

    def validated(self) -> "ThresholdPolicy":
        """Range check used by the file loaders. (Not enforced in __init__ so that
        LinkSurvivalPredictor.is_link_reliable keeps its original behaviour for any threshold.)"""
        if not 0.0 <= float(self.theta) <= 1.0:
            raise ValueError(f"theta must be in [0, 1], got {self.theta}")
        return self

    def is_reliable(self, probability) -> bool:
        """Scalar decision: probability >= theta (same rule as LinkSurvivalPredictor.is_link_reliable)."""
        return bool(probability >= self.theta)

    def mask(self, probabilities) -> np.ndarray:
        """Vectorised decision for an array of probabilities."""
        return np.asarray(probabilities) >= self.theta

    @classmethod
    def primary_from_metadata(cls, checkpoint_metadata_path: str) -> "ThresholdPolicy":
        with open(checkpoint_metadata_path) as f:
            meta = json.load(f)
        return cls(float(meta["reliability_threshold"]["value"]), source=checkpoint_metadata_path).validated()

    @classmethod
    def sensitivity_from_file(cls, sensitivity_json_path: str) -> "ThresholdPolicy":
        with open(sensitivity_json_path) as f:
            meta = json.load(f)
        return cls(float(meta["sensitivity_threshold"]), source=sensitivity_json_path).validated()
