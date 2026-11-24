"""
Common interface for reliability models.

A ReliabilityModel consumes UNSCALED feature windows of shape (N, history_window_s, n_features)
in the order given by `feature_names` and returns survival probabilities in [0, 1].
It declares the geometric definition its labels used (`label_range_m`, `horizon_s`), so the
caller can detect a mismatch with a technology's discovery range. It contains no threshold,
pricing, link-capacity or optimisation logic.

IMPORTANT: a distance-survival probability is NOT a guarantee of radio link quality.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List

import numpy as np


class ReliabilityModel(ABC):
    name: str = "unnamed"
    label_range_m: float
    history_window_s: int
    horizon_s: int
    feature_names: List[str]

    @abstractmethod
    def predict_proba(self, windows: np.ndarray) -> np.ndarray:
        """(N, history_window_s, n_features) unscaled windows -> (N,) probabilities in [0, 1]."""

    def describe(self) -> Dict[str, Any]:
        return {"name": self.name, "class": type(self).__name__, "label_range_m": self.label_range_m,
                "history_window_s": self.history_window_s, "horizon_s": self.horizon_s,
                "feature_names": list(self.feature_names),
                "semantics": "probability the pair stays within label_range_m for horizon_s "
                             "(geometric survival; not a radio link-quality guarantee)"}

    def range_mismatch(self, discovery_range_m: float) -> bool:
        """True if the technology's discovery range differs from this model's label range."""
        return abs(float(discovery_range_m) - float(self.label_range_m)) > 1e-9
