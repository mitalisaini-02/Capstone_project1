"""
Kinematic-extrapolation reliability model (no training): wraps the existing
KinematicExtrapolationBaseline behind the ReliabilityModel interface. Demonstrates that a
reliability model can be swapped without touching the LSTM or the marketplace.
"""
from typing import List

import numpy as np

from src.evaluation.baselines import KinematicExtrapolationBaseline
from src.reliability.base import ReliabilityModel


class KinematicSurvivalModel(ReliabilityModel):
    def __init__(self, label_range_m: float, horizon_s: int, history_window_s: int,
                 feature_names: List[str], scale_m: float = 100.0):
        if feature_names[4:6] != ["delta_x_m", "delta_y_m"]:
            raise ValueError("KinematicExtrapolationBaseline expects delta_x_m, delta_y_m at positions 4 and 5")
        self.name = f"kinematic_extrapolation_{int(label_range_m)}m"
        self.label_range_m = float(label_range_m)
        self.horizon_s = int(horizon_s)
        self.history_window_s = int(history_window_s)
        self.feature_names = list(feature_names)
        self._baseline = KinematicExtrapolationBaseline(comm_range_m=label_range_m,
                                                        horizon_s=float(horizon_s), scale_m=scale_m)

    def predict_proba(self, windows: np.ndarray) -> np.ndarray:
        return self._baseline.predict_proba(np.asarray(windows, dtype=float))
