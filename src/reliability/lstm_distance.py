"""
Adapter exposing the EXISTING LinkSurvivalPredictor through the ReliabilityModel interface.
The predictor, its weights, scaler and preprocessing are used unchanged; no threshold is
applied here (see policy.ThresholdPolicy).
"""
import json
from typing import Optional

import numpy as np

from src.models.predictor import LinkSurvivalPredictor
from src.reliability.base import ReliabilityModel


class LSTMDistanceSurvivalModel(ReliabilityModel):
    def __init__(self, checkpoint_path: str, model_config_path: str, scaler_path: str,
                 metadata_path: str, device: Optional[str] = "cpu", batch_size: int = 4096):
        with open(metadata_path) as f:
            meta = json.load(f)
        feat = meta["features"]
        self.name = "lstm_distance_survival_1000m"
        self.label_range_m = float(feat["comm_range_m"])
        self.history_window_s = int(feat["history_window_s"])
        self.horizon_s = int(feat["prediction_horizon_s"])
        self.feature_names = list(feat["names_in_order"])
        self.batch_size = int(batch_size)
        # Threshold is deliberately NOT read here: it is a policy decision, not a model property.
        self._predictor = LinkSurvivalPredictor(checkpoint_path, model_config_path, scaler_path,
                                                device=device)
        if self._predictor.scaler.feature_names and \
                list(self._predictor.scaler.feature_names) != self.feature_names:
            raise ValueError("Scaler feature order does not match checkpoint metadata feature order")

    @property
    def predictor(self) -> LinkSurvivalPredictor:
        return self._predictor

    def predict_proba(self, windows: np.ndarray) -> np.ndarray:
        windows = np.asarray(windows)
        if windows.ndim != 3 or windows.shape[1:] != (self.history_window_s, len(self.feature_names)):
            raise ValueError(f"expected (N, {self.history_window_s}, {len(self.feature_names)}), "
                             f"got {windows.shape}")
        if len(windows) == 0:
            return np.empty((0,), dtype=float)
        return np.concatenate([self._predictor.predict_batch_probabilities(windows[s:s + self.batch_size])
                               for s in range(0, len(windows), self.batch_size)])
