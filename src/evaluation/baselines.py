"""
Non-LSTM Baseline predictors for link survival benchmarking.
Implements:
1. 2D Vector Kinematic Extrapolation Baseline.
2. 1D Radial Kinematic Baseline.
3. Distance Threshold Baseline (tuned on validation set).
"""
from typing import Dict, Any, Tuple
import numpy as np


class KinematicExtrapolationBaseline:
    """
    2D Vector Kinematic Extrapolation Baseline:
    Estimates relative velocity vector (dvx, dvy) from consecutive relative displacements:
      dvx = delta_x(t) - delta_x(t - 1)
      dvy = delta_y(t) - delta_y(t - 1)
    Projects 2D displacement at future horizon T:
      future_dx = delta_x(t) + dvx * T
      future_dy = delta_y(t) + dvy * T
      d_future = sqrt(future_dx^2 + future_dy^2)
    Predicts link survival (1) if d_future <= comm_range_m, else (0).
    Probability is mapped via a calibrated sigmoid centered at comm_range_m.
    """

    def __init__(self, comm_range_m: float = 1000.0, horizon_s: float = 30.0, scale_m: float = 100.0):
        self.comm_range_m = comm_range_m
        self.horizon_s = horizon_s
        self.scale_m = scale_m

    def compute_projected_distance(self, X_unscaled: np.ndarray) -> np.ndarray:
        """
        Computes 2D vector projected distance at t + horizon_s.
        X_unscaled: (N, seq_len, 7) where:
          feature 4 is delta_x_m
          feature 5 is delta_y_m
        """
        curr_dx = X_unscaled[:, -1, 4]
        curr_dy = X_unscaled[:, -1, 5]

        prev_dx = X_unscaled[:, -2, 4]
        prev_dy = X_unscaled[:, -2, 5]

        # Velocity components over dt = 1.0s
        dvx = curr_dx - prev_dx
        dvy = curr_dy - prev_dy

        future_dx = curr_dx + dvx * self.horizon_s
        future_dy = curr_dy + dvy * self.horizon_s

        d_projected = np.hypot(future_dx, future_dy)
        return d_projected

    def predict_proba(self, X_unscaled: np.ndarray) -> np.ndarray:
        """
        Probability of link survival:
        P = 1 / (1 + exp((d_future - comm_range_m) / scale))
        P = 0.5 when d_future == comm_range_m.
        P > 0.5 when d_future < comm_range_m (inside range).
        P < 0.5 when d_future > comm_range_m (outside range).
        """
        d_proj = self.compute_projected_distance(X_unscaled)
        # Margin: positive if inside range, negative if outside
        margin = self.comm_range_m - d_proj
        probs = 1.0 / (1.0 + np.exp(-margin / self.scale_m))
        return probs

    def predict(self, X_unscaled: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        probs = self.predict_proba(X_unscaled)
        return (probs >= threshold).astype(int)


class DistanceThresholdBaseline:
    """
    Distance Threshold Baseline:
    Predicts survival (1) if current distance <= d_threshold, else 0.
    Threshold d_threshold is tuned strictly on validation data to maximize F1.
    """

    def __init__(self, comm_range_m: float = 1000.0, scale_m: float = 100.0):
        self.comm_range_m = comm_range_m
        self.scale_m = scale_m
        self.best_threshold_dist = comm_range_m * 0.8

    def fit_threshold(self, X_unscaled: np.ndarray, y: np.ndarray) -> float:
        """Finds distance threshold that maximizes balanced accuracy / F1 on validation set."""
        curr_dist = X_unscaled[:, -1, 0]
        candidates = np.linspace(200.0, self.comm_range_m, 81)
        best_score = -1.0
        best_d = self.comm_range_m * 0.8

        for d_cand in candidates:
            preds = (curr_dist <= d_cand).astype(int)
            tp = np.sum((preds == 1) & (y == 1))
            fp = np.sum((preds == 1) & (y == 0))
            fn = np.sum((preds == 0) & (y == 1))
            tn = np.sum((preds == 0) & (y == 0))

            sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            # Balanced accuracy balances sensitivity and specificity
            score = 0.5 * (sens + spec)

            if score > best_score:
                best_score = score
                best_d = d_cand

        self.best_threshold_dist = float(best_d)
        return self.best_threshold_dist

    def predict_proba(self, X_unscaled: np.ndarray) -> np.ndarray:
        curr_dist = X_unscaled[:, -1, 0]
        margin = self.best_threshold_dist - curr_dist
        return 1.0 / (1.0 + np.exp(-margin / self.scale_m))

    def predict(self, X_unscaled: np.ndarray) -> np.ndarray:
        curr_dist = X_unscaled[:, -1, 0]
        return (curr_dist <= self.best_threshold_dist).astype(int)
