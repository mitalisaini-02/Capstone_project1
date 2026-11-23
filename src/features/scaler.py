"""
Feature standardization module: fit exclusively on training sequences.
"""
import json
import os
from typing import Dict, List, Optional
import numpy as np


class SequenceStandardScaler:
    """Standardizes 3D sequence tensors (N, seq_len, n_features) using training stats."""

    def __init__(self, feature_names: Optional[List[str]] = None):
        self.feature_names = feature_names or []
        self.mean_: Optional[np.ndarray] = None
        self.scale_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "SequenceStandardScaler":
        """
        Computes mean and std across (N * seq_len) for each feature.
        X shape: (N, seq_len, n_features)
        """
        if X.ndim != 3:
            raise ValueError(f"Expected 3D tensor (N, seq_len, n_features), got shape {X.shape}")

        n_features = X.shape[-1]
        flat = X.reshape(-1, n_features)

        self.mean_ = np.mean(flat, axis=0)
        self.scale_ = np.std(flat, axis=0)
        # Avoid division by zero for constant features
        self.scale_[self.scale_ < 1e-7] = 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Applies z-score normalization: (X - mean) / scale."""
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Scaler has not been fitted yet.")
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def to_dict(self) -> Dict[str, any]:
        return {
            "feature_names": self.feature_names,
            "mean": self.mean_.tolist() if self.mean_ is not None else [],
            "scale": self.scale_.tolist() if self.scale_ is not None else []
        }

    def save_json(self, filepath: str) -> None:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_json(cls, filepath: str) -> "SequenceStandardScaler":
        with open(filepath, "r") as f:
            data = json.load(f)
        scaler = cls(feature_names=data.get("feature_names"))
        scaler.mean_ = np.array(data["mean"], dtype=np.float32)
        scaler.scale_ = np.array(data["scale"], dtype=np.float32)
        return scaler
