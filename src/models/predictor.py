"""
Stage 4 Clean Prediction Interface: Encapsulates LSTM model, feature scaler, and probability inference.
"""
import os
import json
from typing import Dict, Any, Union, Optional
import logging
import numpy as np
import torch

from src.models.lstm import LinkSurvivalLSTM
from src.features.scaler import SequenceStandardScaler

logger = logging.getLogger(__name__)


class LinkSurvivalPredictor:
    """
    Decoupled inference wrapper for MEC coordinator integration.
    Loads trained weights, config, and scaler, accepting raw unscaled feature sequences.
    """

    def __init__(
        self,
        checkpoint_path: str = "outputs/checkpoints/best_model.pt",
        config_path: str = "outputs/checkpoints/model_config.json",
        scaler_path: str = "outputs/checkpoints/scaler.json",
        device: Optional[str] = None,
        metadata_path: Optional[str] = None
    ):
        """
        metadata_path: optional checkpoint_metadata.json (written by
        scripts/sweep_reliability_threshold.py). When given, the decision
        threshold is taken from its "reliability_threshold" entry instead of
        the checkpoint's stored "best_threshold". When omitted, behaviour is
        unchanged.
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint '{checkpoint_path}' not found.")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Model config '{config_path}' not found.")
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler '{scaler_path}' not found.")

        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))

        # 1. Load config
        with open(config_path, "r") as f:
            self.model_config = json.load(f)

        # 2. Load scaler
        self.scaler = SequenceStandardScaler.load_json(scaler_path)

        # 3. Instantiate model
        self.model = LinkSurvivalLSTM(
            input_dim=self.model_config.get("input_dim", 7),
            hidden_dim=self.model_config.get("hidden_dim", 64),
            num_layers=self.model_config.get("num_layers", 2),
            dropout=self.model_config.get("dropout", 0.2),
            bidirectional=self.model_config.get("bidirectional", False)
        )

        # 4. Load weights
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        self.tuned_threshold = checkpoint.get("best_threshold", 0.50)
        self.threshold_source = "checkpoint"
        self.metadata: Optional[Dict[str, Any]] = None

        if metadata_path is not None:
            if not os.path.exists(metadata_path):
                raise FileNotFoundError(f"Checkpoint metadata '{metadata_path}' not found.")
            with open(metadata_path, "r") as f:
                self.metadata = json.load(f)
            rel = self.metadata.get("reliability_threshold", {})
            if "value" not in rel:
                raise ValueError(f"'{metadata_path}' has no reliability_threshold.value entry.")
            self.tuned_threshold = float(rel["value"])
            self.threshold_source = "metadata"
        elif "best_threshold" not in checkpoint or abs(self.tuned_threshold - 0.5) < 1e-12:
            logger.warning(
                "Checkpoint '%s' stores the default threshold 0.5, which may not be the tuned "
                "value. Pass metadata_path=... to use the validated reliability threshold.",
                checkpoint_path,
            )

    def predict_probability(self, history_sequence: np.ndarray) -> float:
        """
        Accepts unscaled historical sequence of shape (seq_len, n_features) or (1, seq_len, n_features).
        Returns predicted link-survival probability in [0.0, 1.0].
        """
        if history_sequence.ndim == 2:
            history_sequence = np.expand_dims(history_sequence, axis=0)

        # Standardize using fitted training scaler
        scaled_seq = self.scaler.transform(history_sequence)
        tensor_x = torch.tensor(scaled_seq, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            prob = self.model.predict_proba(tensor_x).cpu().item()
        return float(prob)

    def predict_batch_probabilities(self, batch_sequences: np.ndarray) -> np.ndarray:
        """
        Accepts unscaled batch of shape (batch_size, seq_len, n_features).
        Returns ndarray of survival probabilities in [0.0, 1.0].
        """
        scaled_batch = self.scaler.transform(batch_sequences)
        tensor_x = torch.tensor(scaled_batch, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            probs = self.model.predict_proba(tensor_x).cpu().numpy()
        return probs.astype(float)

    def is_link_reliable(self, probability: float, threshold: Optional[float] = None) -> bool:
        """Checks if survival probability satisfies reliability threshold."""
        th = threshold if threshold is not None else self.tuned_threshold
        return bool(probability >= th)
