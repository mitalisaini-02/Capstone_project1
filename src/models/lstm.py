"""
PyTorch LSTM architecture for PUE-CUE Link-Survival Prediction.
"""
from typing import Tuple, Dict, Any
import torch
import torch.nn as nn


class LinkSurvivalLSTM(nn.Module):
    """
    Stacked LSTM with a multi-layer classification head for link-survival prediction.
    Outputs raw logits for numerical stability with BCEWithLogitsLoss.
    """

    def __init__(
        self,
        input_dim: int = 7,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = False
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional
        )

        head_input_dim = hidden_dim * self.num_directions
        self.classifier = nn.Sequential(
            nn.Linear(head_input_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Input x: Tensor of shape (batch_size, seq_len, input_dim)
        Output: Raw logits Tensor of shape (batch_size,)
        """
        # lstm_out shape: (batch_size, seq_len, hidden_dim * num_directions)
        lstm_out, _ = self.lstm(x)
        # Use final timestep output
        last_step = lstm_out[:, -1, :]
        logits = self.classifier(last_step).squeeze(-1)
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Computes survival probabilities in [0.0, 1.0] using sigmoid."""
        with torch.no_grad():
            logits = self.forward(x)
            return torch.sigmoid(logits)
