"""
Model checkpoint management utility.
"""
import os
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
import torch.optim as optim


def save_checkpoint(
    filepath: str,
    model: nn.Module,
    optimizer: Optional[optim.Optimizer] = None,
    epoch: int = 0,
    val_loss: float = 0.0,
    val_f1: float = 0.0,
    best_threshold: float = 0.5,
    config: Optional[Dict[str, Any]] = None
) -> None:
    """Atomically saves model checkpoint to disk."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "val_loss": val_loss,
        "val_f1": val_f1,
        "best_threshold": best_threshold,
        "config": config or {}
    }
    torch.save(state, filepath)


def load_checkpoint(
    filepath: str,
    model: nn.Module,
    optimizer: Optional[optim.Optimizer] = None
) -> Dict[str, Any]:
    """Loads model weights and training metadata from checkpoint file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint file '{filepath}' not found.")
    checkpoint = torch.load(filepath, map_location=torch.device("cpu"), weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer and checkpoint.get("optimizer_state_dict"):
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint
