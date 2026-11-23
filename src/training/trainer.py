"""
Model training and validation loops with Early Stopping.
"""
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.training.checkpoints import save_checkpoint


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device
) -> float:
    """Trains model for one epoch and returns average training loss."""
    model.train()
    running_loss = 0.0
    total_samples = 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        # Gradient clipping to prevent exploding gradients in recurrent networks
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        running_loss += loss.item() * len(y_batch)
        total_samples += len(y_batch)

    return running_loss / total_samples if total_samples > 0 else 0.0


def evaluate_loss_and_predictions(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Evaluates validation loss and extracts predicted probabilities and targets."""
    model.eval()
    running_loss = 0.0
    total_samples = 0
    all_probs = []
    all_targets = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(X_batch)
            loss = criterion(logits, y_batch)

            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs)
            all_targets.extend(y_batch.cpu().numpy())

            running_loss += loss.item() * len(y_batch)
            total_samples += len(y_batch)

    avg_loss = running_loss / total_samples if total_samples > 0 else 0.0
    return avg_loss, np.array(all_probs), np.array(all_targets)


def train_lstm_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    checkpoint_path: str,
    max_epochs: int = 50,
    learning_rate: float = 0.001,
    weight_decay: float = 1e-4,
    patience: int = 10,
    pos_weight: Optional[float] = None,
    config: Optional[Dict[str, Any]] = None,
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """
    Executes complete training loop with Early Stopping and checkpointing.
    Returns: history dict {'train_loss': [...], 'val_loss': [...], 'best_epoch': int}
    """
    if device is None:
        device = torch.device("cpu")

    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    if pos_weight is not None:
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
    else:
        criterion = nn.BCEWithLogitsLoss()

    history = {
        "train_loss": [],
        "val_loss": [],
        "best_epoch": 0,
        "best_val_loss": float("inf")
    }

    best_val_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(1, max_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_probs, val_targets = evaluate_loss_and_predictions(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            history["best_val_loss"] = best_val_loss
            history["best_epoch"] = epoch
            epochs_no_improve = 0
            save_checkpoint(
                filepath=checkpoint_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                val_loss=val_loss,
                config=config
            )
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                # Early stopping triggered
                break

    return history
