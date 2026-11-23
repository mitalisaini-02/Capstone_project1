"""
Plotting and visualization module for Stage 3 training and evaluation curves.
"""
import os
from typing import Dict, Any, List
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np


def plot_loss_curves(
    train_loss: List[float],
    val_loss: List[float],
    best_epoch: int,
    output_path: str
) -> None:
    """Plots and saves training vs validation loss curves."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    epochs = range(1, len(train_loss) + 1)

    plt.figure(figsize=(7, 4.5))
    plt.plot(epochs, train_loss, label="Training Loss", color="#1f77b4", lw=2)
    plt.plot(epochs, val_loss, label="Validation Loss", color="#ff7f0e", lw=2)
    if best_epoch > 0 and best_epoch <= len(val_loss):
        plt.axvline(x=best_epoch, color="red", linestyle="--", label=f"Best Model (Epoch {best_epoch})")

    plt.xlabel("Epoch")
    plt.ylabel("Binary Cross-Entropy Loss")
    plt.title("LSTM Training & Validation Loss Curves")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_confusion_matrix(
    cm: Dict[str, int],
    model_name: str,
    output_path: str
) -> None:
    """Plots and saves 2x2 confusion matrix."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    matrix = np.array([
        [cm["tn"], cm["fp"]],
        [cm["fn"], cm["tp"]]
    ])

    plt.figure(figsize=(5, 4))
    plt.imshow(matrix, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix — {model_name}")
    plt.colorbar()

    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["Predicted Break (0)", "Predicted Survive (1)"])
    plt.yticks(tick_marks, ["Actual Break (0)", "Actual Survive (1)"])

    thresh = matrix.max() / 2.0
    for i in range(2):
        for j in range(2):
            plt.text(
                j, i, format(matrix[i, j], "d"),
                ha="center", va="center",
                color="white" if matrix[i, j] > thresh else "black",
                fontsize=13
            )

    plt.ylabel("Ground Truth")
    plt.xlabel("Predicted Class")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
