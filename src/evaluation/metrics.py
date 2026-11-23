"""
Evaluation metrics calculation, ROC-AUC, PR-AUC, and validation threshold tuning.
"""
from typing import Dict, Any, Tuple
import numpy as np


def compute_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Computes exact Area Under the ROC Curve (ROC-AUC) using Mann-Whitney U rank statistic."""
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)

    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)

    if n_pos == 0 or n_neg == 0:
        return 0.5

    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(y_score) + 1)

    unique_scores, counts = np.unique(y_score, return_counts=True)
    for score, count in zip(unique_scores, counts):
        if count > 1:
            tie_mask = (y_score == score)
            ranks[tie_mask] = np.mean(ranks[tie_mask])

    pos_rank_sum = np.sum(ranks[y_true == 1])
    u_stat = pos_rank_sum - (n_pos * (n_pos + 1)) / 2.0
    auc = u_stat / (n_pos * n_neg)
    return float(np.clip(auc, 0.0, 1.0))


def compute_pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Computes Area Under the Precision-Recall Curve (PR-AUC) via trapezoidal integration."""
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)

    order = np.argsort(-y_score)
    y_sorted = y_true[order]

    tp_cum = np.cumsum(y_sorted == 1)
    fp_cum = np.cumsum(y_sorted == 0)

    n_pos = np.sum(y_true == 1)
    if n_pos == 0:
        return 0.0

    recalls = tp_cum / n_pos
    precisions = tp_cum / (tp_cum + fp_cum)

    # Insert boundary point (recall=0, precision=first precision or 1.0)
    recalls = np.concatenate([[0.0], recalls])
    precisions = np.concatenate([[precisions[0] if len(precisions) > 0 else 1.0], precisions])

    # Trapezoidal integration along recall axis
    try:
        from scipy.integrate import trapezoid
        pr_auc = float(trapezoid(precisions, recalls))
    except ImportError:
        # Fallback to manual trapezoid formula: sum 0.5 * (y1 + y2) * (x2 - x1)
        dx = np.diff(recalls)
        pr_auc = float(np.sum(0.5 * (precisions[:-1] + precisions[1:]) * dx))
    return float(np.clip(pr_auc, 0.0, 1.0))


def compute_classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, Any]:
    """Computes standard binary classification metrics at a given decision threshold."""
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)

    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    accuracy = (tp + tn) / len(y_true) if len(y_true) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    balanced_acc = 0.5 * (recall + specificity)
    roc_auc = compute_roc_auc(y_true, y_prob)
    pr_auc = compute_pr_auc(y_true, y_prob)
    brier_score = float(np.mean((y_prob - y_true) ** 2))

    return {
        "threshold": round(threshold, 4),
        "predicted_positive_count": int(np.sum(y_pred == 1)),
        "predicted_negative_count": int(np.sum(y_pred == 0)),
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "specificity": round(specificity, 4),
        "f1_score": round(f1, 4),
        "balanced_accuracy": round(balanced_acc, 4),
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "brier_score": round(brier_score, 4),
        "confusion_matrix": {
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn
        }
    }


def tune_classification_threshold(
    y_val: np.ndarray,
    val_probs: np.ndarray,
    target_metric: str = "balanced_accuracy"
) -> Tuple[float, float]:
    """
    Finds optimal probability threshold in [0.15, 0.85] on validation data.
    Defaults to 'balanced_accuracy' to avoid majority-class trivial collapse.
    """
    thresholds = np.linspace(0.15, 0.85, 71)
    best_thresh = 0.50
    best_val = -1.0

    for th in thresholds:
        m = compute_classification_metrics(y_val, val_probs, threshold=th)
        if target_metric == "balanced_accuracy":
            score = m["balanced_accuracy"]
        elif target_metric == "f1":
            # Penalize degenerate zero-specificity solutions
            score = m["f1_score"] if m["specificity"] > 0.05 else 0.0
        else:
            score = m["accuracy"]

        if score > best_val:
            best_val = score
            best_thresh = float(th)

    return best_thresh, best_val
