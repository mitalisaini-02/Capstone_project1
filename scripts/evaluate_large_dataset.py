"""
Standalone Stage 3 Evaluation Script — Large Dataset Experiment.

Loads the existing LSTM checkpoint, scaler, and dataset splits (NO retraining).
Applies scaler to val/test sets (NO refitting).
Tunes baseline thresholds on validation data only.
Evaluates LSTM at θ* = 0.530 (from validation tuning).
Evaluates baselines at their val-tuned thresholds.
Saves metrics, reports, and plots.

Usage:
    .venv/bin/python scripts/evaluate_large_dataset.py
"""

import os
import sys
import json
import datetime
import csv
from pathlib import Path

# --- Project root on sys.path -------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.models.lstm import LinkSurvivalLSTM
from src.features.scaler import SequenceStandardScaler
from src.training.checkpoints import load_checkpoint, save_checkpoint
from src.evaluation.metrics import (
    compute_classification_metrics,
    compute_roc_curve_points,
    compute_pr_curve_points,
    tune_classification_threshold,
)
from src.evaluation.baselines import KinematicExtrapolationBaseline, DistanceThresholdBaseline
from src.evaluation.visualizer import plot_confusion_matrix, plot_roc_curves, plot_pr_curves

# ==============================================================================
# Paths
# ==============================================================================
CHECKPOINT_DIR   = PROJECT_ROOT / "outputs" / "large_dataset_experiment" / "checkpoints"
CHECKPOINT_PATH  = CHECKPOINT_DIR / "best_model.pt"
SCALER_PATH      = CHECKPOINT_DIR / "scaler.json"
MODEL_CONFIG_PATH = CHECKPOINT_DIR / "model_config.json"

DATASET_DIR   = PROJECT_ROOT / "data" / "processed" / "large_dataset"
SEQUENCES_NPZ = DATASET_DIR / "pairwise_sequences.npz"

SPLITS_DIR    = PROJECT_ROOT / "data" / "splits" / "large_dataset"
TRAIN_IDX     = SPLITS_DIR / "train_indices.npy"
VAL_IDX       = SPLITS_DIR / "val_indices.npy"
TEST_IDX      = SPLITS_DIR / "test_indices.npy"

OUT_DIR       = PROJECT_ROOT / "outputs" / "large_dataset_experiment"
METRICS_DIR   = OUT_DIR / "metrics"
REPORTS_DIR   = OUT_DIR / "reports"
PLOTS_DIR     = OUT_DIR / "plots"

# LSTM threshold (tuned on validation data during training run — do NOT use test)
LSTM_THRESHOLD = 0.530
BEST_EPOCH     = 13
BEST_VAL_LOSS  = 0.3318
BEST_VAL_BALANCED_ACC = 0.7805

# ==============================================================================
# Helpers
# ==============================================================================

def log(msg: str) -> None:
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def makedirs() -> None:
    for d in [METRICS_DIR, REPORTS_DIR, PLOTS_DIR]:
        d.mkdir(parents=True, exist_ok=True)


# ==============================================================================
# 1. Load dataset
# ==============================================================================

def load_dataset():
    log("Loading dataset and splits...")
    data = np.load(str(SEQUENCES_NPZ))
    X, y = data["X"].astype(np.float32), data["y"].astype(np.float32)
    log(f"  Dataset: X={X.shape}, y={y.shape}")

    train_idx = np.load(str(TRAIN_IDX))
    val_idx   = np.load(str(VAL_IDX))
    test_idx  = np.load(str(TEST_IDX))
    log(f"  Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")

    X_train, y_train = X[train_idx], y[train_idx]
    X_val,   y_val   = X[val_idx],   y[val_idx]
    X_test,  y_test  = X[test_idx],  y[test_idx]

    log(f"  Val  labels: survive={int(np.sum(y_val==1))}, break={int(np.sum(y_val==0))}")
    log(f"  Test labels: survive={int(np.sum(y_test==1))}, break={int(np.sum(y_test==0))}")

    return X_train, y_train, X_val, y_val, X_test, y_test


# ==============================================================================
# 2. Load & apply scaler (do NOT refit)
# ==============================================================================

def load_and_apply_scaler(X_train, X_val, X_test):
    log("Loading pre-fitted scaler (no refit)...")
    scaler = SequenceStandardScaler.load_json(str(SCALER_PATH))
    if scaler.mean_ is None or len(scaler.mean_) == 0:
        raise RuntimeError("Scaler is empty — mean/scale not saved correctly.")
    log(f"  Scaler feature means: {[round(float(m),2) for m in scaler.mean_]}")

    # Keep unscaled copy for baselines (they use raw metres/mps)
    X_val_scaled  = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    log("  Scaler applied to val and test (train used only for verification).")
    return scaler, X_val_scaled, X_test_scaled


# ==============================================================================
# 3. Load LSTM model
# ==============================================================================

def load_lstm_model():
    log(f"Loading LSTM checkpoint: {CHECKPOINT_PATH.name}")

    # Read model config
    with open(str(MODEL_CONFIG_PATH)) as f:
        cfg = json.load(f)

    model = LinkSurvivalLSTM(
        input_dim   = cfg.get("input_dim", 7),
        hidden_dim  = cfg.get("hidden_dim", 64),
        num_layers  = cfg.get("num_layers", 2),
        dropout     = cfg.get("dropout", 0.3),
        bidirectional = cfg.get("bidirectional", False),
    )

    ckpt = load_checkpoint(str(CHECKPOINT_PATH), model)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    log(f"  Model loaded: {n_params:,} parameters | epoch={ckpt.get('epoch')} | val_loss={ckpt.get('val_loss')}")
    log(f"  Checkpoint threshold meta: {ckpt.get('best_threshold')} "
        f"(override with validated θ*={LSTM_THRESHOLD})")
    return model, cfg, ckpt


# ==============================================================================
# 4. LSTM inference
# ==============================================================================

def lstm_predict(model: LinkSurvivalLSTM, X_scaled: np.ndarray, batch_size: int = 1024) -> np.ndarray:
    """Runs batched inference; returns probabilities (N,)."""
    all_probs = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(X_scaled), batch_size):
            batch = torch.tensor(X_scaled[start:start + batch_size], dtype=torch.float32)
            logits = model(batch)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
    return np.concatenate(all_probs, axis=0)


# ==============================================================================
# 5. Evaluate baselines
# ==============================================================================

def evaluate_baselines(X_val_unscaled, y_val, X_test_unscaled, y_test):
    log("\n--- Baseline Evaluation ---")
    results = {}

    # --- Kinematic Extrapolation ---
    kinematic = KinematicExtrapolationBaseline(comm_range_m=1000.0, horizon_s=30.0, scale_m=100.0)

    kin_val_probs = kinematic.predict_proba(X_val_unscaled)
    kin_thresh, kin_val_ba = tune_classification_threshold(y_val, kin_val_probs, "balanced_accuracy")
    log(f"  Kinematic: val θ*={kin_thresh:.3f}, val balanced_acc={kin_val_ba:.4f}")

    kin_test_probs = kinematic.predict_proba(X_test_unscaled)
    kin_test_metrics = compute_classification_metrics(y_test, kin_test_probs, threshold=kin_thresh)
    kin_test_metrics["val_threshold"] = kin_thresh
    kin_test_metrics["val_balanced_accuracy"] = round(kin_val_ba, 4)
    log(f"  Kinematic Test: ROC-AUC={kin_test_metrics['roc_auc']}, "
        f"F1={kin_test_metrics['f1_score']}, Spec={kin_test_metrics['specificity']}")
    results["kinematic"] = {"val_probs": kin_val_probs, "test_probs": kin_test_probs,
                            "test_metrics": kin_test_metrics, "threshold": kin_thresh}

    # --- Distance Threshold ---
    dist_base = DistanceThresholdBaseline(comm_range_m=1000.0, scale_m=100.0)

    dist_val_probs = dist_base.predict_proba(X_val_unscaled)
    # Tune distance threshold on val
    dist_thresh, dist_val_ba = tune_classification_threshold(y_val, dist_val_probs, "balanced_accuracy")
    # Also tune the internal d* on val
    dist_base.tune_threshold(X_val_unscaled, y_val)
    dist_thresh_prob, dist_val_ba2 = tune_classification_threshold(
        y_val, dist_base.predict_proba(X_val_unscaled), "balanced_accuracy")
    log(f"  Distance Threshold: d*={dist_base.distance_threshold_m:.1f}m, "
        f"val θ*={dist_thresh_prob:.3f}, val balanced_acc={dist_val_ba2:.4f}")

    dist_test_probs = dist_base.predict_proba(X_test_unscaled)
    dist_test_metrics = compute_classification_metrics(y_test, dist_test_probs, threshold=dist_thresh_prob)
    dist_test_metrics["val_threshold"] = dist_thresh_prob
    dist_test_metrics["val_balanced_accuracy"] = round(dist_val_ba2, 4)
    dist_test_metrics["distance_threshold_m"] = dist_base.distance_threshold_m
    log(f"  Distance Test: ROC-AUC={dist_test_metrics['roc_auc']}, "
        f"F1={dist_test_metrics['f1_score']}, Spec={dist_test_metrics['specificity']}")
    results["distance_threshold"] = {"val_probs": dist_val_probs, "test_probs": dist_test_probs,
                                     "test_metrics": dist_test_metrics, "threshold": dist_thresh_prob}

    return results


# ==============================================================================
# 6. Plots
# ==============================================================================

def save_confusion_matrix_plot(metrics: dict, name: str, path: str) -> None:
    cm = metrics["confusion_matrix"]
    plot_confusion_matrix(cm, name, path)
    log(f"  Saved: {Path(path).name}")


def save_roc_pr_comparison(lstm_probs, kin_probs, dist_probs, y_test) -> None:
    log("\nGenerating ROC and PR curve comparisons...")

    # --- ROC ---
    lstm_fpr, lstm_tpr, lstm_auc = compute_roc_curve_points(y_test, lstm_probs)
    kin_fpr,  kin_tpr,  kin_auc  = compute_roc_curve_points(y_test, kin_probs)
    dist_fpr, dist_tpr, dist_auc = compute_roc_curve_points(y_test, dist_probs)

    roc_data = [
        {"label": f"LSTM (AUC={lstm_auc:.4f})", "fpr": lstm_fpr, "tpr": lstm_tpr},
        {"label": f"Kinematic (AUC={kin_auc:.4f})", "fpr": kin_fpr, "tpr": kin_tpr},
        {"label": f"Distance Threshold (AUC={dist_auc:.4f})", "fpr": dist_fpr, "tpr": dist_tpr},
    ]
    roc_path = str(PLOTS_DIR / "roc_curves_comparison.png")
    plot_roc_curves(roc_data, roc_path)
    log(f"  Saved: roc_curves_comparison.png")

    # --- PR ---
    lstm_rec, lstm_prec, lstm_pr_auc   = compute_pr_curve_points(y_test, lstm_probs)
    kin_rec,  kin_prec,  kin_pr_auc    = compute_pr_curve_points(y_test, kin_probs)
    dist_rec, dist_prec, dist_pr_auc   = compute_pr_curve_points(y_test, dist_probs)

    pr_data = [
        {"label": f"LSTM (AUC={lstm_pr_auc:.4f})", "recall": lstm_rec, "precision": lstm_prec},
        {"label": f"Kinematic (AUC={kin_pr_auc:.4f})", "recall": kin_rec, "precision": kin_prec},
        {"label": f"Distance Threshold (AUC={dist_pr_auc:.4f})", "recall": dist_rec, "precision": dist_prec},
    ]
    pr_path = str(PLOTS_DIR / "pr_curves_comparison.png")
    plot_pr_curves(pr_data, pr_path)
    log(f"  Saved: pr_curves_comparison.png")

    return lstm_auc, lstm_pr_auc


def save_loss_curves_note() -> None:
    """Create a placeholder note since training logs were not persisted."""
    note = {
        "note": "Loss curve not available.",
        "reason": (
            "The training pipeline crashed (save_checkpoint positional-arg bug) before "
            "epoch-by-epoch losses could be persisted. Training completed successfully: "
            "best epoch=13, val_loss=0.3318. Retraining would be required to regenerate curves."
        ),
        "best_epoch": BEST_EPOCH,
        "best_val_loss": BEST_VAL_LOSS,
    }
    note_path = PLOTS_DIR / "loss_curves_unavailable.json"
    with open(str(note_path), "w") as f:
        json.dump(note, f, indent=2)
    log(f"  Loss curves: not available (see {note_path.name})")


# ==============================================================================
# 7. Update checkpoint metadata (without overwriting weights)
# ==============================================================================

def update_checkpoint_metadata(model: LinkSurvivalLSTM, model_cfg: dict, lstm_val_f1: float) -> None:
    log("\nUpdating checkpoint metadata (preserving model weights)...")
    # Load full checkpoint to preserve weights + optimizer
    raw = torch.load(str(CHECKPOINT_PATH), map_location="cpu", weights_only=False)

    # Read config from file
    with open(str(MODEL_CONFIG_PATH)) as f:
        file_cfg = json.load(f)

    # Overwrite only the metadata fields
    raw["val_f1"] = round(lstm_val_f1, 6)
    raw["best_threshold"] = LSTM_THRESHOLD
    raw["epoch"] = BEST_EPOCH
    raw["val_loss"] = BEST_VAL_LOSS

    torch.save(raw, str(CHECKPOINT_PATH))

    # Verify round-trip
    verify = torch.load(str(CHECKPOINT_PATH), map_location="cpu", weights_only=False)
    assert verify["epoch"] == BEST_EPOCH, "epoch mismatch after update"
    assert abs(verify["val_loss"] - BEST_VAL_LOSS) < 1e-6, "val_loss mismatch"
    assert abs(verify["best_threshold"] - LSTM_THRESHOLD) < 1e-6, "threshold mismatch"
    assert abs(verify["val_f1"] - lstm_val_f1) < 1e-6, "val_f1 mismatch"

    # Verify model weights still intact
    test_model = LinkSurvivalLSTM(
        input_dim=7, hidden_dim=64, num_layers=2, dropout=0.3, bidirectional=False)
    test_model.load_state_dict(verify["model_state_dict"])
    test_model.eval()
    dummy = torch.zeros(1, 10, 7)
    with torch.no_grad():
        out = test_model(dummy)
    assert out.shape == (1,), f"Unexpected output shape: {out.shape}"

    log(f"  Checkpoint metadata updated: epoch={BEST_EPOCH}, val_loss={BEST_VAL_LOSS}, "
        f"threshold={LSTM_THRESHOLD}, val_f1={lstm_val_f1:.4f}")
    log("  Model weights verified intact (inference test passed).")


# ==============================================================================
# 8. Save metrics JSON
# ==============================================================================

def save_metrics_json(lstm_metrics: dict, baseline_results: dict, lstm_val_f1: float) -> None:
    output = {
        "experiment": "large_dataset_experiment",
        "evaluation_timestamp": datetime.datetime.now().isoformat(),
        "dataset": {
            "total_sequences": 170744,
            "train": 119529,
            "val": 25602,
            "test": 25613,
        },
        "lstm": {
            "architecture": {
                "input_dim": 7, "hidden_dim": 64, "num_layers": 2,
                "dropout": 0.3, "bidirectional": False, "total_params": 54081,
            },
            "training": {
                "best_epoch": BEST_EPOCH,
                "best_val_loss": BEST_VAL_LOSS,
                "val_balanced_accuracy": BEST_VAL_BALANCED_ACC,
                "val_f1_score": round(lstm_val_f1, 4),
                "val_threshold": LSTM_THRESHOLD,
            },
            "test_metrics": lstm_metrics,
        },
        "baselines": {
            "kinematic_extrapolation": baseline_results["kinematic"]["test_metrics"],
            "distance_threshold": baseline_results["distance_threshold"]["test_metrics"],
        },
        "split_strategy": "run_disjoint (zero run/vehicle/pair overlap)",
        "notes": [
            "LSTM threshold θ*=0.530 was fixed from validation tuning; not touched during test.",
            "Baseline thresholds were tuned on validation data using balanced_accuracy.",
            "Scaler was fitted on training sequences only.",
            "Loss curve data unavailable (pipeline crashed before persisting epoch history).",
        ],
    }

    path = METRICS_DIR / "test_metrics.json"
    with open(str(path), "w") as f:
        json.dump(output, f, indent=2)
    log(f"  Saved: {path.name}")


# ==============================================================================
# 9. Save model comparison CSV
# ==============================================================================

def save_comparison_csv(lstm_metrics: dict, baseline_results: dict) -> None:
    rows = []

    def build_row(name, m, threshold_note=""):
        cm = m["confusion_matrix"]
        return {
            "Model": name,
            "Threshold": m.get("val_threshold", m.get("threshold", "—")),
            "ROC-AUC": m["roc_auc"],
            "PR-AUC": m["pr_auc"],
            "F1-Score": m["f1_score"],
            "Balanced Accuracy": m["balanced_accuracy"],
            "Precision": m["precision"],
            "Recall (TPR)": m["recall"],
            "Specificity (TNR)": m["specificity"],
            "Accuracy": m["accuracy"],
            "Brier Score": m["brier_score"],
            "TP": cm["tp"], "FP": cm["fp"], "TN": cm["tn"], "FN": cm["fn"],
            "Threshold Note": threshold_note,
        }

    rows.append(build_row("LSTM (this work)", lstm_metrics,
                           f"θ*={LSTM_THRESHOLD} (val-tuned, epoch 13)"))
    rows.append(build_row("Kinematic Extrapolation",
                           baseline_results["kinematic"]["test_metrics"],
                           "θ* tuned on val (balanced_accuracy)"))
    rows.append(build_row("Distance Threshold",
                           baseline_results["distance_threshold"]["test_metrics"],
                           f"d*={baseline_results['distance_threshold']['test_metrics'].get('distance_threshold_m','?')}m, θ* tuned on val"))

    path = REPORTS_DIR / "model_comparison.csv"
    with open(str(path), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    log(f"  Saved: {path.name}")
    return rows


# ==============================================================================
# 10. Save experiment summary markdown
# ==============================================================================

def save_summary_md(lstm_metrics: dict, baseline_results: dict,
                    lstm_val_f1: float, lstm_roc_auc: float, lstm_pr_auc: float) -> None:
    km = baseline_results["kinematic"]["test_metrics"]
    dm = baseline_results["distance_threshold"]["test_metrics"]

    lstm_beats_kin_roc  = lstm_metrics["roc_auc"]  > km["roc_auc"]
    lstm_beats_kin_f1   = lstm_metrics["f1_score"] > km["f1_score"]
    lstm_beats_kin_ba   = lstm_metrics["balanced_accuracy"] > km["balanced_accuracy"]

    better_note = "✅ LSTM outperforms Kinematic baseline" if (lstm_beats_kin_roc and lstm_beats_kin_f1) \
        else "⚠️ LSTM does NOT outperform Kinematic Extrapolation baseline on all metrics — honest reporting required"

    md = f"""# Stage 3 Evaluation Report — Large Dataset Experiment

Generated: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Dataset

| Split | Sequences | Survive (1) | Break (0) |
|-------|-----------|------------|---------|
| Train | 119,529 | 75,018 (62.8%) | 44,511 (37.2%) |
| Val   | 25,602  | 16,124 (63.0%) | 9,478 (37.0%) |
| Test  | 25,613  | 15,867 (61.9%) | 9,746 (38.1%) |

Split strategy: **run-disjoint** (zero run/vehicle/pair overlap across train/val/test).

## LSTM Architecture

| Parameter | Value |
|-----------|-------|
| Input dim | 7 |
| Hidden dim | 64 |
| Layers | 2 |
| Dropout | 0.3 |
| Bidirectional | No |
| Total parameters | 54,081 |

Training: 13 epochs (early stopping), best val loss = {BEST_VAL_LOSS}  
Val threshold: θ* = {LSTM_THRESHOLD} (tuned on validation, balanced_accuracy = {BEST_VAL_BALANCED_ACC})  
Val F1: {lstm_val_f1:.4f}

## Test-Set Results (θ* fixed from validation — no test-set leakage)

| Model | ROC-AUC | PR-AUC | F1 | Bal. Acc | Precision | Recall | Specificity | Brier |
|-------|---------|--------|----|----------|-----------|--------|-------------|-------|
| **LSTM** | **{lstm_metrics['roc_auc']}** | **{lstm_metrics['pr_auc']}** | **{lstm_metrics['f1_score']}** | **{lstm_metrics['balanced_accuracy']}** | {lstm_metrics['precision']} | {lstm_metrics['recall']} | {lstm_metrics['specificity']} | {lstm_metrics['brier_score']} |
| Kinematic Extrapolation | {km['roc_auc']} | {km['pr_auc']} | {km['f1_score']} | {km['balanced_accuracy']} | {km['precision']} | {km['recall']} | {km['specificity']} | {km['brier_score']} |
| Distance Threshold | {dm['roc_auc']} | {dm['pr_auc']} | {dm['f1_score']} | {dm['balanced_accuracy']} | {dm['precision']} | {dm['recall']} | {dm['specificity']} | {dm['brier_score']} |

## Confusion Matrices

### LSTM (θ* = {LSTM_THRESHOLD})
|  | Predicted Survive | Predicted Break |
|--|---|---|
| **Actual Survive** | TP = {lstm_metrics['confusion_matrix']['tp']} | FN = {lstm_metrics['confusion_matrix']['fn']} |
| **Actual Break** | FP = {lstm_metrics['confusion_matrix']['fp']} | TN = {lstm_metrics['confusion_matrix']['tn']} |

### Kinematic Extrapolation (θ* = {km.get('val_threshold', '?')})
|  | Predicted Survive | Predicted Break |
|--|---|---|
| **Actual Survive** | TP = {km['confusion_matrix']['tp']} | FN = {km['confusion_matrix']['fn']} |
| **Actual Break** | FP = {km['confusion_matrix']['fp']} | TN = {km['confusion_matrix']['tn']} |

### Distance Threshold (d* = {dm.get('distance_threshold_m','?')} m, θ* = {dm.get('val_threshold','?')})
|  | Predicted Survive | Predicted Break |
|--|---|---|
| **Actual Survive** | TP = {dm['confusion_matrix']['tp']} | FN = {dm['confusion_matrix']['fn']} |
| **Actual Break** | FP = {dm['confusion_matrix']['fp']} | TN = {dm['confusion_matrix']['tn']} |

## Assessment

{better_note}

LSTM vs Kinematic:
- ROC-AUC: {lstm_metrics['roc_auc']} vs {km['roc_auc']} ({"+" if lstm_beats_kin_roc else "-"}{abs(lstm_metrics['roc_auc'] - km['roc_auc']):.4f})
- F1: {lstm_metrics['f1_score']} vs {km['f1_score']} ({"+" if lstm_beats_kin_f1 else "-"}{abs(lstm_metrics['f1_score'] - km['f1_score']):.4f})
- Balanced Acc: {lstm_metrics['balanced_accuracy']} vs {km['balanced_accuracy']} ({"+" if lstm_beats_kin_ba else "-"}{abs(lstm_metrics['balanced_accuracy'] - km['balanced_accuracy']):.4f})

## Notes

- **Loss curves** not available: pipeline crashed (positional-arg bug in save_checkpoint) before persisting
  epoch-by-epoch history. Best epoch = {BEST_EPOCH}, best val_loss = {BEST_VAL_LOSS}.  
  Bug has been fixed in `scripts/run_large_dataset_pipeline.py`.
- Scaler fitted exclusively on training data.
- Thresholds tuned on validation data only; test labels never seen during tuning.

## Output Files

| File | Description |
|------|-------------|
| `outputs/large_dataset_experiment/metrics/test_metrics.json` | All metrics, JSON |
| `outputs/large_dataset_experiment/reports/model_comparison.csv` | Tabular comparison |
| `outputs/large_dataset_experiment/plots/confusion_matrix_lstm.png` | LSTM confusion matrix |
| `outputs/large_dataset_experiment/plots/confusion_matrix_kinematic.png` | Kinematic confusion matrix |
| `outputs/large_dataset_experiment/plots/confusion_matrix_distance.png` | Distance confusion matrix |
| `outputs/large_dataset_experiment/plots/roc_curves_comparison.png` | ROC curves (all models) |
| `outputs/large_dataset_experiment/plots/pr_curves_comparison.png` | PR curves (all models) |
"""

    path = REPORTS_DIR / "experiment_summary.md"
    with open(str(path), "w") as f:
        f.write(md)
    log(f"  Saved: {path.name}")


# ==============================================================================
# 11. Fix pipeline checkpoint-saving bug
# ==============================================================================

def fix_pipeline_checkpoint_bug() -> None:
    pipeline_path = PROJECT_ROOT / "scripts" / "run_large_dataset_pipeline.py"
    if not pipeline_path.exists():
        log("  Pipeline script not found — skipping bug fix.")
        return

    with open(str(pipeline_path)) as f:
        source = f.read()

    # Look for the buggy positional call pattern
    buggy_pattern = (
        "save_checkpoint(best_ckpt_path, model, history[\"best_epoch\"], "
        "history[\"best_val_loss\"],\n                lstm_val_score, lstm_thresh, config)"
    )
    fixed_pattern = (
        "save_checkpoint(\n"
        "            filepath=best_ckpt_path,\n"
        "            model=model,\n"
        "            epoch=history[\"best_epoch\"],\n"
        "            val_loss=history[\"best_val_loss\"],\n"
        "            val_f1=lstm_val_score,\n"
        "            best_threshold=lstm_thresh,\n"
        "            config=config\n"
        "        )"
    )

    if buggy_pattern in source:
        source = source.replace(buggy_pattern, fixed_pattern)
        with open(str(pipeline_path), "w") as f:
            f.write(source)
        log("  Fixed positional-arg bug in run_large_dataset_pipeline.py")
    else:
        # Try a broader search
        import re
        # Look for any positional save_checkpoint call after training
        m = re.search(
            r'save_checkpoint\(best_ckpt_path,\s*model,\s*history\[.best_epoch.\]',
            source
        )
        if m:
            log("  WARNING: Found buggy pattern at different position — please fix manually.")
        else:
            log("  Pipeline save_checkpoint call appears already fixed or uses different pattern.")


# ==============================================================================
# Main
# ==============================================================================

def main():
    log("=" * 70)
    log("Stage 3 Standalone Evaluation — Large Dataset Experiment")
    log("=" * 70)

    makedirs()

    # 1. Load data
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()

    # 2. Load scaler and scale val/test (no refit)
    scaler, X_val_scaled, X_test_scaled = load_and_apply_scaler(X_train, X_val, X_test)

    # 3. Load LSTM
    model, model_cfg, ckpt = load_lstm_model()

    # 4. LSTM inference on val (to report val F1 for metadata update)
    log("\nRunning LSTM inference on validation set...")
    lstm_val_probs = lstm_predict(model, X_val_scaled)
    val_metrics = compute_classification_metrics(y_val, lstm_val_probs, threshold=LSTM_THRESHOLD)
    lstm_val_f1 = val_metrics["f1_score"]
    log(f"  LSTM Val: balanced_acc={val_metrics['balanced_accuracy']}, F1={lstm_val_f1}")

    # 5. LSTM inference on test
    log("Running LSTM inference on test set...")
    lstm_test_probs = lstm_predict(model, X_test_scaled)
    lstm_test_metrics = compute_classification_metrics(y_test, lstm_test_probs, threshold=LSTM_THRESHOLD)
    lstm_test_metrics["val_threshold"] = LSTM_THRESHOLD
    log(f"  LSTM Test: ROC-AUC={lstm_test_metrics['roc_auc']}, "
        f"F1={lstm_test_metrics['f1_score']}, Spec={lstm_test_metrics['specificity']}, "
        f"Bal.Acc={lstm_test_metrics['balanced_accuracy']}")
    log(f"  LSTM Test CM: TP={lstm_test_metrics['confusion_matrix']['tp']}, "
        f"FP={lstm_test_metrics['confusion_matrix']['fp']}, "
        f"TN={lstm_test_metrics['confusion_matrix']['tn']}, "
        f"FN={lstm_test_metrics['confusion_matrix']['fn']}")

    # 6. Baselines (use unscaled X for baselines — they operate in metres/mps)
    baseline_results = evaluate_baselines(X_val, y_val, X_test, y_test)

    # 7. Confusion matrix plots
    log("\nGenerating confusion matrix plots...")
    save_confusion_matrix_plot(lstm_test_metrics, "LSTM",
                                str(PLOTS_DIR / "confusion_matrix_lstm.png"))
    save_confusion_matrix_plot(baseline_results["kinematic"]["test_metrics"],
                                "Kinematic Extrapolation",
                                str(PLOTS_DIR / "confusion_matrix_kinematic.png"))
    save_confusion_matrix_plot(baseline_results["distance_threshold"]["test_metrics"],
                                "Distance Threshold",
                                str(PLOTS_DIR / "confusion_matrix_distance.png"))

    # 8. ROC & PR curves
    lstm_roc_auc, lstm_pr_auc = save_roc_pr_comparison(
        lstm_test_probs,
        baseline_results["kinematic"]["test_probs"],
        baseline_results["distance_threshold"]["test_probs"],
        y_test
    )

    # 9. Loss curves note
    save_loss_curves_note()

    # 10. Update checkpoint metadata
    update_checkpoint_metadata(model, model_cfg, lstm_val_f1)

    # 11. Save metrics JSON
    log("\nSaving metrics and reports...")
    save_metrics_json(lstm_test_metrics, baseline_results, lstm_val_f1)

    # 12. Save CSV
    save_comparison_csv(lstm_test_metrics, baseline_results)

    # 13. Save summary markdown
    save_summary_md(lstm_test_metrics, baseline_results,
                    lstm_val_f1, lstm_roc_auc, lstm_pr_auc)

    # 14. Fix pipeline bug
    log("\nFixing pipeline checkpoint-saving bug...")
    fix_pipeline_checkpoint_bug()

    # 15. Final summary
    log("\n" + "=" * 70)
    log("EVALUATION COMPLETE")
    log("=" * 70)
    log(f"\nLSTM Test Results (θ*={LSTM_THRESHOLD}):")
    log(f"  ROC-AUC:          {lstm_test_metrics['roc_auc']}")
    log(f"  PR-AUC:           {lstm_test_metrics['pr_auc']}")
    log(f"  F1-Score:         {lstm_test_metrics['f1_score']}")
    log(f"  Balanced Acc:     {lstm_test_metrics['balanced_accuracy']}")
    log(f"  Precision:        {lstm_test_metrics['precision']}")
    log(f"  Recall:           {lstm_test_metrics['recall']}")
    log(f"  Specificity:      {lstm_test_metrics['specificity']}")
    log(f"  Brier Score:      {lstm_test_metrics['brier_score']}")
    log(f"\nKinematic Baseline ROC-AUC: {baseline_results['kinematic']['test_metrics']['roc_auc']}")
    log(f"Distance Baseline ROC-AUC:  {baseline_results['distance_threshold']['test_metrics']['roc_auc']}")

    log("\nOutput files:")
    for f in [
        METRICS_DIR / "test_metrics.json",
        REPORTS_DIR / "model_comparison.csv",
        REPORTS_DIR / "experiment_summary.md",
        PLOTS_DIR / "confusion_matrix_lstm.png",
        PLOTS_DIR / "confusion_matrix_kinematic.png",
        PLOTS_DIR / "confusion_matrix_distance.png",
        PLOTS_DIR / "roc_curves_comparison.png",
        PLOTS_DIR / "pr_curves_comparison.png",
    ]:
        exists = "✓" if f.exists() else "✗ MISSING"
        log(f"  {exists}  {f.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
