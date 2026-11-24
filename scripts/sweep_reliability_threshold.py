#!/usr/bin/env python3
"""
Phase 1 — Reliability-threshold sweep for the existing LSTM (no retraining).

Protocol (approved):
  * Sweep theta in {0.50, 0.51, ..., 0.90} on VALIDATION data only.
  * Selection rule: the smallest theta with
        predicted-survive precision >= 0.90  AND  recall >= 0.50
    (precision >= 0.90 means at most ~10% of links the filter approves break
    within 30 s). If no theta satisfies both, fall back to the theta with the
    highest validation balanced accuracy.
  * Only after theta* is fixed and saved is the held-out TEST set evaluated,
    once, at theta*. Test results are never used to change theta*.

Outputs (new files only; the script refuses to overwrite):
  outputs/large_dataset_experiment/reliability_threshold/
      threshold_sweep_val.csv
      selected_threshold.json
      test_metrics_at_selected_threshold.json
      threshold_sweep_val.png
  outputs/large_dataset_experiment/checkpoints/checkpoint_metadata.json
"""
import csv
import json
import os
import sys
from datetime import datetime

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from src.models.predictor import LinkSurvivalPredictor  # noqa: E402
from src.evaluation.metrics import compute_classification_metrics  # noqa: E402
from src.utils.integrity import sha256_file, verify_manifest, write_json_if_absent  # noqa: E402

EXP_DIR = "outputs/large_dataset_experiment"
CKPT_DIR = f"{EXP_DIR}/checkpoints"
CHECKPOINT = f"{CKPT_DIR}/best_model.pt"
MODEL_CONFIG = f"{CKPT_DIR}/model_config.json"
SCALER = f"{CKPT_DIR}/scaler.json"
METADATA_OUT = f"{CKPT_DIR}/checkpoint_metadata.json"
OUT_DIR = f"{EXP_DIR}/reliability_threshold"
EXPERIMENT_CONFIG = "config/stage3_large_dataset.json"
SEQUENCES = "data/processed/large_dataset/pairwise_sequences.npz"
SPLIT_DIR = "data/splits/large_dataset"
MANIFEST = "outputs/PROTECTED_ARTIFACTS.sha256"

THRESHOLDS = np.round(np.arange(50, 91) / 100.0, 2)   # 0.50 ... 0.90 (41 values)
MIN_PRECISION = 0.90
MIN_RECALL = 0.50
ORIGINAL_THRESHOLD = 0.53     # balanced-accuracy-tuned value from the training run


def raw_rates(y_true: np.ndarray, probs: np.ndarray, theta: float):
    """Unrounded precision/recall/specificity, used for the selection rule."""
    pred = probs >= theta
    tp = int(np.sum(pred & (y_true == 1)))
    fp = int(np.sum(pred & (y_true == 0)))
    fn = int(np.sum(~pred & (y_true == 1)))
    tn = int(np.sum(~pred & (y_true == 0)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    return precision, recall, specificity, 0.5 * (recall + specificity)


def select_threshold(rows):
    """Applies the approved rule to the validation sweep rows."""
    for r in rows:   # rows are in ascending theta order -> first match is the smallest
        if r["precision_raw"] >= MIN_PRECISION and r["recall_raw"] >= MIN_RECALL:
            return r["threshold"], "precision_constrained"
    best = max(rows, key=lambda r: (r["balanced_accuracy_raw"], -r["threshold"]))
    return best["threshold"], "fallback_max_balanced_accuracy"


def batch_probs(predictor, X):
    return np.concatenate([predictor.predict_batch_probabilities(X[s:s + 4096])
                           for s in range(0, len(X), 4096)])


def save_plot(rows, theta_star, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted, grid, surface = "#1f1f1e", "#6b6a64", "#e4e3de", "#fcfcfb"
    blue, orange, aqua = "#2a78d6", "#eb6834", "#1baf7a"
    th = [r["threshold"] for r in rows]
    prec = [r["precision_raw"] for r in rows]
    rec = [r["recall_raw"] for r in rows]
    bal = [r["balanced_accuracy_raw"] for r in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6), facecolor=surface)
    for ax in (ax1, ax2):
        ax.set_facecolor(surface)
        ax.grid(True, color=grid, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(muted)
        ax.tick_params(colors=muted, labelsize=9)

    # Left: metrics vs threshold (direct labels at line ends + legend)
    for vals, color, name in ((prec, blue, "Precision (survive)"),
                              (rec, orange, "Recall (survive)"),
                              (bal, aqua, "Balanced accuracy")):
        ax1.plot(th, vals, color=color, linewidth=2, label=name)
        ax1.annotate(name, (th[-1], vals[-1]), xytext=(6, 0), textcoords="offset points",
                     va="center", fontsize=8.5, color=ink)
    ax1.axhline(MIN_PRECISION, color=muted, linewidth=1, linestyle="--")
    ax1.axhline(MIN_RECALL, color=muted, linewidth=1, linestyle=":")
    ax1.text(0.502, MIN_PRECISION + 0.008, "precision floor 0.90", fontsize=8, color=muted)
    ax1.text(0.502, MIN_RECALL + 0.008, "recall floor 0.50", fontsize=8, color=muted)
    ax1.axvline(theta_star, color=ink, linewidth=1)
    ax1.text(theta_star + 0.004, 0.975, f"θ* = {theta_star:.2f}", fontsize=9, color=ink, va="top")
    ax1.set_xlim(0.50, 0.99)
    ax1.set_ylim(0.2, 1.0)
    ax1.set_xlabel("Reliability threshold θ", color=ink)
    ax1.set_ylabel("Validation metric", color=ink)
    ax1.set_title("Validation metrics vs. threshold", loc="left", color=ink, fontsize=11)
    ax1.legend(frameon=False, fontsize=8.5, loc="lower left", bbox_to_anchor=(0.0, 0.02))

    # Right: precision–recall trade-off across the sweep
    ax2.plot(rec, prec, color=blue, linewidth=2, marker="o", markersize=3.5)
    i = th.index(theta_star)
    ax2.plot([rec[i]], [prec[i]], marker="o", markersize=9, color=blue,
             markeredgecolor=surface, markeredgewidth=2)
    ax2.annotate(f"θ* = {theta_star:.2f}\nP = {prec[i]:.3f}, R = {rec[i]:.3f}",
                 (rec[i], prec[i]), xytext=(-125, -38), textcoords="offset points",
                 fontsize=8.5, color=ink)
    for t_lab in (0.50, 0.70, 0.90):
        k = th.index(t_lab)
        if t_lab != theta_star:
            ax2.annotate(f"θ={t_lab:.2f}", (rec[k], prec[k]), xytext=(6, 4),
                         textcoords="offset points", fontsize=8, color=muted)
    ax2.axhline(MIN_PRECISION, color=muted, linewidth=1, linestyle="--")
    ax2.axvline(MIN_RECALL, color=muted, linewidth=1, linestyle=":")
    ax2.set_xlabel("Recall (survive class)", color=ink)
    ax2.set_ylabel("Precision (survive class)", color=ink)
    ax2.set_title("Precision–recall trade-off (θ = 0.50 → 0.90)", loc="left", color=ink, fontsize=11)

    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=surface)
    plt.close(fig)


def main() -> int:
    print("=" * 72)
    print(" Phase 1: Reliability threshold sweep (validation-only selection)")
    print("=" * 72)

    changed, missing = verify_manifest(MANIFEST)
    if changed or missing:
        print(f"ABORT: protected artifacts differ from snapshot: changed={changed} missing={missing}")
        return 1

    targets = [f"{OUT_DIR}/threshold_sweep_val.csv", f"{OUT_DIR}/selected_threshold.json",
               f"{OUT_DIR}/test_metrics_at_selected_threshold.json",
               f"{OUT_DIR}/threshold_sweep_val.png", METADATA_OUT]
    existing = [p for p in targets if os.path.exists(p)]
    if existing:
        print(f"ABORT: refusing to overwrite existing outputs: {existing}")
        return 1
    os.makedirs(OUT_DIR, exist_ok=True)

    with open(f"{SPLIT_DIR}/split_info.json") as f:
        split_info = json.load(f)
    with open(EXPERIMENT_CONFIG) as f:
        exp_cfg = json.load(f)

    data = np.load(SEQUENCES)
    X, y = data["X"].astype(np.float32), data["y"].astype(np.int64)
    val_idx = np.load(f"{SPLIT_DIR}/val_indices.npy")
    X_val, y_val = X[val_idx], y[val_idx]
    print(f"Validation runs: {split_info['val_runs']} ({len(y_val):,} sequences)")

    predictor = LinkSurvivalPredictor(CHECKPOINT, MODEL_CONFIG, SCALER, device="cpu")
    val_probs = batch_probs(predictor, X_val)

    # ---- 1. Validation sweep ----
    rows = []
    for theta in THRESHOLDS:
        theta = float(theta)
        m = compute_classification_metrics(y_val, val_probs, threshold=theta)
        p, r, s, b = raw_rates(y_val, val_probs, theta)
        cm = m["confusion_matrix"]
        rows.append({
            "threshold": theta,
            "precision": m["precision"], "recall": m["recall"], "specificity": m["specificity"],
            "f1_score": m["f1_score"], "balanced_accuracy": m["balanced_accuracy"],
            "accuracy": m["accuracy"],
            "approved_link_break_rate": round(1.0 - p, 4) if cm["tp"] + cm["fp"] else None,
            "predicted_positive_count": m["predicted_positive_count"],
            "tp": cm["tp"], "fp": cm["fp"], "tn": cm["tn"], "fn": cm["fn"],
            "meets_precision": p >= MIN_PRECISION, "meets_recall": r >= MIN_RECALL,
            "precision_raw": p, "recall_raw": r, "specificity_raw": s, "balanced_accuracy_raw": b,
        })

    theta_star, rule_used = select_threshold(rows)
    sel = next(r for r in rows if r["threshold"] == theta_star)
    val_roc_auc = compute_classification_metrics(y_val, val_probs, 0.5)["roc_auc"]

    with open(f"{OUT_DIR}/threshold_sweep_val.csv", "x", newline="") as f:
        cols = [k for k in rows[0] if not k.endswith("_raw")]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in cols})

    selection = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "selected_threshold": theta_star,
        "rule_used": rule_used,
        "rule": {
            "primary": f"smallest theta in [0.50, 0.90] with val precision >= {MIN_PRECISION} "
                       f"and val recall >= {MIN_RECALL}",
            "fallback": "theta with maximum validation balanced accuracy",
            "grid": [float(t) for t in THRESHOLDS],
        },
        "selected_on": "validation", "validation_runs": split_info["val_runs"],
        "validation_samples": int(len(y_val)),
        "validation_metrics_at_selected": {k: v for k, v in sel.items() if not k.endswith("_raw")},
        "validation_roc_auc": val_roc_auc,
        "thresholds_meeting_both_constraints": [r["threshold"] for r in rows
                                                if r["meets_precision"] and r["meets_recall"]],
    }
    with open(f"{OUT_DIR}/selected_threshold.json", "x") as f:
        json.dump(selection, f, indent=2)
    print(f"\nSelected θ* = {theta_star:.2f}  (rule: {rule_used})")
    print(f"  val precision={sel['precision']}, recall={sel['recall']}, "
          f"specificity={sel['specificity']}, balanced acc={sel['balanced_accuracy']}")

    save_plot(rows, theta_star, f"{OUT_DIR}/threshold_sweep_val.png")

    # ---- 2. Held-out test, evaluated ONCE at the frozen theta* ----
    test_idx = np.load(f"{SPLIT_DIR}/test_indices.npy")
    X_test, y_test = X[test_idx], y[test_idx]
    test_probs = batch_probs(predictor, X_test)
    test_m = compute_classification_metrics(y_test, test_probs, threshold=theta_star)
    test_out = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "threshold": theta_star, "threshold_selected_on": "validation (frozen before test evaluation)",
        "test_runs": split_info["test_runs"], "test_samples": int(len(y_test)),
        "metrics": test_m,
        "approved_link_break_rate": round(1.0 - test_m["precision"], 4),
        "note": "Single evaluation at the frozen threshold; not used for any tuning.",
    }
    with open(f"{OUT_DIR}/test_metrics_at_selected_threshold.json", "x") as f:
        json.dump(test_out, f, indent=2)
    print(f"\nTest @ θ*={theta_star:.2f}: precision={test_m['precision']}, recall={test_m['recall']}, "
          f"specificity={test_m['specificity']}, F1={test_m['f1_score']}, "
          f"balanced acc={test_m['balanced_accuracy']}, ROC-AUC={test_m['roc_auc']}")

    # ---- 3. New checkpoint metadata file (best_model.pt is never written) ----
    metadata = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "description": "Corrected metadata for the large-dataset LinkSurvivalLSTM. "
                       "Supersedes the 'best_threshold'/'val_f1' fields stored inside best_model.pt "
                       "(0.5 / 0.0), which were never updated because the pipeline crashed.",
        "model": {"checkpoint": CHECKPOINT, "model_config": MODEL_CONFIG, "scaler": SCALER,
                  "architecture": json.load(open(MODEL_CONFIG))},
        "training": {"best_epoch": 13, "best_val_loss": 0.3318, "retrained": False},
        "features": {"names_in_order": exp_cfg["features"]["feature_names"],
                     "history_window_s": exp_cfg["link"]["history_window_s"],
                     "prediction_horizon_s": exp_cfg["link"]["prediction_horizon_s"],
                     "comm_range_m": exp_cfg["link"]["comm_range_m"],
                     "step_length_s": exp_cfg["data"]["step_length_s"],
                     "link_model": "abstracted vehicular sidelink/access link (range-based)"},
        "classification_threshold": {
            "value": ORIGINAL_THRESHOLD,
            "selected_by": "max validation balanced accuracy in [0.15, 0.85] (original training run)"},
        "reliability_threshold": {
            "value": theta_star, "rule_used": rule_used,
            "selected_by": selection["rule"]["primary"],
            "details": f"{OUT_DIR}/selected_threshold.json"},
        "split": {"strategy": split_info["strategy"], "train_runs": split_info["train_runs"],
                  "val_runs": split_info["val_runs"], "test_runs": split_info["test_runs"]},
        "sha256": {CHECKPOINT: sha256_file(CHECKPOINT), SCALER: sha256_file(SCALER),
                   MODEL_CONFIG: sha256_file(MODEL_CONFIG)},
    }
    write_json_if_absent(METADATA_OUT, metadata)
    print(f"\nWrote {METADATA_OUT}")

    changed, missing = verify_manifest(MANIFEST)
    ok = not changed and not missing
    print(f"Protected artifacts unchanged after run: {ok}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
