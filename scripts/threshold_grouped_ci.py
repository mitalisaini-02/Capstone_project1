#!/usr/bin/env python3
"""
Grouped (cluster) bootstrap confidence intervals for the reliability threshold —
VALIDATION DATA ONLY. Defines the θ = 0.65 SENSITIVITY variant.

The primary baseline threshold remains θ = 0.62 (checkpoint_metadata.json); this
script never modifies it and never reads test data.

Method
  * LSTM validation probabilities (existing model, no retraining).
  * Cluster bootstrap (B = 2000, seed 42) resampling whole groups, because samples of
    the same pair/consumer are strongly correlated. Groups: pair (primary), consumer.
  * Per-run precision is also reported (only 4 validation runs: too few for a run-level CI).
  * Sensitivity rule: smallest θ in [0.50, 0.90] whose one-sided 95 % lower bound
    (pair clusters) has precision >= 0.90 and recall >= 0.50.

Outputs (new directory, write-once):
  outputs/large_dataset_experiment/reliability_threshold_robust/
      grouped_ci_val.csv
      sensitivity_threshold.json
"""
import json
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from src.models.predictor import LinkSurvivalPredictor  # noqa: E402
from src.utils.integrity import verify_manifest  # noqa: E402

CKPT = "outputs/large_dataset_experiment/checkpoints/"
OUT = "outputs/large_dataset_experiment/reliability_threshold_robust"
THRESHOLDS = np.round(np.arange(50, 91) / 100.0, 2)
B, SEED, ALPHA = 2000, 42, 0.05
MIN_PRECISION, MIN_RECALL = 0.90, 0.50
PRIMARY_THETA = 0.62


def group_counts(groups, probs, y):
    u, inv = np.unique(groups, return_inverse=True)
    G = len(u)
    TP = np.zeros((G, len(THRESHOLDS)))
    FP, FN = np.zeros_like(TP), np.zeros_like(TP)
    for k, t in enumerate(THRESHOLDS):
        pr = probs >= t
        TP[:, k] = np.bincount(inv, weights=(pr & (y == 1)), minlength=G)
        FP[:, k] = np.bincount(inv, weights=(pr & (y == 0)), minlength=G)
        FN[:, k] = np.bincount(inv, weights=(~pr & (y == 1)), minlength=G)
    return u, TP, FP, FN


def bootstrap_bounds(TP, FP, FN, rng):
    G = TP.shape[0]
    W = rng.multinomial(G, np.full(G, 1.0 / G), size=B)
    tp, fp, fn = W @ TP, W @ FP, W @ FN
    prec = tp / np.maximum(tp + fp, 1)
    rec = tp / np.maximum(tp + fn, 1)
    return (np.percentile(prec, 100 * ALPHA, axis=0), np.percentile(prec, 2.5, axis=0),
            np.percentile(rec, 100 * ALPHA, axis=0))


def main() -> int:
    changed, missing = verify_manifest("outputs/PROTECTED_ARTIFACTS.sha256")
    if changed or missing:
        print(f"ABORT: protected artifacts changed={changed} missing={missing}")
        return 1
    if os.path.exists(OUT):
        print(f"ABORT: {OUT} already exists (overwrite protection)")
        return 1

    data = np.load("data/processed/large_dataset/pairwise_sequences.npz")
    meta = json.load(open("data/processed/large_dataset/metadata.json"))
    vi = np.load("data/splits/large_dataset/val_indices.npy")
    X, y = data["X"][vi].astype(np.float32), data["y"][vi].astype(int)
    pred = LinkSurvivalPredictor(CKPT + "best_model.pt", CKPT + "model_config.json", CKPT + "scaler.json",
                                 device="cpu")
    probs = np.concatenate([pred.predict_batch_probabilities(X[s:s + 4096]) for s in range(0, len(X), 4096)])
    runs = np.array([meta[i]["run_id"] for i in vi])
    pairs = np.array([meta[i]["pair_id"] for i in vi])
    cons = np.array([f"{meta[i]['run_id']}|{meta[i]['consumer_id']}" for i in vi])

    rng = np.random.default_rng(SEED)
    _, TPp, FPp, FNp = group_counts(pairs, probs, y)
    p_lo, p_lo2, r_lo = bootstrap_bounds(TPp, FPp, FNp, rng)
    _, TPc, FPc, FNc = group_counts(cons, probs, y)
    pc_lo, _, rc_lo = bootstrap_bounds(TPc, FPc, FNc, rng)
    ur, TPr, FPr, _ = group_counts(runs, probs, y)
    per_run = TPr / np.maximum(TPr + FPr, 1)

    tp, fp, fn = TPp.sum(0), FPp.sum(0), FNp.sum(0)
    rows = []
    for k, t in enumerate(THRESHOLDS):
        row = {"threshold": float(t), "precision": tp[k] / (tp[k] + fp[k]), "recall": tp[k] / (tp[k] + fn[k]),
               "precision_lb95_onesided_pair": p_lo[k], "precision_lb975_pair": p_lo2[k],
               "recall_lb95_onesided_pair": r_lo[k],
               "precision_lb95_onesided_consumer": pc_lo[k], "recall_lb95_onesided_consumer": rc_lo[k],
               "precision_min_over_val_runs": per_run[:, k].min()}
        row.update({f"precision_{r}": per_run[j, k] for j, r in enumerate(ur)})
        rows.append(row)
    df = pd.DataFrame(rows).round(4)
    ok = df[(df.precision_lb95_onesided_pair >= MIN_PRECISION) & (df.recall_lb95_onesided_pair >= MIN_RECALL)]
    theta_sens = float(ok.threshold.min()) if len(ok) else None

    os.makedirs(OUT)
    with open(f"{OUT}/grouped_ci_val.csv", "x", newline="") as f:
        df.to_csv(f, index=False)
    r62 = df[df.threshold == PRIMARY_THETA].iloc[0]
    out = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "role": "SENSITIVITY VARIANT — the primary baseline threshold remains 0.62",
        "primary_threshold": PRIMARY_THETA,
        "sensitivity_threshold": theta_sens,
        "rule": f"smallest theta in [0.50, 0.90] with pair-clustered one-sided 95% lower bound "
                f"precision >= {MIN_PRECISION} and recall >= {MIN_RECALL}",
        "data": "validation runs only " + str(sorted(set(runs.tolist()))) + "; test data not read",
        "bootstrap": {"B": B, "seed": SEED, "groups": {"pair": int(len(np.unique(pairs))),
                                                       "consumer": int(len(np.unique(cons))),
                                                       "run": int(len(ur))}},
        "thresholds_meeting_rule": ok.threshold.tolist(),
        "primary_at_0.62": {"precision": float(r62.precision),
                            "precision_lb95_onesided_pair": float(r62.precision_lb95_onesided_pair),
                            "min_run_precision": float(r62.precision_min_over_val_runs)},
        "caveat": "Pairs within a run share traffic conditions, so pair-level intervals are somewhat "
                  "optimistic; 4 validation runs are too few for a run-level interval.",
    }
    with open(f"{OUT}/sensitivity_threshold.json", "x") as f:
        json.dump(out, f, indent=2)
    print(json.dumps({k: out[k] for k in ("primary_threshold", "sensitivity_threshold",
                                          "thresholds_meeting_rule", "primary_at_0.62")}, indent=2))
    changed, missing = verify_manifest("outputs/PROTECTED_ARTIFACTS.sha256")
    return 0 if theta_sens is not None and not changed and not missing else 1


if __name__ == "__main__":
    sys.exit(main())
