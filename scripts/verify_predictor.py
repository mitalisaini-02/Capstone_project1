#!/usr/bin/env python3
"""
Phase 1 — Verify LinkSurvivalPredictor inference against the recorded results.

Checks (read-only with respect to all existing artifacts):
  1. Protected checkpoint/scaler/config hashes are unchanged (before and after).
  2. The predictor loads the large-dataset checkpoint, scaler and config.
  3. Scaler statistics and feature order match the recorded scaler.json/config.
  4. Single-sequence and batch inference agree.
  5. Test-set metrics at the ORIGINAL recorded threshold (0.53) reproduce
     outputs/large_dataset_experiment/metrics/test_metrics.json.
     This only re-checks an existing result; nothing is tuned on test data.

Writes one new report (refuses to overwrite):
  outputs/large_dataset_experiment/verification/predictor_verification.json

Exit code 0 = all checks passed, 1 = at least one check failed.
"""
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
from src.utils.integrity import verify_manifest, write_json_if_absent  # noqa: E402

EXP_DIR = "outputs/large_dataset_experiment"
CKPT_DIR = f"{EXP_DIR}/checkpoints"
CHECKPOINT = f"{CKPT_DIR}/best_model.pt"
MODEL_CONFIG = f"{CKPT_DIR}/model_config.json"
SCALER = f"{CKPT_DIR}/scaler.json"
RECORDED_METRICS = f"{EXP_DIR}/metrics/test_metrics.json"
EXPERIMENT_CONFIG = "config/stage3_large_dataset.json"
SEQUENCES = "data/processed/large_dataset/pairwise_sequences.npz"
TEST_IDX = "data/splits/large_dataset/test_indices.npy"
MANIFEST = "outputs/PROTECTED_ARTIFACTS.sha256"
REPORT = f"{EXP_DIR}/verification/predictor_verification.json"

ORIGINAL_THRESHOLD = 0.53   # recorded validation-tuned threshold (balanced accuracy)
TOLERANCE = 1e-3            # recorded metrics are rounded to 4 decimals
COMPARED_METRICS = ["accuracy", "precision", "recall", "specificity", "f1_score",
                    "balanced_accuracy", "roc_auc", "pr_auc", "brier_score"]


def main() -> int:
    checks = []

    def record(name, passed, detail):
        checks.append({"check": name, "passed": bool(passed), "detail": detail})
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}: {detail}")

    print("=" * 72)
    print(" Phase 1: LinkSurvivalPredictor verification (large dataset model)")
    print("=" * 72)

    # 1. Hashes before
    changed, missing = verify_manifest(MANIFEST)
    record("protected_hashes_before", not changed and not missing,
           f"changed={changed}, missing={missing}")

    # 2. Load predictor exactly as the MEC pipeline will (no metadata file)
    predictor = LinkSurvivalPredictor(CHECKPOINT, MODEL_CONFIG, SCALER, device="cpu")
    record("predictor_loads", True,
           f"checkpoint threshold={predictor.tuned_threshold} (source={predictor.threshold_source})")

    # 3. Scaler + feature order
    with open(SCALER) as f:
        scaler_json = json.load(f)
    with open(EXPERIMENT_CONFIG) as f:
        exp_cfg = json.load(f)
    expected_features = exp_cfg["features"]["feature_names"]
    same_stats = (np.allclose(predictor.scaler.mean_, scaler_json["mean"], atol=1e-6)
                  and np.allclose(predictor.scaler.scale_, scaler_json["scale"], atol=1e-6))
    record("scaler_matches_file", same_stats, "mean/scale identical to scaler.json")
    record("feature_order", predictor.scaler.feature_names == expected_features,
           f"{predictor.scaler.feature_names}")

    # 4. Data: held-out test split (used only to re-check the recorded result)
    data = np.load(SEQUENCES)
    X, y = data["X"].astype(np.float32), data["y"].astype(np.int64)
    test_idx = np.load(TEST_IDX)
    X_test, y_test = X[test_idx], y[test_idx]
    record("input_shape", X_test.shape[1:] == (10, 7), f"X_test shape={X_test.shape}")

    probs = np.concatenate([predictor.predict_batch_probabilities(X_test[s:s + 4096])
                            for s in range(0, len(X_test), 4096)])
    in_range = bool(np.all((probs >= 0.0) & (probs <= 1.0)))
    record("probabilities_in_[0,1]", in_range, f"min={probs.min():.4f}, max={probs.max():.4f}")

    single = np.array([predictor.predict_probability(X_test[i]) for i in range(200)])
    max_diff = float(np.max(np.abs(single - probs[:200])))
    record("single_vs_batch", max_diff < 1e-5, f"max |diff| over 200 samples = {max_diff:.2e}")

    # 5. Reproduce recorded test metrics at the ORIGINAL threshold
    with open(RECORDED_METRICS) as f:
        recorded = json.load(f)["lstm"]["test_metrics"]
    reproduced = compute_classification_metrics(y_test, probs, threshold=ORIGINAL_THRESHOLD)

    comparison = {}
    for k in COMPARED_METRICS:
        diff = abs(reproduced[k] - recorded[k])
        comparison[k] = {"recorded": recorded[k], "reproduced": reproduced[k], "abs_diff": round(diff, 6)}
        record(f"metric_{k}", diff <= TOLERANCE, f"recorded={recorded[k]} reproduced={reproduced[k]}")
    cm_match = reproduced["confusion_matrix"] == recorded["confusion_matrix"]
    record("confusion_matrix", cm_match,
           f"recorded={recorded['confusion_matrix']} reproduced={reproduced['confusion_matrix']}")

    # 6. Hashes after
    changed, missing = verify_manifest(MANIFEST)
    record("protected_hashes_after", not changed and not missing,
           f"changed={changed}, missing={missing}")

    all_passed = all(c["passed"] for c in checks)
    report = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "all_passed": all_passed,
        "checkpoint": CHECKPOINT,
        "threshold_used_for_reproduction": ORIGINAL_THRESHOLD,
        "test_samples": int(len(y_test)),
        "note": "Test data used only to re-check the previously recorded result; no tuning.",
        "metric_comparison": comparison,
        "checks": checks,
    }
    try:
        write_json_if_absent(REPORT, report)
        print(f"\nReport written: {REPORT}")
    except FileExistsError as e:
        print(f"\nReport NOT written: {e}")

    print("\nRESULT:", "ALL CHECKS PASSED" if all_passed else "ONE OR MORE CHECKS FAILED")
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
