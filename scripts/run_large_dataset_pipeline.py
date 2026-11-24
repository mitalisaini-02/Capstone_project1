"""
run_large_dataset_pipeline.py
------------------------------
Stage 3 LSTM Training and Baseline Evaluation Pipeline for the Large Multi-Run SUMO Dataset.

Key features:
  - Loads combined dataset (pairwise_sequences.npz, metadata.json)
  - Loads run-disjoint train/val/test splits
  - Validates dataset integrity (no NaN/Inf, correct shape, run/vehicle/pair disjointness)
  - Computes BCE pos_weight strictly from training labels
  - Fits SequenceStandardScaler strictly on training sequences
  - Evaluates Kinematic Extrapolation and Distance Threshold baselines
  - Trains LinkSurvivalLSTM with early stopping on validation loss
  - Tunes classification threshold on validation set only
  - Evaluates all models on the identical held-out test set
  - Plots: Loss Curves, Confusion Matrices, Multi-Model ROC Curves, Multi-Model PR Curves
  - Generates comprehensive markdown and CSV reports in outputs/large_dataset_experiment/

Usage:
  python scripts/run_large_dataset_pipeline.py --config config/stage3_large_dataset.json
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils.logger import get_logger
from src.utils.seed import set_seed
from src.utils.config_loader import load_config
from src.features.scaler import SequenceStandardScaler
from src.evaluation.baselines import KinematicExtrapolationBaseline, DistanceThresholdBaseline
from src.models.lstm import LinkSurvivalLSTM
from src.training.dataset import create_data_loaders
from src.training.trainer import train_lstm_model, evaluate_loss_and_predictions
from src.training.checkpoints import load_checkpoint, save_checkpoint
from src.evaluation.metrics import (
    compute_classification_metrics,
    tune_classification_threshold,
    compute_roc_curve_points,
    compute_pr_curve_points
)
from src.evaluation.visualizer import (
    plot_loss_curves,
    plot_confusion_matrix,
    plot_roc_curves,
    plot_pr_curves
)
from src.models.predictor import LinkSurvivalPredictor


def load_large_dataset(config: dict):
    dataset_dir = config["data"]["processed_dir"]
    splits_dir  = config["data"]["splits_dir"]

    npz_path  = os.path.join(dataset_dir, "pairwise_sequences.npz")
    meta_path = os.path.join(dataset_dir, "metadata.json")
    split_info_path = os.path.join(splits_dir, "split_info.json")
    split_audit_path = os.path.join(splits_dir, "split_audit.json")
    train_idx_path  = os.path.join(splits_dir, "train_indices.npy")
    val_idx_path    = os.path.join(splits_dir, "val_indices.npy")
    test_idx_path   = os.path.join(splits_dir, "test_indices.npy")

    for p in [npz_path, meta_path, split_info_path, train_idx_path, val_idx_path, test_idx_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Required file not found: {p}\nRun build_combined_dataset.py and split_large_dataset.py first.")

    data = np.load(npz_path)
    X, y = data["X"], data["y"]

    with open(meta_path) as f:
        metadata = json.load(f)

    train_idx = np.load(train_idx_path)
    val_idx   = np.load(val_idx_path)
    test_idx  = np.load(test_idx_path)

    with open(split_info_path) as f:
        split_info = json.load(f)

    audit_data = {}
    if os.path.exists(split_audit_path):
        with open(split_audit_path) as f:
            audit_data = json.load(f)

    X_train, y_train = X[train_idx], y[train_idx]
    X_val,   y_val   = X[val_idx],   y[val_idx]
    X_test,  y_test  = X[test_idx],  y[test_idx]

    meta_train = [metadata[i] for i in train_idx]
    meta_val   = [metadata[i] for i in val_idx]
    meta_test  = [metadata[i] for i in test_idx]

    return (X_train, y_train, meta_train,
            X_val,   y_val,   meta_val,
            X_test,  y_test,  meta_test,
            split_info, audit_data)


def main():
    parser = argparse.ArgumentParser(description="Stage 3 pipeline for large multi-run SUMO dataset.")
    parser.add_argument("--config", type=str, default="config/stage3_large_dataset.json",
                        help="Path to config JSON file.")
    parser.add_argument("--output-dir", type=str, default="outputs/large_dataset_experiment",
                        help="Base output directory. Use a NEW directory to train a new model.")
    parser.add_argument("--allow-overwrite", action="store_true",
                        help="Allow replacing an existing best_model.pt in --output-dir (dangerous).")
    args = parser.parse_args()

    # Setup directories
    outputs_base = args.output_dir
    checkpoints_dir = os.path.join(outputs_base, "checkpoints")

    # Guard: never silently replace an existing trained checkpoint.
    existing_ckpt = os.path.join(checkpoints_dir, "best_model.pt")
    if os.path.exists(existing_ckpt) and not args.allow_overwrite:
        raise SystemExit(
            f"Refusing to run: '{existing_ckpt}' already exists and would be overwritten.\n"
            f"Pass --output-dir <new_dir> to train into a fresh directory, or "
            f"--allow-overwrite if you really intend to replace it."
        )
    metrics_dir = os.path.join(outputs_base, "metrics")
    plots_dir = os.path.join(outputs_base, "plots")
    logs_dir = os.path.join(outputs_base, "logs")
    reports_dir = os.path.join(outputs_base, "reports")

    for d in [checkpoints_dir, metrics_dir, plots_dir, logs_dir, reports_dir]:
        os.makedirs(d, exist_ok=True)

    log_file = os.path.join(logs_dir, "pipeline.log")
    logger = get_logger("LargeDatasetPipeline", log_file=log_file)

    logger.info("=" * 80)
    logger.info(" Stage 3: Large Dataset LSTM Link-Survival Training & Evaluation")
    logger.info("=" * 80)

    config = load_config(args.config)
    seed   = config["training"]["seed"]
    set_seed(seed)
    dataset_tag = config.get("dataset_tag", "large_dataset")
    logger.info(f"Config: {args.config} | Dataset Tag: {dataset_tag} | Seed: {seed}")

    # 1. Load pre-built dataset + run-disjoint split
    logger.info("Loading pre-built dataset and run-disjoint split...")
    (X_train, y_train, meta_train,
     X_val,   y_val,   meta_val,
     X_test,  y_test,  meta_test,
     split_info, audit_data) = load_large_dataset(config)

    total_seqs = len(y_train) + len(y_val) + len(y_test)
    logger.info(f"Dataset successfully loaded: {total_seqs:,} total sequences")
    logger.info(f"  Train : {len(y_train):,} ({100*len(y_train)/total_seqs:.1f}%) | pos={int(np.sum(y_train==1)):,}, neg={int(np.sum(y_train==0)):,}")
    logger.info(f"  Val   : {len(y_val):,} ({100*len(y_val)/total_seqs:.1f}%) | pos={int(np.sum(y_val==1)):,}, neg={int(np.sum(y_val==0)):,}")
    logger.info(f"  Test  : {len(y_test):,} ({100*len(y_test)/total_seqs:.1f}%) | pos={int(np.sum(y_test==1)):,}, neg={int(np.sum(y_test==0)):,}")

    # 2. Dataset Validation Checks
    logger.info("Performing pre-training dataset validation checks...")
    for name, arr in [("Train", X_train), ("Val", X_val), ("Test", X_test)]:
        nan_count = int(np.sum(np.isnan(arr)))
        inf_count = int(np.sum(np.isinf(arr)))
        assert nan_count == 0, f"NaN values detected in {name} features!"
        assert inf_count == 0, f"Infinite values detected in {name} features!"
        assert arr.shape[1] == 10 and arr.shape[2] == 7, f"Invalid shape {arr.shape} in {name}!"
    logger.info("  ✓ Integrity check: No NaN or Inf feature values. Shapes verified: (N, 10, 7).")

    if audit_data:
        assert audit_data.get("audit_passed", False), "Disjoint split audit check failed!"
        logger.info("  ✓ Disjointness audit verified: ZERO run, vehicle, or pair overlap across splits.")

    # 3. Calculate pos_weight strictly from training set
    n_train_pos = int(np.sum(y_train == 1))
    n_train_neg = int(np.sum(y_train == 0))
    pos_weight = float(n_train_neg / max(n_train_pos, 1))
    logger.info(f"Calculated pos_weight from training set: {pos_weight:.4f} (neg={n_train_neg}, pos={n_train_pos})")

    # 4. Fit scaler strictly on training set
    logger.info("Fitting SequenceStandardScaler on training sequences only...")
    feature_names = config["features"]["feature_names"]
    scaler = SequenceStandardScaler(feature_names=feature_names)
    X_train_sc = scaler.fit_transform(X_train)
    X_val_sc   = scaler.transform(X_val)
    X_test_sc  = scaler.transform(X_test)

    scaler_path = os.path.join(checkpoints_dir, "scaler.json")
    scaler.save_json(scaler_path)
    logger.info(f"Scaler saved to {scaler_path}")

    # 5. Evaluate Non-LSTM Baselines
    comm_range = config["link"]["comm_range_m"]
    horizon_s  = config["link"]["prediction_horizon_s"]
    tune_metric = config["evaluation"].get("tune_metric", "balanced_accuracy")
    logger.info(f"\nEvaluating Non-LSTM Baselines (Threshold tuning metric: '{tune_metric}')...")

    # Baseline 1: 2D Kinematic Extrapolation
    kinematic_base = KinematicExtrapolationBaseline(comm_range_m=comm_range, horizon_s=horizon_s)
    val_probs_kin  = kinematic_base.predict_proba(X_val)
    test_probs_kin = kinematic_base.predict_proba(X_test)
    kin_thresh, kin_val_score = tune_classification_threshold(y_val, val_probs_kin, target_metric=tune_metric)
    kin_test_metrics = compute_classification_metrics(y_test, test_probs_kin, threshold=kin_thresh)
    logger.info(f"  Kinematic Extrapolation -> Val {tune_metric}: {kin_val_score:.4f} (θ*={kin_thresh:.3f}) | Test ROC-AUC: {kin_test_metrics['roc_auc']:.4f}, F1: {kin_test_metrics['f1_score']:.4f}, Spec: {kin_test_metrics['specificity']:.4f}")

    # Baseline 2: Distance Threshold
    dist_base = DistanceThresholdBaseline(comm_range_m=comm_range)
    dist_base.fit_threshold(X_val, y_val)
    val_probs_dist  = dist_base.predict_proba(X_val)
    test_probs_dist = dist_base.predict_proba(X_test)
    dist_thresh, dist_val_score = tune_classification_threshold(y_val, val_probs_dist, target_metric=tune_metric)
    dist_test_metrics = compute_classification_metrics(y_test, test_probs_dist, threshold=dist_thresh)
    logger.info(f"  Distance Threshold (d*={dist_base.best_threshold_dist:.1f}m) -> Val {tune_metric}: {dist_val_score:.4f} (θ*={dist_thresh:.3f}) | Test ROC-AUC: {dist_test_metrics['roc_auc']:.4f}, F1: {dist_test_metrics['f1_score']:.4f}, Spec: {dist_test_metrics['specificity']:.4f}")

    # 6. Build PyTorch DataLoaders & Model
    train_loader, val_loader, test_loader = create_data_loaders(
        train_data=(X_train_sc, y_train, meta_train),
        val_data=(X_val_sc, y_val, meta_val),
        test_data=(X_test_sc, y_test, meta_test),
        batch_size=config["training"]["batch_size"],
    )

    model_cfg = config["model"]
    model = LinkSurvivalLSTM(
        input_dim=model_cfg["input_dim"],
        hidden_dim=model_cfg["hidden_dim"],
        num_layers=model_cfg["num_layers"],
        dropout=model_cfg["dropout"],
        bidirectional=model_cfg["bidirectional"],
    )
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Instantiated {model_cfg['name']}: {n_params:,} trainable parameters.")

    model_config_path = os.path.join(checkpoints_dir, "model_config.json")
    with open(model_config_path, "w") as f:
        json.dump(model_cfg, f, indent=2)

    # 7. Train LSTM
    best_ckpt_path = os.path.join(checkpoints_dir, "best_model.pt")
    logger.info("\nTraining LSTM with Early Stopping...")
    history = train_lstm_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        checkpoint_path=best_ckpt_path,
        max_epochs=config["training"]["max_epochs"],
        learning_rate=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
        patience=config["training"]["early_stopping_patience"],
        pos_weight=pos_weight,
        config=config,
        device=torch.device("cpu"),
    )
    logger.info(f"Training completed. Best Validation Loss: {history['best_val_loss']:.4f} at Epoch {history['best_epoch']}.")

    # 8. Reload Best Checkpoint & Tune Threshold on Validation Data Only
    load_checkpoint(best_ckpt_path, model)
    criterion = torch.nn.BCEWithLogitsLoss()
    _, val_probs_lstm, _ = evaluate_loss_and_predictions(model, val_loader, criterion, torch.device("cpu"))
    lstm_thresh, lstm_val_score = tune_classification_threshold(y_val, val_probs_lstm, target_metric=tune_metric)
    logger.info(f"Validation threshold tuned: optimal θ* = {lstm_thresh:.3f} (Val {tune_metric}: {lstm_val_score:.4f})")

    # Keyword arguments are required: the previous positional call passed
    # history["best_epoch"] into the `optimizer` slot and crashed here.
    # NOTE: `val_f1` holds the validation score of `tune_metric`
    # (balanced_accuracy by default), not necessarily F1.
    save_checkpoint(
        filepath=best_ckpt_path,
        model=model,
        epoch=history["best_epoch"],
        val_loss=history["best_val_loss"],
        val_f1=lstm_val_score,
        best_threshold=lstm_thresh,
        config=config,
    )

    # 9. Test Set Evaluation
    _, test_probs_lstm, _ = evaluate_loss_and_predictions(model, test_loader, criterion, torch.device("cpu"))
    lstm_test_metrics = compute_classification_metrics(y_test, test_probs_lstm, threshold=lstm_thresh)

    # 10. Multi-Model Comparison Table
    logger.info("\n" + "=" * 125)
    logger.info(f" FINAL HELD-OUT TEST EVALUATION COMPARISON ({dataset_tag.upper()} — {len(y_test):,} samples)")
    logger.info("=" * 125)
    hdr = f"{'Model':<34} | {'F1':>6} | {'ROC-AUC':>7} | {'PR-AUC':>6} | {'Prec':>6} | {'Rec':>6} | {'Spec':>6} | {'Acc':>6} | {'Bal-Acc':>7} | {'TP/FP/TN/FN'}"
    logger.info(hdr)
    logger.info("-" * 125)

    comparison_rows = [
        {"Model": "Kinematic Extrapolation",   **kin_test_metrics},
        {"Model": "Distance Threshold",         **dist_test_metrics},
        {"Model": "LSTM Link-Survival",         **lstm_test_metrics},
    ]

    for r in comparison_rows:
        cm = r["confusion_matrix"]
        tp, fp = cm["tp"], cm["fp"]
        tn, fn = cm["tn"], cm["fn"]
        logger.info(
            f"{r['Model']:<34} | "
            f"{r['f1_score']:>6.4f} | "
            f"{r['roc_auc']:>7.4f} | "
            f"{r['pr_auc']:>6.4f} | "
            f"{r['precision']:>6.4f} | "
            f"{r['recall']:>6.4f} | "
            f"{r['specificity']:>6.4f} | "
            f"{r['accuracy']:>6.4f} | "
            f"{r['balanced_accuracy']:>7.4f} | "
            f"TP={tp} FP={fp} TN={tn} FN={fn}"
        )
    logger.info("=" * 125)

    # Objective conclusion
    lstm_auc = lstm_test_metrics["roc_auc"]
    kin_auc  = kin_test_metrics["roc_auc"]
    lstm_f1  = lstm_test_metrics["f1_score"]
    kin_f1   = kin_test_metrics["f1_score"]
    lstm_ba  = lstm_test_metrics["balanced_accuracy"]
    kin_ba   = kin_test_metrics["balanced_accuracy"]

    if lstm_auc > kin_auc and lstm_f1 > kin_f1:
        conclusion = "LSTM outperforms the Kinematic Baseline on both ROC-AUC and F1-score."
    elif lstm_auc > kin_auc:
        conclusion = "LSTM outperforms the Kinematic Baseline on ROC-AUC, with comparable balanced classification performance."
    elif lstm_ba > kin_ba:
        conclusion = "LSTM outperforms the Kinematic Baseline on balanced accuracy, while kinematic baseline remains competitive on discrimination."
    else:
        conclusion = "Kinematic baseline outperforms or matches LSTM on held-out test data."
    logger.info(f"\nCONCLUSION: {conclusion}\n")

    # 11. Plot Curves
    logger.info("Generating evaluation plots...")
    # Loss curves
    loss_path = os.path.join(plots_dir, "loss_curves.png")
    plot_loss_curves(history["train_loss"], history["val_loss"], history["best_epoch"], loss_path)

    # Confusion matrices
    cm_lstm_path = os.path.join(plots_dir, "confusion_matrix_lstm.png")
    plot_confusion_matrix(lstm_test_metrics["confusion_matrix"], "LSTM Link-Survival", cm_lstm_path)

    cm_kin_path = os.path.join(plots_dir, "confusion_matrix_kinematic.png")
    plot_confusion_matrix(kin_test_metrics["confusion_matrix"], "Kinematic Baseline", cm_kin_path)

    cm_dist_path = os.path.join(plots_dir, "confusion_matrix_distance.png")
    plot_confusion_matrix(dist_test_metrics["confusion_matrix"], "Distance Baseline", cm_dist_path)

    # Multi-model ROC Curves
    roc_curves_data = {
        "LSTM Link-Survival": compute_roc_curve_points(y_test, test_probs_lstm),
        "Kinematic Baseline": compute_roc_curve_points(y_test, test_probs_kin),
        "Distance Baseline": compute_roc_curve_points(y_test, test_probs_dist),
    }
    roc_plot_path = os.path.join(plots_dir, "roc_curves_comparison.png")
    plot_roc_curves(roc_curves_data, roc_plot_path)

    # Multi-model PR Curves
    pr_curves_data = {
        "LSTM Link-Survival": compute_pr_curve_points(y_test, test_probs_lstm),
        "Kinematic Baseline": compute_pr_curve_points(y_test, test_probs_kin),
        "Distance Baseline": compute_pr_curve_points(y_test, test_probs_dist),
    }
    pr_plot_path = os.path.join(plots_dir, "pr_curves_comparison.png")
    plot_pr_curves(pr_curves_data, pr_plot_path)

    logger.info(f"Plots saved to {plots_dir}")

    # 12. Save Metrics & Reports
    metrics_out = os.path.join(metrics_dir, "test_metrics.json")
    with open(metrics_out, "w") as f:
        json.dump({
            "dataset_tag": dataset_tag,
            "total_sequences": total_seqs,
            "train_samples": len(y_train),
            "val_samples": len(y_val),
            "test_samples": len(y_test),
            "pos_weight_calculated": pos_weight,
            "tuned_threshold_lstm": float(lstm_thresh),
            "tuned_threshold_kinematic": float(kin_thresh),
            "tuned_threshold_distance": float(dist_thresh),
            "conclusion": conclusion,
            "comparison": comparison_rows
        }, f, indent=2)

    # CSV report
    csv_rows = []
    for r in comparison_rows:
        cm = r["confusion_matrix"]
        csv_rows.append({
            "Model": r["Model"],
            "F1-Score": r["f1_score"],
            "ROC-AUC": r["roc_auc"],
            "PR-AUC": r["pr_auc"],
            "Precision": r["precision"],
            "Recall": r["recall"],
            "Specificity": r["specificity"],
            "Accuracy": r["accuracy"],
            "Balanced-Accuracy": r["balanced_accuracy"],
            "TP": cm["tp"],
            "FP": cm["fp"],
            "TN": cm["tn"],
            "FN": cm["fn"]
        })
    df_cmp = pd.DataFrame(csv_rows)
    csv_report_path = os.path.join(reports_dir, "model_comparison.csv")
    df_cmp.to_csv(csv_report_path, index=False)

    # Markdown report
    md_report_path = os.path.join(reports_dir, "experiment_summary.md")
    with open(md_report_path, "w") as f:
        f.write(f"# Stage 3 Large Dataset Experiment Summary\n\n")
        f.write(f"**Dataset Tag:** `{dataset_tag}`  \n")
        f.write(f"**Total Sequences:** {total_seqs:,} (Train: {len(y_train):,}, Val: {len(y_val):,}, Test: {len(y_test):,})  \n")
        f.write(f"**Split Strategy:** `run_disjoint` (Zero run/vehicle/pair leakage)  \n")
        f.write(f"**BCE Pos Weight:** {pos_weight:.4f}  \n\n")
        f.write(f"## Final Test Evaluation Table\n\n")
        f.write(df_cmp.to_markdown(index=False))
        f.write(f"\n\n## Conclusion\n\n{conclusion}\n")

    logger.info(f"Reports saved to {reports_dir}")

    # 13. Verify Predictor Interface
    logger.info("\nVerifying Stage 4 LinkSurvivalPredictor interface...")
    predictor = LinkSurvivalPredictor(best_ckpt_path, model_config_path, scaler_path)
    sample_p = predictor.predict_probability(X_test[0])
    sample_rel = predictor.is_link_reliable(sample_p)
    logger.info(f"Smoke test passed: Sample P(survival) = {sample_p:.4f}, Reliable = {sample_rel}")

    logger.info(f"\nLarge Dataset Pipeline ({dataset_tag}) finished successfully!")


if __name__ == "__main__":
    main()
