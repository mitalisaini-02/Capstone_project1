#!/usr/bin/env python3
"""
scripts/run_stage3_pipeline.py
Master Entry-Point for Stage 3: LSTM Link-Survival Prediction Pipeline.

Supports both 300s initial trace and full 659s trace via configurable JSON configs.
Evaluates 2D Kinematic Extrapolation, Distance Threshold, and LSTM models on held-out test data.
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import torch

# Ensure project root is in python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils.logger import get_logger
from src.utils.seed import set_seed
from src.utils.config_loader import load_config
from src.data.loader import load_mobility_data
from src.features.sequence_builder import build_feature_sequences
from src.data.splitter import split_dataset
from src.features.scaler import SequenceStandardScaler
from src.evaluation.baselines import KinematicExtrapolationBaseline, DistanceThresholdBaseline
from src.models.lstm import LinkSurvivalLSTM
from src.training.dataset import create_data_loaders
from src.training.trainer import train_lstm_model, evaluate_loss_and_predictions
from src.training.checkpoints import load_checkpoint, save_checkpoint
from src.evaluation.metrics import compute_classification_metrics, tune_classification_threshold
from src.evaluation.visualizer import plot_loss_curves, plot_confusion_matrix
from src.models.predictor import LinkSurvivalPredictor


def main():
    parser = argparse.ArgumentParser(description="Run Stage 3 LSTM Link-Survival Prediction Pipeline")
    parser.add_argument("--config", type=str, default="config/stage3_config.json", help="Path to config file")
    parser.add_argument("--split-strategy", type=str, default=None, help="Override split strategy")
    args = parser.parse_args()

    logger = get_logger("Stage3Pipeline")
    logger.info("=" * 80)
    logger.info(" Stage 3: LSTM Link-Survival Prediction Pipeline")
    logger.info("=" * 80)

    # 1. Load configuration
    config = load_config(args.config)
    if args.split_strategy:
        config["split"]["strategy"] = args.split_strategy

    seed = config["training"]["seed"]
    set_seed(seed)
    dataset_tag = config.get("dataset_tag", "300s_trace")
    logger.info(f"Configuration: '{args.config}' | Dataset Tag: {dataset_tag} | Seed: {seed}")

    # 2. Load and validate mobility data
    raw_file = config["data"]["raw_mobility_file"]
    max_time_s = config["data"]["max_simulation_time_s"]
    logger.info(f"Loading and validating mobility data from '{raw_file}' (t <= {max_time_s}s)...")

    mobility_df, lifecycles = load_mobility_data(
        mobility_filepath=raw_file,
        roles_filepath=config["data"]["ue_roles_file"],
        max_time_s=max_time_s
    )
    logger.info(f"Validated mobility records: {len(mobility_df)} rows, {len(lifecycles)} vehicles.")

    # 3. Construct feature sequences and labels
    comm_range = config["link"]["comm_range_m"]
    horizon_s = config["link"]["prediction_horizon_s"]
    hist_window = config["link"]["history_window_s"]
    step_s = config["data"]["step_length_s"]

    logger.info(
        f"Generating sequences: History H={hist_window}s, Horizon T={horizon_s}s, Range R={comm_range}m..."
    )
    X, y, metadata = build_feature_sequences(
        df=mobility_df,
        lifecycles=lifecycles,
        comm_range_m=comm_range,
        history_window_s=hist_window,
        prediction_horizon_s=horizon_s,
        step_s=step_s
    )
    logger.info(f"Generated feature tensor X shape: {X.shape}, label vector y shape: {y.shape}")

    unique_labels, label_counts = np.unique(y, return_counts=True)
    class_dist = {int(u): int(c) for u, c in zip(unique_labels, label_counts)}
    pos_ratio = class_dist.get(1, 0) / len(y) if len(y) > 0 else 0.0
    logger.info(f"Class distribution: {class_dist} (Positive / Surviving ratio: {pos_ratio * 100:.1f}%)")

    # Save processed raw sequences
    processed_dir = config["data"]["processed_dir"]
    os.makedirs(processed_dir, exist_ok=True)
    np.savez_compressed(os.path.join(processed_dir, "pairwise_sequences.npz"), X=X, y=y)
    with open(os.path.join(processed_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    # 4. Leakage-safe Dataset Split
    split_strategy = config["split"]["strategy"]
    logger.info(f"Splitting dataset using strategy: '{split_strategy}'...")
    train_data, val_data, test_data, split_info = split_dataset(
        X=X,
        y=y,
        metadata=metadata,
        strategy=split_strategy,
        train_ratio=config["split"]["train_ratio"],
        val_ratio=config["split"]["val_ratio"],
        test_ratio=config["split"]["test_ratio"],
        seed=seed
    )
    X_train, y_train, meta_train = train_data
    X_val, y_val, meta_val = val_data
    X_test, y_test, meta_test = test_data

    logger.info(f"Split results: Train={len(y_train)} samples, Val={len(y_val)} samples, Test={len(y_test)} samples.")
    splits_dir = config["data"]["splits_dir"]
    os.makedirs(splits_dir, exist_ok=True)
    with open(os.path.join(splits_dir, "split_info.json"), "w") as f:
        json.dump(split_info, f, indent=2)

    # 5. Fit Scaler strictly on training set
    logger.info("Fitting SequenceStandardScaler strictly on training sequences...")
    feature_names = config["features"]["feature_names"]
    scaler = SequenceStandardScaler(feature_names=feature_names)
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    checkpoints_dir = config["outputs"]["checkpoints_dir"]
    os.makedirs(checkpoints_dir, exist_ok=True)
    scaler_path = os.path.join(checkpoints_dir, "scaler.json")
    scaler.save_json(scaler_path)
    logger.info(f"Scaler statistics saved to: {scaler_path}")

    # 6. Evaluate Non-LSTM Baselines
    logger.info("\nEvaluating Non-LSTM Baselines on Validation and Test sets...")
    tune_metric = config["evaluation"].get("tune_metric", "balanced_accuracy")

    # Baseline 1: 2D Kinematic Extrapolation
    kinematic_base = KinematicExtrapolationBaseline(comm_range_m=comm_range, horizon_s=horizon_s)
    val_probs_kin = kinematic_base.predict_proba(X_val)
    test_probs_kin = kinematic_base.predict_proba(X_test)

    kin_val_thresh, _ = tune_classification_threshold(y_val, val_probs_kin, target_metric=tune_metric)
    kin_test_metrics = compute_classification_metrics(y_test, test_probs_kin, threshold=kin_val_thresh)
    logger.info(
        f"Kinematic Baseline -> Test F1: {kin_test_metrics['f1_score']:.4f}, "
        f"ROC-AUC: {kin_test_metrics['roc_auc']:.4f}, PR-AUC: {kin_test_metrics['pr_auc']:.4f}, Specificity: {kin_test_metrics['specificity']:.4f}"
    )

    # Baseline 2: Distance Threshold
    dist_base = DistanceThresholdBaseline(comm_range_m=comm_range)
    dist_base.fit_threshold(X_val, y_val)
    dist_val_probs = dist_base.predict_proba(X_val)
    dist_test_probs = dist_base.predict_proba(X_test)

    dist_thresh, _ = tune_classification_threshold(y_val, dist_val_probs, target_metric=tune_metric)
    dist_test_metrics = compute_classification_metrics(y_test, dist_test_probs, threshold=dist_thresh)
    logger.info(
        f"Distance Threshold Baseline (d*={dist_base.best_threshold_dist:.1f}m) -> Test F1: {dist_test_metrics['f1_score']:.4f}, "
        f"ROC-AUC: {dist_test_metrics['roc_auc']:.4f}, PR-AUC: {dist_test_metrics['pr_auc']:.4f}, Specificity: {dist_test_metrics['specificity']:.4f}"
    )

    # 7. Build PyTorch DataLoaders & Model
    train_loader, val_loader, test_loader = create_data_loaders(
        train_data=(X_train_scaled, y_train, meta_train),
        val_data=(X_val_scaled, y_val, meta_val),
        test_data=(X_test_scaled, y_test, meta_test),
        batch_size=config["training"]["batch_size"]
    )

    model_cfg = config["model"]
    model = LinkSurvivalLSTM(
        input_dim=model_cfg["input_dim"],
        hidden_dim=model_cfg["hidden_dim"],
        num_layers=model_cfg["num_layers"],
        dropout=model_cfg["dropout"],
        bidirectional=model_cfg["bidirectional"]
    )
    logger.info(f"Instantiated {model_cfg['name']}: {sum(p.numel() for p in model.parameters())} trainable parameters.")

    model_config_path = os.path.join(checkpoints_dir, "model_config.json")
    with open(model_config_path, "w") as f:
        json.dump(model_cfg, f, indent=2)

    # 8. Train Model
    best_checkpoint_path = os.path.join(checkpoints_dir, "best_model.pt")
    logger.info("Beginning LSTM training with Early Stopping...")

    history = train_lstm_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        checkpoint_path=best_checkpoint_path,
        max_epochs=config["training"]["max_epochs"],
        learning_rate=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
        patience=config["training"]["early_stopping_patience"],
        pos_weight=config["training"].get("pos_weight"),
        config=config,
        device=torch.device("cpu")
    )
    logger.info(
        f"Training completed. Best Validation Loss: {history['best_val_loss']:.4f} at Epoch {history['best_epoch']}."
    )

    # 9. Reload Best Model Checkpoint & Tune Threshold
    load_checkpoint(best_checkpoint_path, model)
    criterion = torch.nn.BCEWithLogitsLoss()
    _, val_probs_lstm, _ = evaluate_loss_and_predictions(model, val_loader, criterion, torch.device("cpu"))

    tuned_threshold, val_score = tune_classification_threshold(y_val, val_probs_lstm, target_metric=tune_metric)
    logger.info(f"Validation threshold tuned: optimal theta* = {tuned_threshold:.3f} (Val {tune_metric}: {val_score:.4f})")

    # Update checkpoint with tuned threshold
    save_checkpoint(
        filepath=best_checkpoint_path,
        model=model,
        epoch=history["best_epoch"],
        val_loss=history["best_val_loss"],
        val_f1=val_score,
        best_threshold=tuned_threshold,
        config=config
    )

    # 10. Held-Out Test Evaluation
    _, test_probs_lstm, _ = evaluate_loss_and_predictions(model, test_loader, criterion, torch.device("cpu"))
    lstm_test_metrics = compute_classification_metrics(y_test, test_probs_lstm, threshold=tuned_threshold)

    logger.info("\n" + "=" * 120)
    logger.info(f" FINAL TEST EVALUATION COMPARISON (HELD-OUT TEST SET — {dataset_tag.upper()})")
    logger.info("=" * 120)
    logger.info(f"{'Model':<34} | {'F1':<8} | {'ROC-AUC':<8} | {'PR-AUC':<8} | {'Prec':<8} | {'Rec':<8} | {'Spec':<8} | {'Acc':<8} | {'Pred(+/-)':<10}")
    logger.info("-" * 120)

    comparison_rows = [
        {"Model": "Kinematic Extrapolation Baseline", **kin_test_metrics},
        {"Model": "Distance Threshold Baseline", **dist_test_metrics},
        {"Model": "LSTM Link-Survival Model", **lstm_test_metrics}
    ]

    for r in comparison_rows:
        pred_ratio = f"{r['predicted_positive_count']}/{r['predicted_negative_count']}"
        logger.info(
            f"{r['Model']:<34} | "
            f"{r['f1_score']:<8.4f} | "
            f"{r['roc_auc']:<8.4f} | "
            f"{r['pr_auc']:<8.4f} | "
            f"{r['precision']:<8.4f} | "
            f"{r['recall']:<8.4f} | "
            f"{r['specificity']:<8.4f} | "
            f"{r['accuracy']:<8.4f} | "
            f"{pred_ratio:<10}"
        )
    logger.info("=" * 120 + "\n")

    # 11. Save Metrics and Plots
    metrics_dir = config["outputs"]["metrics_dir"]
    os.makedirs(metrics_dir, exist_ok=True)
    metrics_path = os.path.join(metrics_dir, "test_metrics.json")
    test_u, test_c = np.unique(y_test, return_counts=True)
    test_class_dist = {int(k): int(v) for k, v in zip(test_u, test_c)}
    with open(metrics_path, "w") as f:
        json.dump({
            "dataset_tag": dataset_tag,
            "split_strategy": split_strategy,
            "test_samples": len(y_test),
            "test_class_distribution": test_class_dist,
            "tuned_threshold": float(tuned_threshold),
            "comparison": comparison_rows
        }, f, indent=2)
    logger.info(f"Evaluation metrics saved to: {metrics_path}")

    plots_dir = config["outputs"]["plots_dir"]
    os.makedirs(plots_dir, exist_ok=True)

    loss_curve_path = os.path.join(plots_dir, "loss_curves.png")
    plot_loss_curves(history["train_loss"], history["val_loss"], history["best_epoch"], loss_curve_path)
    logger.info(f"Loss curves saved to: {loss_curve_path}")

    cm_path = os.path.join(plots_dir, "confusion_matrix.png")
    plot_confusion_matrix(lstm_test_metrics["confusion_matrix"], f"LSTM Link-Survival ({dataset_tag})", cm_path)
    logger.info(f"Confusion matrix saved to: {cm_path}")

    # 12. Verify Stage 4 Predictor Interface
    logger.info("\nVerifying Stage 4 LinkSurvivalPredictor interface...")
    predictor = LinkSurvivalPredictor(
        checkpoint_path=best_checkpoint_path,
        config_path=model_config_path,
        scaler_path=scaler_path
    )
    sample_seq = X_test[0]
    pred_p = predictor.predict_probability(sample_seq)
    is_reliable = predictor.is_link_reliable(pred_p)
    logger.info(f"Inference smoke test passed: Sample P(survival) = {pred_p:.4f}, Reliable = {is_reliable}")

    logger.info(f"\n Stage 3 Pipeline execution for '{dataset_tag}' completed successfully!")


if __name__ == "__main__":
    main()
