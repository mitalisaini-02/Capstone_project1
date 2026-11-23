"""
run_large_dataset_pipeline.py
------------------------------
Stage 3 LSTM pipeline for the large multi-run SUMO dataset.

Differences from run_stage3_pipeline.py:
  - Loads pre-built pairwise_sequences.npz and metadata.json (from build_combined_dataset.py)
  - Uses pre-built run-disjoint split indices (from split_large_dataset.py)
  - Saves all outputs to outputs/large_dataset_experiment/
  - Does NOT modify Stage 1, Stage 2, or existing datasets

Usage:
  python scripts/run_large_dataset_pipeline.py --config config/stage3_large_dataset.json
"""

import os
import sys
import json
import argparse
import numpy as np
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
from src.evaluation.metrics import compute_classification_metrics, tune_classification_threshold
from src.evaluation.visualizer import plot_loss_curves, plot_confusion_matrix
from src.models.predictor import LinkSurvivalPredictor


def load_large_dataset(config: dict):
    """
    Load pre-built sequences and run-disjoint split indices.
    Returns (X_train, y_train, meta_train, X_val, y_val, meta_val, X_test, y_test, meta_test)
    """
    dataset_dir = config["data"]["processed_dir"]
    splits_dir  = config["data"]["splits_dir"]

    npz_path  = os.path.join(dataset_dir, "pairwise_sequences.npz")
    meta_path = os.path.join(dataset_dir, "metadata.json")
    split_info_path = os.path.join(splits_dir, "split_info.json")
    train_idx_path  = os.path.join(splits_dir, "train_indices.npy")
    val_idx_path    = os.path.join(splits_dir, "val_indices.npy")
    test_idx_path   = os.path.join(splits_dir, "test_indices.npy")

    for p in [npz_path, meta_path, split_info_path, train_idx_path, val_idx_path, test_idx_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Required file not found: {p}\n"
                f"Run build_combined_dataset.py and split_large_dataset.py first."
            )

    data   = np.load(npz_path)
    X, y   = data["X"], data["y"]

    with open(meta_path) as f:
        metadata = json.load(f)

    train_idx = np.load(train_idx_path)
    val_idx   = np.load(val_idx_path)
    test_idx  = np.load(test_idx_path)

    with open(split_info_path) as f:
        split_info = json.load(f)

    X_train, y_train = X[train_idx], y[train_idx]
    X_val,   y_val   = X[val_idx],   y[val_idx]
    X_test,  y_test  = X[test_idx],  y[test_idx]

    meta_train = [metadata[i] for i in train_idx]
    meta_val   = [metadata[i] for i in val_idx]
    meta_test  = [metadata[i] for i in test_idx]

    return (X_train, y_train, meta_train,
            X_val,   y_val,   meta_val,
            X_test,  y_test,  meta_test,
            split_info)


def main():
    parser = argparse.ArgumentParser(description="Stage 3 pipeline for large multi-run SUMO dataset.")
    parser.add_argument("--config", type=str, default="config/stage3_large_dataset.json",
                        help="Path to config JSON file.")
    args = parser.parse_args()

    logger = get_logger("LargeDatasetPipeline")
    logger.info("=" * 80)
    logger.info(" Stage 3 — Large Dataset LSTM Pipeline")
    logger.info("=" * 80)

    config = load_config(args.config)
    seed   = config["training"]["seed"]
    set_seed(seed)
    dataset_tag = config.get("dataset_tag", "large_dataset")
    logger.info(f"Config: {args.config} | Tag: {dataset_tag} | Seed: {seed}")

    # 1. Load pre-built dataset + run-disjoint split
    logger.info("Loading pre-built dataset and run-disjoint split indices...")
    (X_train, y_train, meta_train,
     X_val,   y_val,   meta_val,
     X_test,  y_test,  meta_test,
     split_info) = load_large_dataset(config)

    total = len(y_train) + len(y_val) + len(y_test)
    logger.info(f"Dataset loaded: {total:,} total sequences")
    logger.info(f"  Train : {len(y_train):,} | pos={int(np.sum(y_train==1)):,} neg={int(np.sum(y_train==0)):,}")
    logger.info(f"  Val   : {len(y_val):,}   | pos={int(np.sum(y_val==1)):,}   neg={int(np.sum(y_val==0)):,}")
    logger.info(f"  Test  : {len(y_test):,}  | pos={int(np.sum(y_test==1)):,}  neg={int(np.sum(y_test==0)):,}")
    logger.info(f"  Runs — Train: {split_info.get('n_runs_train','?')}  Val: {split_info.get('n_runs_val','?')}  Test: {split_info.get('n_runs_test','?')}")

    # 2. Fit scaler on TRAIN ONLY
    logger.info("Fitting SequenceStandardScaler on training set only...")
    feature_names = config["features"]["feature_names"]
    scaler = SequenceStandardScaler(feature_names=feature_names)
    X_train_sc = scaler.fit_transform(X_train)
    X_val_sc   = scaler.transform(X_val)
    X_test_sc  = scaler.transform(X_test)

    checkpoints_dir = config["outputs"]["checkpoints_dir"]
    os.makedirs(checkpoints_dir, exist_ok=True)
    scaler_path = os.path.join(checkpoints_dir, "scaler.json")
    scaler.save_json(scaler_path)
    logger.info(f"Scaler saved: {scaler_path}")

    # Save split_info for reproducibility
    splits_out = os.path.join(checkpoints_dir, "split_info.json")
    with open(splits_out, "w") as f:
        json.dump(split_info, f, indent=2)

    # 3. Baselines (use scaled X for LSTM; raw X for baselines since they interpret features directly)
    comm_range = config["link"]["comm_range_m"]
    horizon_s  = config["link"]["prediction_horizon_s"]
    tune_metric = config["evaluation"].get("tune_metric", "balanced_accuracy")
    logger.info("Evaluating baselines on Val and Test...")

    kinematic_base   = KinematicExtrapolationBaseline(comm_range_m=comm_range, horizon_s=horizon_s)
    val_probs_kin    = kinematic_base.predict_proba(X_val)
    test_probs_kin   = kinematic_base.predict_proba(X_test)
    kin_thresh, _    = tune_classification_threshold(y_val, val_probs_kin, target_metric=tune_metric)
    kin_test_metrics = compute_classification_metrics(y_test, test_probs_kin, threshold=kin_thresh)
    logger.info(f"  Kinematic: ROC-AUC={kin_test_metrics['roc_auc']:.4f}  F1={kin_test_metrics['f1_score']:.4f}  Spec={kin_test_metrics['specificity']:.4f}")

    dist_base  = DistanceThresholdBaseline(comm_range_m=comm_range)
    dist_base.fit_threshold(X_val, y_val)
    val_probs_dist    = dist_base.predict_proba(X_val)
    test_probs_dist   = dist_base.predict_proba(X_test)
    dist_thresh, _    = tune_classification_threshold(y_val, val_probs_dist, target_metric=tune_metric)
    dist_test_metrics = compute_classification_metrics(y_test, test_probs_dist, threshold=dist_thresh)
    logger.info(f"  Distance:  ROC-AUC={dist_test_metrics['roc_auc']:.4f}  F1={dist_test_metrics['f1_score']:.4f}  Spec={dist_test_metrics['specificity']:.4f}")

    # 4. LSTM
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
    logger.info(f"LSTM: {n_params:,} trainable parameters")

    model_config_path = os.path.join(checkpoints_dir, "model_config.json")
    with open(model_config_path, "w") as f:
        json.dump(model_cfg, f, indent=2)

    best_ckpt_path = os.path.join(checkpoints_dir, "best_model.pt")
    logger.info("Training LSTM...")
    history = train_lstm_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        checkpoint_path=best_ckpt_path,
        max_epochs=config["training"]["max_epochs"],
        learning_rate=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
        patience=config["training"]["early_stopping_patience"],
        pos_weight=config["training"].get("pos_weight"),
        config=config,
        device=torch.device("cpu"),
    )
    logger.info(f"Training done. Best val loss: {history['best_val_loss']:.4f} at epoch {history['best_epoch']}")

    # 5. Tune threshold on val, evaluate on test
    load_checkpoint(best_ckpt_path, model)
    criterion = torch.nn.BCEWithLogitsLoss()
    _, val_probs_lstm, _ = evaluate_loss_and_predictions(model, val_loader, criterion, torch.device("cpu"))
    tuned_threshold, val_score = tune_classification_threshold(y_val, val_probs_lstm, target_metric=tune_metric)
    logger.info(f"Threshold tuned: θ*={tuned_threshold:.3f}  Val {tune_metric}={val_score:.4f}")

    save_checkpoint(best_ckpt_path, model, history["best_epoch"], history["best_val_loss"],
                    val_score, tuned_threshold, config)

    _, test_probs_lstm, _ = evaluate_loss_and_predictions(model, test_loader, criterion, torch.device("cpu"))
    lstm_test_metrics = compute_classification_metrics(y_test, test_probs_lstm, threshold=tuned_threshold)

    # 6. Report
    logger.info("\n" + "=" * 120)
    logger.info(f" FINAL TEST RESULTS — {dataset_tag.upper()}")
    logger.info("=" * 120)
    hdr = f"{'Model':<36} | {'F1':>6} | {'ROC-AUC':>7} | {'PR-AUC':>6} | {'Prec':>6} | {'Rec':>6} | {'Spec':>6} | {'Acc':>6} | {'Bal-Acc':>7} | {'TP/FP/TN/FN'}"
    logger.info(hdr)
    logger.info("-" * 120)

    rows = [
        {"Model": "Kinematic Extrapolation",   **kin_test_metrics},
        {"Model": "Distance Threshold",         **dist_test_metrics},
        {"Model": "LSTM Link-Survival",         **lstm_test_metrics},
    ]
    for r in rows:
        cm = r["confusion_matrix"]
        tp, fp = cm[1][1], cm[0][1]
        tn, fn = cm[0][0], cm[1][0]
        logger.info(
            f"{r['Model']:<36} | "
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
    logger.info("=" * 120)

    # Honest conclusion
    lstm_auc  = lstm_test_metrics["roc_auc"]
    kin_auc   = kin_test_metrics["roc_auc"]
    lstm_ba   = lstm_test_metrics.get("balanced_accuracy", 0)
    kin_ba    = kin_test_metrics.get("balanced_accuracy", 0)
    if lstm_auc > kin_auc and lstm_ba > kin_ba:
        conclusion = "LSTM OUTPERFORMS kinematic baseline on both ROC-AUC and balanced accuracy."
    elif lstm_auc > kin_auc:
        conclusion = "LSTM has higher ROC-AUC than kinematic baseline, but not higher balanced accuracy."
    else:
        conclusion = "LSTM does NOT outperform kinematic baseline. See metrics above."
    logger.info(f"\nCONCLUSION: {conclusion}")

    # 7. Save metrics and plots
    metrics_dir = config["outputs"]["metrics_dir"]
    os.makedirs(metrics_dir, exist_ok=True)
    metrics_out = os.path.join(metrics_dir, "test_metrics.json")
    with open(metrics_out, "w") as f:
        json.dump({
            "dataset_tag": dataset_tag,
            "split_strategy": "run_disjoint",
            "total_sequences": total,
            "train_sequences": len(y_train),
            "val_sequences": len(y_val),
            "test_sequences": len(y_test),
            "split_info_summary": {
                k: v for k, v in split_info.items()
                if k not in ("train_runs", "val_runs", "test_runs")
            },
            "tuned_threshold": float(tuned_threshold),
            "conclusion": conclusion,
            "comparison": [
                {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in r.items()}
                for r in rows
            ],
        }, f, indent=2)
    logger.info(f"Metrics saved: {metrics_out}")

    plots_dir = config["outputs"]["plots_dir"]
    os.makedirs(plots_dir, exist_ok=True)
    plot_loss_curves(history["train_loss"], history["val_loss"], history["best_epoch"],
                     os.path.join(plots_dir, "loss_curves.png"))
    plot_confusion_matrix(lstm_test_metrics["confusion_matrix"],
                          f"LSTM ({dataset_tag})",
                          os.path.join(plots_dir, "confusion_matrix.png"))
    logger.info(f"Plots saved to {plots_dir}")

    # 8. Smoke-test predictor interface
    predictor = LinkSurvivalPredictor(best_ckpt_path, model_config_path, scaler_path)
    p = predictor.predict_probability(X_test[0])
    logger.info(f"Predictor smoke test: P(survival)={p:.4f}")
    logger.info(f"\nLarge Dataset Pipeline ({dataset_tag}) completed successfully!")


if __name__ == "__main__":
    main()
