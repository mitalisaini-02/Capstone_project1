# Stage 3 Evaluation Report — Large Dataset Experiment

Generated: 2026-11-24 03:51:13

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

Training: 13 epochs (early stopping), best val loss = 0.3318  
Val threshold: θ* = 0.53 (tuned on validation, balanced_accuracy = 0.7805)  
Val F1: 0.8136

## Test-Set Results (θ* fixed from validation — no test-set leakage)

| Model | ROC-AUC | PR-AUC | F1 | Bal. Acc | Precision | Recall | Specificity | Brier |
|-------|---------|--------|----|----------|-----------|--------|-------------|-------|
| **LSTM** | **0.8506** | **0.8967** | **0.788** | **0.7591** | 0.8454 | 0.738 | 0.7802 | 0.1574 |
| Kinematic Extrapolation | 0.8354 | 0.8783 | 0.7941 | 0.7532 | 0.8292 | 0.7618 | 0.7445 | 0.1951 |
| Distance Threshold | 0.6562 | 0.7388 | 0.6692 | 0.6176 | 0.7235 | 0.6224 | 0.6128 | 0.2671 |

## Confusion Matrices

### LSTM (θ* = 0.53)
|  | Predicted Survive | Predicted Break |
|--|---|---|
| **Actual Survive** | TP = 11710 | FN = 4157 |
| **Actual Break** | FP = 2142 | TN = 7604 |

### Kinematic Extrapolation (θ* = 0.43000000000000005)
|  | Predicted Survive | Predicted Break |
|--|---|---|
| **Actual Survive** | TP = 12088 | FN = 3779 |
| **Actual Break** | FP = 2490 | TN = 7256 |

### Distance Threshold (d* = 740.0 m, θ* = 0.5)
|  | Predicted Survive | Predicted Break |
|--|---|---|
| **Actual Survive** | TP = 9876 | FN = 5991 |
| **Actual Break** | FP = 3774 | TN = 5972 |

## Assessment

⚠️ LSTM does NOT outperform Kinematic Extrapolation baseline on all metrics — honest reporting required

LSTM vs Kinematic:
- ROC-AUC: 0.8506 vs 0.8354 (+0.0152)
- F1: 0.788 vs 0.7941 (-0.0061)
- Balanced Acc: 0.7591 vs 0.7532 (+0.0059)

## Notes

- **Loss curves** not available: pipeline crashed (positional-arg bug in save_checkpoint) before persisting
  epoch-by-epoch history. Best epoch = 13, best val_loss = 0.3318.  
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
