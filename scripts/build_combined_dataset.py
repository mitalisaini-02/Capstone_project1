"""
build_combined_dataset.py
--------------------------
Combines multiple SUMO run mobility CSVs (including pilot_001 and run_NNN)
into a single large dataset for Stage 3 LSTM training.

Features:
  - Discovers all run_NNN and pilot_NNN subdirectories in data/raw/sumo/
  - Prefixes vehicle IDs with run_id (run_id__vehicle_id) to ensure uniqueness
  - Uses MobilityValidator.validate_timestamps_and_lifecycles for robust lifecycle tracking
  - Extracts 7 kinematic features over 10s history and computes 30s link-survival label
  - Discards incomplete / censored future horizons
  - Aggregates detailed dataset summary (records, vehicles, pairs, sequences, class dist)
  - Saves:
      data/processed/large_dataset/pairwise_sequences.npz
      data/processed/large_dataset/metadata.json
      data/processed/large_dataset/dataset_summary.json

Do NOT modify Stage 1 or Stage 2 scripts or outputs.
Do NOT overwrite original processed dirs (data/processed, data/processed_full_trace).
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Set

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.validator import MobilityValidator
from src.features.sequence_builder import build_feature_sequences

COMM_RANGE_M = 1000.0
HISTORY_WINDOW_S = 10
PREDICTION_HORIZON_S = 30
STEP_S = 1.0


def process_one_run(run_dir: Path, run_id: str) -> tuple:
    mobility_csv = run_dir / "mobility.csv"
    cfg_path = run_dir / "run_config.json"

    if not mobility_csv.exists():
        print(f"    [SKIP] {run_id}: mobility.csv not found.")
        return None, None, None, None

    run_cfg = {}
    if cfg_path.exists():
        try:
            run_cfg = json.loads(cfg_path.read_text())
        except Exception:
            pass

    if run_cfg.get("status") == "error":
        print(f"    [SKIP] {run_id}: run status is error.")
        return None, None, None, None

    try:
        df_raw = pd.read_csv(mobility_csv)
        required_cols = {"time_s", "vehicle_id", "x", "y", "speed_mps"}
        missing = required_cols - set(df_raw.columns)
        if missing:
            print(f"    [ERROR] {run_id}: missing columns {missing}")
            return None, None, None, None

        # Vehicle ID string format
        df_raw["vehicle_id"] = df_raw["vehicle_id"].astype(str)

        # Ensure role column exists
        if "role" not in df_raw.columns:
            print(f"    [ERROR] {run_id}: missing 'role' column.")
            return None, None, None, None

        unique_vids = df_raw["vehicle_id"].nunique()
        providers = df_raw[df_raw["role"] == "provider"]["vehicle_id"].nunique()
        consumers = df_raw[df_raw["role"] == "consumer"]["vehicle_id"].nunique()

        if providers == 0 or consumers == 0:
            print(f"    [SKIP] {run_id}: insufficient providers/consumers (PUE: {providers}, CUE: {consumers}).")
            return None, None, None, None

        # Prefix vehicle IDs with run_id for unique identity across runs
        df_raw["vehicle_id"] = df_raw["vehicle_id"].apply(lambda v: f"{run_id}__{v}")

        # Validate timestamps and extract lifecycles
        lifecycles = MobilityValidator.validate_timestamps_and_lifecycles(df_raw)

        # Build feature sequences
        X, y, meta = build_feature_sequences(
            df=df_raw,
            lifecycles=lifecycles,
            comm_range_m=COMM_RANGE_M,
            history_window_s=HISTORY_WINDOW_S,
            prediction_horizon_s=PREDICTION_HORIZON_S,
            step_s=STEP_S,
        )

        if X is None or len(X) == 0:
            print(f"    [SKIP] {run_id}: no valid sequences produced.")
            return None, None, None, None

        # Annotate metadata with run_id
        for m in meta:
            m["run_id"] = run_id

        n_pos = int(np.sum(y == 1))
        n_neg = int(np.sum(y == 0))
        unique_pairs = len(set(m["pair_id"] for m in meta))

        run_stats = {
            "run_id": run_id,
            "mobility_records": len(df_raw),
            "unique_vehicles": unique_vids,
            "providers": providers,
            "consumers": consumers,
            "eligible_pairs": unique_pairs,
            "sequences": len(X),
            "positive_surviving": n_pos,
            "negative_breaking": n_neg,
        }

        print(f"    ✓ {run_id}: {len(X):,} sequences (pos={n_pos}, neg={n_neg}) | {unique_pairs} pairs | {len(df_raw):,} records")
        return X, y, meta, run_stats

    except Exception as e:
        print(f"    [ERROR] {run_id}: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None


def main() -> None:
    parser = argparse.ArgumentParser(description="Build combined large dataset from SUMO runs.")
    parser.add_argument("--sumo_dir", type=str, default="data/raw/sumo",
                        help="Directory containing run_NNN and pilot_NNN subdirectories.")
    parser.add_argument("--out_dir", type=str, default="data/processed/large_dataset",
                        help="Output directory for combined dataset.")
    args = parser.parse_args()

    sumo_dir = PROJECT_ROOT / args.sumo_dir
    out_dir = PROJECT_ROOT / args.out_dir

    forbidden = {
        (PROJECT_ROOT / "data/processed").resolve(),
        (PROJECT_ROOT / "data/processed_full_trace").resolve(),
    }
    if out_dir.resolve() in forbidden:
        print("ERROR: out_dir points to an existing baseline processed directory. Aborting.")
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)

    # Discover run_NNN and pilot_NNN directories
    run_dirs = sorted([
        d for d in sumo_dir.iterdir()
        if d.is_dir() and (d.name.startswith("run_") or d.name.startswith("pilot_"))
    ])

    if not run_dirs:
        print(f"ERROR: No simulation directories found in {sumo_dir}")
        sys.exit(1)

    print("=" * 70)
    print(" COMBINING MULTI-RUN DATASETS")
    print(f" Found {len(run_dirs)} simulation run directories in {sumo_dir}")
    print(f" Target output: {out_dir}")
    print("=" * 70)

    all_X: List[np.ndarray] = []
    all_y: List[np.ndarray] = []
    all_meta: List[Dict[str, Any]] = []
    per_run_stats: List[Dict[str, Any]] = []

    for run_dir in run_dirs:
        run_id = run_dir.name
        X, y, meta, stats = process_one_run(run_dir, run_id)
        if X is not None and len(X) > 0:
            all_X.append(X)
            all_y.append(y)
            all_meta.extend(meta)
            per_run_stats.append(stats)

    if not all_X:
        print("\nERROR: No valid sequences could be generated from any run.")
        sys.exit(1)

    X_comb = np.concatenate(all_X, axis=0)
    y_comb = np.concatenate(all_y, axis=0)

    total_seqs = len(X_comb)
    pos_cnt = int(np.sum(y_comb == 1))
    neg_cnt = int(np.sum(y_comb == 0))
    total_recs = sum(s["mobility_records"] for s in per_run_stats)
    total_vids = sum(s["unique_vehicles"] for s in per_run_stats)
    all_pairs = set(m["pair_id"] for m in all_meta)

    print("\n" + "=" * 70)
    print(" COMBINED DATASET SUMMARY")
    print("=" * 70)
    print(f" Total simulation runs included : {len(per_run_stats)}")
    print(f" Total mobility records         : {total_recs:,}")
    print(f" Total simulated vehicles       : {total_vids:,}")
    print(f" Total eligible PUE-CUE pairs   : {len(all_pairs):,}")
    print(f" Total valid LSTM sequences     : {total_seqs:,}")
    print(f"   Survival labels (1)          : {pos_cnt:,} ({100 * pos_cnt / total_seqs:.1f}%)")
    print(f"   Break labels (0)             : {neg_cnt:,} ({100 * neg_cnt / total_seqs:.1f}%)")
    print(f" Feature shape                  : {X_comb.shape}")

    # Save pairwise_sequences.npz
    npz_path = out_dir / "pairwise_sequences.npz"
    np.savez_compressed(str(npz_path), X=X_comb, y=y_comb)
    print(f"\n  Saved sequences: {npz_path}")

    # Save metadata.json
    meta_path = out_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(all_meta, f, indent=2)
    print(f"  Saved metadata:  {meta_path}")

    # Save dataset_summary.json
    pos_weight = round(neg_cnt / max(pos_cnt, 1), 4)
    summary = {
        "total_simulation_runs": len(per_run_stats),
        "total_mobility_records": total_recs,
        "total_vehicles": total_vids,
        "total_eligible_pairs": len(all_pairs),
        "total_valid_sequences": total_seqs,
        "survival_labels_positive": pos_cnt,
        "break_labels_negative": neg_cnt,
        "positive_class_ratio": round(pos_cnt / total_seqs, 4),
        "pos_weight_for_bce": pos_weight,
        "input_shape": list(X_comb.shape),
        "history_window_s": HISTORY_WINDOW_S,
        "prediction_horizon_s": PREDICTION_HORIZON_S,
        "comm_range_m": COMM_RANGE_M,
        "step_length_s": STEP_S,
        "per_run_statistics": per_run_stats
    }
    summary_path = out_dir / "dataset_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"  Saved summary:   {summary_path}")


if __name__ == "__main__":
    main()
