"""
build_combined_dataset.py
--------------------------
Combines multiple SUMO run mobility CSVs into a single large dataset.

For each run in data/raw/sumo/run_NNN/:
  - Loads mobility.csv
  - Validates it
  - Prefixes vehicle IDs with run_id to prevent cross-run collisions
  - Calls build_feature_sequences() per run
  - Appends run_id to each metadata entry
  - Concatenates all sequences

Saves:
  data/processed/large_dataset/pairwise_sequences.npz
  data/processed/large_dataset/metadata.json
  data/processed/large_dataset/dataset_summary.json

Usage:
  python scripts/build_combined_dataset.py [--sumo_dir data/raw/sumo]
                                            [--out_dir data/processed/large_dataset]
                                            [--max_runs N]

Constraints:
  - Does NOT overwrite data/processed/ or data/processed_full_trace/ (original datasets)
  - Does NOT modify Stage 1 or Stage 2
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import load_mobility_data
from src.data.validator import MobilityValidator
from src.features.sequence_builder import build_feature_sequences


# ---------------------------------------------------------------------------
# Parameters (must match Stage 3 config)
# ---------------------------------------------------------------------------
COMM_RANGE_M = 1000.0
HISTORY_WINDOW_S = 10
PREDICTION_HORIZON_S = 30
STEP_S = 1.0

# Provider/consumer split: vehicle IDs with int(id) < threshold → provider
# This mirrors generate_sumo_runs.py's role assignment (60% providers)
def _assign_roles_in_df(df: pd.DataFrame, n_providers_frac: float = 0.60) -> pd.DataFrame:
    """
    Re-derive roles from vehicle IDs in this run.
    Vehicle IDs are strings like '0', '1', ... or 'trip_0', etc.
    We sort numeric IDs; first 60% → provider, rest → consumer.
    Non-numeric IDs are sorted lexicographically.
    """
    unique_ids = sorted(df["vehicle_id"].unique(), key=lambda v: (int(v) if v.isdigit() else float('inf'), v))
    n_prov = max(1, int(len(unique_ids) * n_providers_frac))
    provider_set = set(unique_ids[:n_prov])
    df = df.copy()
    df["role"] = df["vehicle_id"].apply(lambda v: "provider" if v in provider_set else "consumer")
    return df


def process_one_run(
    run_dir: Path,
    run_id: str,
) -> tuple:
    """
    Load, validate, build sequences for one run.
    Returns (X, y, metadata) or (None, None, None) on failure.
    """
    mobility_csv = run_dir / "mobility.csv"
    cfg_path = run_dir / "run_config.json"

    if not mobility_csv.exists():
        print(f"    [SKIP] {run_id}: mobility.csv not found.")
        return None, None, None

    # Load run config
    run_cfg = {}
    if cfg_path.exists():
        try:
            run_cfg = json.loads(cfg_path.read_text())
        except Exception:
            pass

    if run_cfg.get("status") not in ("done", None, ""):
        if run_cfg.get("status") == "error":
            print(f"    [SKIP] {run_id}: previously failed (status=error).")
            return None, None, None

    try:
        # Load raw mobility
        df_raw = pd.read_csv(mobility_csv)
        required_cols = {"time_s", "vehicle_id", "x", "y", "speed_mps"}
        missing = required_cols - set(df_raw.columns)
        if missing:
            print(f"    [ERROR] {run_id}: missing columns {missing}")
            return None, None, None

        # Ensure string vehicle IDs
        df_raw["vehicle_id"] = df_raw["vehicle_id"].astype(str)

        # Re-assign roles if 'role' column is missing or unreliable for new runs
        if "role" not in df_raw.columns:
            df_raw = _assign_roles_in_df(df_raw)
        else:
            # Validate that both roles are present
            roles_present = set(df_raw["role"].unique())
            if not {"provider", "consumer"}.issubset(roles_present):
                df_raw = _assign_roles_in_df(df_raw)

        providers = df_raw[df_raw["role"] == "provider"]["vehicle_id"].nunique()
        consumers = df_raw[df_raw["role"] == "consumer"]["vehicle_id"].nunique()
        if providers == 0 or consumers == 0:
            print(f"    [SKIP] {run_id}: no providers or consumers (p={providers}, c={consumers}).")
            return None, None, None

        # Prefix vehicle IDs with run_id to prevent cross-run collision
        df_raw["vehicle_id"] = df_raw["vehicle_id"].apply(lambda v: f"{run_id}__{v}")

        # Validate (basic checks)
        validator = MobilityValidator()
        issues = validator.validate(df_raw)
        hard_issues = [iss for iss in issues if iss.get("severity", "warning") == "error"]
        if hard_issues:
            print(f"    [WARN] {run_id}: {len(hard_issues)} validation errors (continuing anyway).")
            for iss in hard_issues[:3]:
                print(f"      {iss}")

        # Build lifecycles (vehicle first/last time)
        lifecycles: Dict[Any, Dict[str, Any]] = {}
        for vid, grp in df_raw.groupby("vehicle_id"):
            lifecycles[int(vid.split("__", 1)[1]) if "__" in vid else vid] = {
                "first_time": float(grp["time_s"].min()),
                "last_time": float(grp["time_s"].max()),
            }
        # Use prefixed vid as key
        lifecycles_prefixed: Dict[str, Dict[str, Any]] = {}
        for vid, grp in df_raw.groupby("vehicle_id"):
            lifecycles_prefixed[vid] = {
                "first_time": float(grp["time_s"].min()),
                "last_time": float(grp["time_s"].max()),
            }

        # Build sequences
        X, y, meta = build_feature_sequences(
            df=df_raw,
            lifecycles=lifecycles_prefixed,
            comm_range_m=COMM_RANGE_M,
            history_window_s=HISTORY_WINDOW_S,
            prediction_horizon_s=PREDICTION_HORIZON_S,
            step_s=STEP_S,
        )

        if X is None or len(X) == 0:
            print(f"    [SKIP] {run_id}: no valid sequences produced.")
            return None, None, None

        # Inject run_id into metadata
        for m in meta:
            m["run_id"] = run_id

        n_pos = int(np.sum(y == 1))
        n_neg = int(np.sum(y == 0))
        print(f"    ✓ {run_id}: {len(X):,} sequences | pos={n_pos} neg={n_neg} "
              f"| p={providers} providers c={consumers} consumers")
        return X, y, meta

    except Exception as e:
        print(f"    [ERROR] {run_id}: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


def main() -> None:
    parser = argparse.ArgumentParser(description="Build combined large dataset from SUMO runs.")
    parser.add_argument("--sumo_dir", type=str, default="data/raw/sumo",
                        help="Directory containing run_NNN subdirectories.")
    parser.add_argument("--out_dir", type=str, default="data/processed/large_dataset",
                        help="Output directory for combined dataset.")
    parser.add_argument("--max_runs", type=int, default=None,
                        help="Max number of runs to include (default: all).")
    args = parser.parse_args()

    sumo_dir = PROJECT_ROOT / args.sumo_dir
    out_dir = PROJECT_ROOT / args.out_dir

    # Safety: never write to original processed dirs
    forbidden = {
        (PROJECT_ROOT / "data/processed").resolve(),
        (PROJECT_ROOT / "data/processed_full_trace").resolve(),
    }
    if out_dir.resolve() in forbidden:
        print("ERROR: out_dir points to an original processed directory. Aborting.")
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)

    # Discover all run directories
    run_dirs = sorted([d for d in sumo_dir.iterdir() if d.is_dir() and d.name.startswith("run_")])
    if not run_dirs:
        print(f"ERROR: No run_NNN directories found in {sumo_dir}")
        sys.exit(1)

    if args.max_runs:
        run_dirs = run_dirs[: args.max_runs]

    print(f"Found {len(run_dirs)} run directories in {sumo_dir}")
    print(f"Output: {out_dir}\n")

    all_X: List[np.ndarray] = []
    all_y: List[np.ndarray] = []
    all_meta: List[Dict[str, Any]] = []
    run_summaries: List[Dict[str, Any]] = []

    for run_dir in run_dirs:
        run_id = run_dir.name
        print(f"  Processing {run_id}...")
        X, y, meta = process_one_run(run_dir, run_id)
        if X is not None:
            all_X.append(X)
            all_y.append(y)
            all_meta.extend(meta)
            run_summaries.append({
                "run_id": run_id,
                "sequences": len(X),
                "positive": int(np.sum(y == 1)),
                "negative": int(np.sum(y == 0)),
            })
        else:
            run_summaries.append({"run_id": run_id, "sequences": 0, "status": "failed_or_skipped"})

    if not all_X:
        print("\nERROR: No valid sequences from any run. Aborting.")
        sys.exit(1)

    # Concatenate
    X_combined = np.concatenate(all_X, axis=0)
    y_combined = np.concatenate(all_y, axis=0)

    total = len(X_combined)
    n_pos = int(np.sum(y_combined == 1))
    n_neg = int(np.sum(y_combined == 0))

    print(f"\n{'='*60}")
    print(f"COMBINED DATASET")
    print(f"{'='*60}")
    print(f"  Total sequences : {total:,}")
    print(f"  Positive (1)    : {n_pos:,}  ({100*n_pos/total:.1f}%)")
    print(f"  Negative (0)    : {n_neg:,}  ({100*n_neg/total:.1f}%)")
    print(f"  Shape X         : {X_combined.shape}")
    print(f"  Runs included   : {len([r for r in run_summaries if r.get('sequences', 0) > 0])}")

    # Save .npz
    npz_path = out_dir / "pairwise_sequences.npz"
    np.savez_compressed(str(npz_path), X=X_combined, y=y_combined)
    print(f"\n  Saved: {npz_path}")

    # Save metadata JSON
    meta_path = out_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(all_meta, f, indent=2)
    print(f"  Saved: {meta_path}")

    # Save summary
    pos_weight = round(n_neg / max(n_pos, 1), 4)
    summary = {
        "total_sequences": total,
        "positive_samples": n_pos,
        "negative_samples": n_neg,
        "pos_fraction": round(n_pos / total, 4),
        "pos_weight_for_bce": pos_weight,
        "feature_shape": list(X_combined.shape[1:]),
        "feature_dim": X_combined.shape[2] if X_combined.ndim == 3 else None,
        "history_window_s": HISTORY_WINDOW_S,
        "prediction_horizon_s": PREDICTION_HORIZON_S,
        "comm_range_m": COMM_RANGE_M,
        "runs_included": len([r for r in run_summaries if r.get("sequences", 0) > 0]),
        "run_summaries": run_summaries,
    }
    summary_path = out_dir / "dataset_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"  Saved: {summary_path}")
    print(f"\n  pos_weight for BCE loss = {pos_weight}")


if __name__ == "__main__":
    main()
