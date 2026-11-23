"""
inspect_pilot_sequences.py
---------------------------
Runs Stage 3 feature-sequence construction on the pilot_001 mobility trace
and reports detailed statistics WITHOUT training any model.

Reports:
  - Vehicle counts, timestamps, role distribution
  - Valid PUE-CUE pairs within range
  - Total sequences built
  - Positive / negative / censored label counts
  - Per-pair label statistics
  - Any validation errors

Does NOT modify any existing dataset, checkpoint, or model.
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import load_mobility_data
from src.data.validator import MobilityValidator
from src.features.sequence_builder import build_feature_sequences

PILOT_DIR    = PROJECT_ROOT / "data" / "raw" / "sumo" / "pilot_001"
MOBILITY_CSV = PILOT_DIR / "mobility.csv"

COMM_RANGE_M        = 1000.0
HISTORY_WINDOW_S    = 10
PREDICTION_HORIZON_S = 30
STEP_S              = 1.0


def main():
    print("=" * 70)
    print(" PILOT 001 — Stage 3 Sequence Inspection")
    print("=" * 70)

    if not MOBILITY_CSV.exists():
        print(f"ERROR: {MOBILITY_CSV} not found. Run extract_pilot_mobility.py first.")
        sys.exit(1)

    # --- 1. Load raw CSV directly (no ue_roles.csv needed; roles in CSV)
    df = pd.read_csv(MOBILITY_CSV)
    df["vehicle_id"] = df["vehicle_id"].astype(str)

    print(f"\n[1] Raw CSV stats")
    print(f"    Records        : {len(df):,}")
    print(f"    Unique vehicles: {df['vehicle_id'].nunique()}")
    print(f"    Time range     : {df['time_s'].min():.0f}s – {df['time_s'].max():.0f}s")
    print(f"    Unique times   : {df['time_s'].nunique()}")
    role_counts = df.groupby("role")["vehicle_id"].nunique()
    for role, cnt in role_counts.items():
        print(f"    {role:10s}: {cnt} unique vehicles")

    # --- 2. Validate
    print(f"\n[2] Validation")
    validator = MobilityValidator()
    errors = []
    warnings = []
    try:
        validator.validate_schema(df)
        print(f"    Schema     : OK")
    except Exception as e:
        errors.append(str(e))
        print(f"    Schema     : ERROR — {e}")
    try:
        bounds = validator.validate_coordinates(df)
        print(f"    Coordinates: OK  x=[{bounds['x'][0]:.0f},{bounds['x'][1]:.0f}] y=[{bounds['y'][0]:.0f},{bounds['y'][1]:.0f}]")
    except Exception as e:
        warnings.append(str(e))
        print(f"    Coordinates: WARNING — {e}")
    try:
        validator.validate_timestamps_and_lifecycles(df)
        print(f"    Timestamps : OK")
    except Exception as e:
        warnings.append(str(e))
        print(f"    Timestamps : WARNING — {e}")
    print(f"    Total errors: {len(errors)}  warnings: {len(warnings)}")

    # --- 3. Build lifecycles
    print(f"\n[3] Vehicle lifecycles")
    lifecycles = MobilityValidator.validate_timestamps_and_lifecycles(df)
    depart_times = [v["start_t"] for v in lifecycles.values()]
    print(f"    First vehicle at t={min(depart_times):.0f}s")
    print(f"    Last vehicle at  t={max(depart_times):.0f}s")

    # --- 4. Build sequences
    print(f"\n[4] Building feature sequences")
    print(f"    H={HISTORY_WINDOW_S}s  T={PREDICTION_HORIZON_S}s  R={COMM_RANGE_M}m")

    X, y, metadata = build_feature_sequences(
        df=df,
        lifecycles=lifecycles,
        comm_range_m=COMM_RANGE_M,
        history_window_s=HISTORY_WINDOW_S,
        prediction_horizon_s=PREDICTION_HORIZON_S,
        step_s=STEP_S,
    )

    if X is None or len(X) == 0:
        print("    WARNING: No sequences produced!")
        sys.exit(0)

    n_total = len(X)
    n_pos   = int(np.sum(y == 1))
    n_neg   = int(np.sum(y == 0))
    print(f"\n    Total sequences : {n_total:,}")
    print(f"    Positive (1)    : {n_pos:,}  ({100*n_pos/n_total:.1f}%)")
    print(f"    Negative (0)    : {n_neg:,}  ({100*n_neg/n_total:.1f}%)")
    print(f"    Feature shape   : {X.shape}")

    # --- 5. Per-pair stats
    print(f"\n[5] Per-pair label statistics")
    pair_stats = defaultdict(lambda: {"total": 0, "pos": 0, "neg": 0})
    for m, label in zip(metadata, y):
        pid = m["pair_id"]
        pair_stats[pid]["total"] += 1
        if label == 1:
            pair_stats[pid]["pos"] += 1
        else:
            pair_stats[pid]["neg"] += 1

    print(f"    Unique PUE-CUE pairs observed: {len(pair_stats)}")

    pure_pos  = sum(1 for s in pair_stats.values() if s["neg"] == 0)
    pure_neg  = sum(1 for s in pair_stats.values() if s["pos"] == 0)
    mixed     = sum(1 for s in pair_stats.values() if s["pos"] > 0 and s["neg"] > 0)
    print(f"    Pure-positive pairs (always survive)  : {pure_pos}")
    print(f"    Pure-negative pairs (always break)    : {pure_neg}")
    print(f"    Mixed pairs (transitions seen)        : {mixed}")

    # Top 5 pairs by sequence count
    top5 = sorted(pair_stats.items(), key=lambda kv: -kv[1]["total"])[:5]
    print(f"\n    Top-5 pairs by sequence count:")
    for pid, s in top5:
        print(f"      {pid:30s}  total={s['total']:4d}  pos={s['pos']:4d}  neg={s['neg']:4d}")

    # --- 6. Feature sanity check
    print(f"\n[6] Feature sanity (sample from X[0])")
    feat_names = ["distance_m", "relative_speed_mps", "provider_speed_mps",
                  "consumer_speed_mps", "delta_x_m", "delta_y_m", "closing_speed_mps"]
    last_step = X[0, -1, :]  # last timestep of first sequence
    for name, val in zip(feat_names, last_step):
        print(f"    {name:25s} = {val:.4f}")

    nan_count = int(np.sum(np.isnan(X)))
    inf_count = int(np.sum(np.isinf(X)))
    print(f"\n    NaN in X : {nan_count}")
    print(f"    Inf in X : {inf_count}")

    print(f"\n{'='*70}")
    print(f" PILOT 001 SUMMARY")
    print(f"{'='*70}")
    print(f"  Records        : {len(df):,}")
    print(f"  Sequences      : {n_total:,}")
    print(f"  Positive       : {n_pos:,}  ({100*n_pos/n_total:.1f}%)")
    print(f"  Negative       : {n_neg:,}  ({100*n_neg/n_total:.1f}%)")
    print(f"  Unique pairs   : {len(pair_stats)}")
    print(f"  Mixed pairs    : {mixed}")
    if nan_count == 0 and inf_count == 0:
        print(f"  Feature quality: OK (no NaN/Inf)")
    else:
        print(f"  Feature quality: ISSUES — NaN={nan_count}, Inf={inf_count}")

    # Save a quick report JSON
    report = {
        "run_id": "pilot_001",
        "records": len(df),
        "sequences": n_total,
        "positive": n_pos,
        "negative": n_neg,
        "pos_fraction": round(n_pos / n_total, 4),
        "unique_pairs": len(pair_stats),
        "pure_positive_pairs": pure_pos,
        "pure_negative_pairs": pure_neg,
        "mixed_pairs": mixed,
        "nan_count": nan_count,
        "inf_count": inf_count,
        "shape_X": list(X.shape),
    }
    report_path = PILOT_DIR / "sequence_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"\n  Report saved: {report_path}")


if __name__ == "__main__":
    main()
