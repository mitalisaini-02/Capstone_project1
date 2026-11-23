"""
Sequence building module: extracts multi-timestep temporal feature arrays and labels.
"""
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd

from src.features.kinematics import (
    euclidean_distance,
    estimate_velocity_vector,
    relative_displacement,
    relative_speed,
    radial_closing_speed
)
from src.features.labels import compute_link_survival_label


def precompute_vehicle_kinematics(
    df: pd.DataFrame
) -> Tuple[
    Dict[Tuple[float, str], Tuple[float, float]],
    Dict[Tuple[float, str], Tuple[float, float]],
    Dict[Tuple[float, str], float]
]:
    """
    Precomputes indexed positions, estimated velocity vectors, and speeds.
    Keys are (time_s: float, vehicle_id: str) to support both numeric and
    prefixed vehicle IDs (e.g., 'run_000__30' used in multi-run datasets).
    Returns: (pos_lookup, vel_lookup, speed_lookup)
    """
    pos_lookup: Dict[Tuple[float, str], Tuple[float, float]] = {}
    vel_lookup: Dict[Tuple[float, str], Tuple[float, float]] = {}
    speed_lookup: Dict[Tuple[float, str], float] = {}

    for vid, group in df.groupby("vehicle_id"):
        vid_str = str(vid)
        vid_int = int(vid) if vid_str.isdigit() else None
        sorted_g = group.sort_values("time_s").reset_index(drop=True)
        for idx, row in sorted_g.iterrows():
            t = round(float(row["time_s"]), 2)
            pos_curr = (float(row["x"]), float(row["y"]))
            speed = float(row["speed_mps"])

            pos_lookup[(t, vid_str)] = pos_curr
            speed_lookup[(t, vid_str)] = speed
            if vid_int is not None:
                pos_lookup[(t, vid_int)] = pos_curr
                speed_lookup[(t, vid_int)] = speed

            if idx == 0:
                # First observation: velocity direction approximated as stationary or zero
                vel_lookup[(t, vid_str)] = (0.0, 0.0)
                if vid_int is not None:
                    vel_lookup[(t, vid_int)] = (0.0, 0.0)
            else:
                prev_row = sorted_g.iloc[idx - 1]
                t_prev = round(float(prev_row["time_s"]), 2)
                pos_prev = (float(prev_row["x"]), float(prev_row["y"]))
                dt = t - t_prev
                vel_vec = estimate_velocity_vector(pos_curr, pos_prev, dt, speed_mps=speed)
                vel_lookup[(t, vid_str)] = vel_vec
                if vid_int is not None:
                    vel_lookup[(t, vid_int)] = vel_vec

    return pos_lookup, vel_lookup, speed_lookup


def build_feature_sequences(
    df: pd.DataFrame,
    lifecycles: Dict[int, Dict[str, Any]],
    comm_range_m: float = 1000.0,
    history_window_s: int = 10,
    prediction_horizon_s: int = 30,
    step_s: float = 1.0
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    """
    Extracts historical sequence features and binary link-survival labels.

    Returns:
      X: ndarray of shape (N, history_window_s, 7)
      y: ndarray of shape (N,)
      metadata: List of dicts with sample info (time_s, provider_id, consumer_id, pair_id)
    """
    pos_lookup, vel_lookup, speed_lookup = precompute_vehicle_kinematics(df)

    providers = sorted(df[df["role"] == "provider"]["vehicle_id"].unique())
    consumers = sorted(df[df["role"] == "consumer"]["vehicle_id"].unique())

    all_times = sorted(df["time_s"].unique())

    samples_X: List[np.ndarray] = []
    samples_y: List[int] = []
    samples_meta: List[Dict[str, Any]] = []

    for t_curr in all_times:
        t_curr = round(float(t_curr), 2)

        # We need at least history_window_s of past observations
        hist_timestamps = [round(t_curr - (history_window_s - 1 - i) * step_s, 2) for i in range(history_window_s)]
        if hist_timestamps[0] < all_times[0]:
            continue

        for cid in consumers:
            cpos_curr = pos_lookup.get((t_curr, cid))
            if cpos_curr is None:
                continue

            for pid in providers:
                ppos_curr = pos_lookup.get((t_curr, pid))
                if ppos_curr is None:
                    continue

                curr_dist = euclidean_distance(ppos_curr, cpos_curr)
                # Only candidate pairs within communication range are relevant
                if curr_dist > comm_range_m:
                    continue

                # 1. Compute target ground truth label
                label = compute_link_survival_label(
                    provider_id=pid,
                    consumer_id=cid,
                    current_time_s=t_curr,
                    future_horizon_s=float(prediction_horizon_s),
                    comm_range_m=comm_range_m,
                    lifecycles=lifecycles,
                    trajectory_lookup=pos_lookup,
                    step_s=step_s
                )

                if label is None:
                    # Censored: vehicles depart or lack full 30s horizon
                    continue

                # 2. Extract historical sequence over hist_timestamps
                seq_features = []
                has_missing_history = False

                for tau in hist_timestamps:
                    p_pos = pos_lookup.get((tau, pid))
                    c_pos = pos_lookup.get((tau, cid))
                    p_vel = vel_lookup.get((tau, pid))
                    c_vel = vel_lookup.get((tau, cid))
                    p_spd = speed_lookup.get((tau, pid))
                    c_spd = speed_lookup.get((tau, cid))

                    if any(v is None for v in (p_pos, c_pos, p_vel, c_vel, p_spd, c_spd)):
                        has_missing_history = True
                        break

                    d = euclidean_distance(p_pos, c_pos)
                    r_spd = relative_speed(p_vel, c_vel)
                    dx, dy = relative_displacement(p_pos, c_pos)
                    c_spd_rad = radial_closing_speed(p_pos, c_pos, p_vel, c_vel)

                    # 7 features
                    step_feat = [d, r_spd, p_spd, c_spd, dx, dy, c_spd_rad]
                    seq_features.append(step_feat)

                if has_missing_history:
                    continue

                samples_X.append(np.array(seq_features, dtype=np.float32))
                samples_y.append(int(label))
                samples_meta.append({
                    "time_s": t_curr,
                    "provider_id": int(pid) if str(pid).isdigit() else str(pid),
                    "consumer_id": int(cid) if str(cid).isdigit() else str(cid),
                    "pair_id": f"{pid}_{cid}",
                    "current_distance_m": curr_dist,
                    "label": int(label)
                })

    if not samples_X:
        return np.empty((0, history_window_s, 7), dtype=np.float32), np.empty((0,), dtype=np.int64), []

    X = np.stack(samples_X, axis=0)
    y = np.array(samples_y, dtype=np.int64)
    return X, y, samples_meta
