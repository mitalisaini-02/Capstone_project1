"""
Online construction of LSTM input windows for arbitrary (time, PUE, CUE) rows.

Uses the UNCHANGED training preprocessing: `precompute_vehicle_kinematics` from
src/features/sequence_builder.py and the same per-step kinematic functions, in the same
feature order. A window covers [t - (W-1), t] at 1 s steps; rows without a complete history
return no window (they are not eligible for an LSTM prediction).

Equivalence with the training windows is checked bitwise in tests and in the Phase 3 validation.
"""
from typing import List, Tuple

import numpy as np
import pandas as pd

from src.features.kinematics import (euclidean_distance, radial_closing_speed,
                                     relative_displacement, relative_speed)
from src.features.sequence_builder import precompute_vehicle_kinematics


class OnlineFeatureBuilder:
    def __init__(self, mobility_df: pd.DataFrame, history_window_s: int = 10, step_s: float = 1.0):
        cols = {"time_s", "vehicle_id", "x", "y", "speed_mps"}
        if not cols <= set(mobility_df.columns):
            raise ValueError(f"mobility_df needs columns {sorted(cols)}")
        self.W = int(history_window_s)
        self.step = float(step_s)
        self.pos, self.vel, self.spd = precompute_vehicle_kinematics(mobility_df)

    def window(self, t: float, pid, cid):
        """(W, 7) float32 window or None if any history step is missing."""
        t = round(float(t), 2)
        steps = []
        for i in range(self.W):
            tau = round(t - (self.W - 1 - i) * self.step, 2)
            p_pos, c_pos = self.pos.get((tau, pid)), self.pos.get((tau, cid))
            p_vel, c_vel = self.vel.get((tau, pid)), self.vel.get((tau, cid))
            p_spd, c_spd = self.spd.get((tau, pid)), self.spd.get((tau, cid))
            if any(v is None for v in (p_pos, c_pos, p_vel, c_vel, p_spd, c_spd)):
                return None
            d = euclidean_distance(p_pos, c_pos)
            r_spd = relative_speed(p_vel, c_vel)
            dx, dy = relative_displacement(p_pos, c_pos)
            closing = radial_closing_speed(p_pos, c_pos, p_vel, c_vel)
            steps.append([d, r_spd, p_spd, c_spd, dx, dy, closing])
        return np.array(steps, dtype=np.float32)

    def windows(self, times, pues, cues) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (windows (M, W, 7), row_mask (N,)) for the rows that have a full history."""
        out: List[np.ndarray] = []
        mask = np.zeros(len(times), dtype=bool)
        for k, (t, p, c) in enumerate(zip(times, pues, cues)):
            w = self.window(t, p, c)
            if w is not None:
                out.append(w)
                mask[k] = True
        arr = np.stack(out) if out else np.empty((0, self.W, 7), dtype=np.float32)
        return arr, mask
